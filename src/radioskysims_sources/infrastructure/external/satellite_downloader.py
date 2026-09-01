# infrastructure/external/satellite_downloader.py
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, ClassVar

import httpx
import pandas as pd
from dotenv import load_dotenv
from radioskysims_core.config.config_loader import load_config
from radioskysims_core.logging_config import get_logger, log_http_event
from radioskysims_core.paths import get_project_root

from radioskysims_sources.domain.satellite.prn import parse_prn_from_name

# Carrega credenciais locais (gitignored) para o ambiente, sem sobrescrever
# variáveis já exportadas. O arquivo segue o modelo ``credentials.env.example``
# e NUNCA deve ser versionado (ver .gitignore).
load_dotenv(get_project_root() / "credentials.env", override=False)

logger = logging.getLogger(__name__)
slogger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class CelestrakClient:
    """A client for fetching GNSS satellite data from CelesTrak.

    This client provides methods to retrieve and parse GNSS satellite information
    from the CelesTrak website, returning the data in a structured pandas DataFrame
    format.

    Attributes:
        base_url (str): The base URL for CelesTrak GNSS satellite data.

    Example:
        >>> client = CelestrakClient()
        >>> sats_df = client.fetch_sats()
        >>> print(sats_df.head())
    """

    def __init__(self) -> None:
        self.base_url = "https://celestrak.org/NORAD/elements/table.php?GROUP=gnss"

    def fetch_sats(self, url: str | None = None) -> pd.DataFrame:
        """Fetch GNSS satellite data from CelesTrak.

        Args:
            url (str, optional): The URL to fetch data from. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing GNSS satellite data.
        """
        if not url:
            url = self.base_url
        with httpx.Client() as client:
            start = time.perf_counter()
            response = client.get(url)
            duration_ms = (time.perf_counter() - start) * 1000
            log_http_event(
                slogger,
                "celestrak_fetch_sats",
                method="GET",
                url=url,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            if response.status_code == 200:
                sats_df = pd.read_html(response.content)[1]
            else:
                msg = "Failed to retrieve data from CelesTrak"
                raise ValueError(msg)
        cols = [
            "International Designator",
            "NORAD Catalog Number",
            "Name",
            "Period (minutes)",
            "Inclination (degrees)",
            "Apogee Height (km)",
            "Perigee Height (km)",
            "Eccentricity",
            "GP Age (days)",
        ]
        sats_df = sats_df[cols]
        sats_df.columns = [
            "id",
            "norad_id",
            "name",
            "period",
            "inclination",
            "apogee",
            "perigee",
            "eccentricity",
            "gp_age",
        ]
        sats_df["constellation"] = sats_df.name.str.extract(r"^([A-Za-z]+)(?=[\s\-0]|$)")
        # Normaliza nomes de constelações para os padrões GNSS
        sats_df["constellation"] = (
            sats_df["constellation"].map(CONSTELLATION_ALIAS).fillna(sats_df["constellation"])
        )
        # PRN (Pseudo-Random Noise) — extraído do nome CelesTrak. É a ponte
        # com as efemérides RINEX (ex.: "GPS BIIR-5 (PRN 22)" → 22).
        sats_df["prn"] = [
            parse_prn_from_name(n, c)
            for n, c in zip(sats_df["name"], sats_df["constellation"], strict=False)
        ]
        return sats_df.dropna(subset=["constellation"])


# ── Mapa de aliases: nome Celestrak → nome GNSS padrão ────────
CONSTELLATION_ALIAS: dict[str, str] = {
    "GSAT": "GALILEO",
    "COSMOS": "GLONASS",
    "BEIDOU": "BEIDOU",
    "GPS": "GPS",
}


# space_track_client.py


@dataclass
class QueryResult:
    norad_id: int
    payload: dict[str, Any]


# Load settings from config (fallback)
settings = load_config(key="SPACETRACK")


class SpaceTrackClient:
    """
    Client to query Space-Track's gp_history endpoint.

    Features:
      * Credentials loaded via Pydantic from .env (see config.py).
      * Synchronous and asynchronous query methods.
      * Global (class-level) registry of NORAD IDs already queried today,
        so no NORAD ID is queried more than once per day across instances.
      * Thread-safe and async-safe access to the registry.
      * Simple rate-window counters (in-memory) for local rate-limiting.
      * Logging and docstrings.
    """

    BASE_URL = "https://www.space-track.org"
    API_URL = f"{BASE_URL}/basicspacedata/query/class/gp_history"

    # ----------------------------
    # Class-level registry (shared by all instances)
    # ----------------------------
    # Conjunto de (norad_id, epoch_key) consultados no dia corrente.
    # epoch_key = (epoch_start, epoch_end); permite re-consultar o mesmo NORAD ID
    # para uma janela de época diferente dentro do mesmo dia.
    _queried_norad_ids: ClassVar[set[tuple[int, tuple[str, str] | None]]] = set()
    _registry_day: ClassVar[date] = date.today()
    _registry_lock_thread: ClassVar[threading.Lock] = (
        threading.Lock()
    )  # protects _queried_norad_ids for sync code
    _registry_lock_async: ClassVar[asyncio.Lock] = (
        asyncio.Lock()
    )  # protects _queried_norad_ids for async code

    # ----------------------------
    # Instance initialization
    # ----------------------------
    def __init__(self, epoch: datetime | None = None) -> None:
        """
        Args:
            epoch: optional datetime used to set epoch_start and epoch_end window.
                   If None -> defaults used ("now-2" / "now") so Space-Track relative query works.
        """
        # Credenciais: prioriza ambiente (ST_USER/ST_PWD, ou o alias
        # SPACETRACK_USERNAME/SPACETRACK_PASSWORD de credentials.env), com
        # fallback para settings.toml (onde NÃO deve haver valores reais).
        self.username = (
            os.getenv("ST_USER") or os.getenv("SPACETRACK_USERNAME") or settings.get("ST_USER")
        )
        self.password = (
            os.getenv("ST_PWD") or os.getenv("SPACETRACK_PASSWORD") or settings.get("ST_PWD")
        )
        if not self.username or not self.password:
            msg = "Missing Space-Track credentials (ST_USER/ST_PWD)."
            raise ValueError(msg)

        self.epoch = epoch
        self.query_limit = 50
        self.batch_size = 50

        # Simple rate windows to avoid bursts (per-process in-memory)
        self._rate_window_minute: deque[float] = deque()
        self._rate_window_hour: deque[float] = deque()
        self.max_per_minute = 30
        self.max_per_hour = 300

        self._set_epoch()

    # ----------------------------
    # Epoch helpers
    # ----------------------------
    def _set_epoch(self) -> None:
        # Janela de ±5 dias em torno da época de referência. A janela antiga
        # (±2 dias) deixava de fora satélites cujo TLE mais recente caía fora
        # dela (ex.: satélites atualizados a cada poucos dias), resultando em
        # "0 TLEs" indevido no update_orbital_elements.
        if self.epoch is None:
            # Use the Space-Track relative syntax
            self.epoch_start = "now-5"
            self.epoch_end = "now"
        else:
            # Convert to YYYY-MM-DD
            self.epoch_start = (self.epoch - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            self.epoch_end = (self.epoch + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    # ----------------------------
    # Registry rotation and helpers
    # ----------------------------
    @classmethod
    def _rotate_registry_if_needed(cls) -> None:
        """Reset registry when day changes (synchronous part)."""
        today = date.today()
        if today != cls._registry_day:
            with cls._registry_lock_thread:
                # re-check inside lock
                if today != cls._registry_day:
                    cls._registry_day = today
                    cls._queried_norad_ids.clear()
                    logger.info("Registry rotated for new day: %s", today.isoformat())

    @classmethod
    async def _rotate_registry_if_needed_async(cls) -> None:
        """Reset registry when day changes (asynchronous part)."""
        today = date.today()
        if today != cls._registry_day:
            # acquire async lock -> upgrade to thread lock to mutate safely
            async with cls._registry_lock_async:
                with cls._registry_lock_thread:
                    if today != cls._registry_day:
                        cls._registry_day = today
                        cls._queried_norad_ids.clear()
                        logger.info(
                            "Registry rotated for new day (async): %s",
                            today.isoformat(),
                        )

    @classmethod
    def filter_new_sync(
        cls,
        ids: Iterable[int],
        *,
        epoch_key: tuple[str, str] | None = None,
    ) -> list[int]:
        """
        Synchronous filter: return only NORAD IDs not yet queried today for the
        given epoch window, and atomically mark them as queried.

        Parameters
        ----------
        ids:
            NORAD IDs a filtrar.
        epoch_key:
            Tupla ``(epoch_start, epoch_end)`` identificando a janela de época.
            Permite re-consultar o mesmo NORAD ID quando a janela muda, sem
            exceder o limite diário para a mesma janela.
        """
        cls._rotate_registry_if_needed()
        with cls._registry_lock_thread:
            new = []
            for i in ids:
                key = (i, epoch_key)
                if key not in cls._queried_norad_ids:
                    cls._queried_norad_ids.add(key)
                    new.append(i)
            return new

    @classmethod
    async def filter_new_async(
        cls,
        ids: Iterable[int],
        *,
        epoch_key: tuple[str, str] | None = None,
    ) -> list[int]:
        """
        Async filter: return only NORAD IDs not yet queried today for the given
        epoch window, and atomically mark them as queried.
        """
        await cls._rotate_registry_if_needed_async()
        async with cls._registry_lock_async:
            # Use thread lock to protect the actual set (same process)
            with cls._registry_lock_thread:
                new = []
                for i in ids:
                    key = (i, epoch_key)
                    if key not in cls._queried_norad_ids:
                        cls._queried_norad_ids.add(key)
                        new.append(i)
                return new

    # ----------------------------
    # Query builder
    # ----------------------------
    def _build_queries(self, norad_ids: list[int], decay: str | None = None) -> list[str]:
        """
        Build the Space-Track query strings, batching by self.query_limit.
        """
        decay_part = "null-val" if decay is None else decay

        # Remove duplicates in input order
        seen: set[int] = set()
        unique_ids: list[int] = []
        for x in norad_ids:
            if x not in seen:
                seen.add(x)
                unique_ids.append(x)

        chunks: list[list[int]] = [
            unique_ids[i : i + self.query_limit]
            for i in range(0, len(unique_ids), self.query_limit)
        ]

        queries: list[str] = []
        for chunk in chunks:
            ids_encoded = "%2C".join(map(str, chunk))
            q = (
                f"{self.API_URL}/NORAD_CAT_ID/{ids_encoded}/"
                f"EPOCH/>{self.epoch_start}%2C<{self.epoch_end}/DECAY_DATE/{decay_part}/"
                "orderby/EPOCH%20desc/emptyresult/show"
            )
            queries.append(q)
        return queries

    # ----------------------------
    # Rate limiting helpers (per-process)
    # ----------------------------
    def _prune_rate_windows(self) -> None:
        now = time.time()
        while self._rate_window_minute and now - self._rate_window_minute[0] > 60:
            self._rate_window_minute.popleft()
        while self._rate_window_hour and now - self._rate_window_hour[0] > 3600:
            self._rate_window_hour.popleft()

    async def _wait_rate_limit_async(self) -> None:
        self._prune_rate_windows()
        while (
            len(self._rate_window_minute) >= self.max_per_minute
            or len(self._rate_window_hour) >= self.max_per_hour
        ):
            logger.debug("Rate limit reached, waiting (async)...")
            await asyncio.sleep(0.5)
            self._prune_rate_windows()
        now = time.time()
        self._rate_window_minute.append(now)
        self._rate_window_hour.append(now)

    def _wait_rate_limit_sync(self) -> None:
        self._prune_rate_windows()
        while (
            len(self._rate_window_minute) >= self.max_per_minute
            or len(self._rate_window_hour) >= self.max_per_hour
        ):
            logger.debug("Rate limit reached, waiting (sync)...")
            time.sleep(0.5)
            self._prune_rate_windows()
        now = time.time()
        self._rate_window_minute.append(now)
        self._rate_window_hour.append(now)

    # ----------------------------
    # HTTP helpers (network calls)
    # ----------------------------
    async def _do_query_async(self, client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
        await self._wait_rate_limit_async()
        start = time.perf_counter()
        try:
            r = await client.get(query)
            r.raise_for_status()
            log_http_event(
                slogger,
                "spacetrack_query_async",
                method="GET",
                url=query,
                status_code=r.status_code,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return r.json()
        except Exception as exc:  # capture exceptions but don't raise to keep batch behavior
            log_http_event(
                slogger,
                "spacetrack_query_async_failed",
                method="GET",
                url=query,
                status_code=None,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )
            logger.error("Async query failed: %s -> %s", query, exc)
            return []

    def _do_query_sync(self, client: httpx.Client, query: str) -> list[dict[str, Any]]:
        self._wait_rate_limit_sync()
        start = time.perf_counter()
        try:
            r = client.get(query)
            r.raise_for_status()
            log_http_event(
                slogger,
                "spacetrack_query_sync",
                method="GET",
                url=query,
                status_code=r.status_code,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return r.json()
        except Exception as exc:
            log_http_event(
                slogger,
                "spacetrack_query_sync_failed",
                method="GET",
                url=query,
                status_code=None,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )
            logger.error("Sync query failed: %s -> %s", query, exc)
            return []

    # ----------------------------
    # Public methods — Async
    # ----------------------------
    async def fetch_elements(self, norad_ids: Iterable[int]) -> pd.DataFrame:
        """
        Asynchronously fetch orbital elements for the provided NORAD IDs.

        Returns a pandas.DataFrame. If all norad_ids were already queried today,
        returns an empty DataFrame.
        """
        requested = list(norad_ids)
        epoch_key = (self.epoch_start, self.epoch_end)
        new_ids = await self.filter_new_async(requested, epoch_key=epoch_key)
        if not new_ids:
            logger.info(
                "SpaceTrack (async): %d NORAD IDs já consultados para a janela de "
                "época %s..%s hoje (cache diário); usando dados existentes.",
                len(requested),
                self.epoch_start,
                self.epoch_end,
            )
            return pd.DataFrame()

        queries = self._build_queries(new_ids)
        async with httpx.AsyncClient(timeout=60) as client:
            # login
            login_url = f"{self.BASE_URL}/ajaxauth/login"
            msg = f"SpaceTrack query queuing. {queries}"
            logger.info(msg)

            try:
                await client.post(
                    login_url,
                    data={"identity": self.username, "password": self.password},
                )
            except Exception as exc:
                logger.error("Async login failed: %s", exc)
                # proceed — queries will probably fail, but keep flow

            # execute queries in batches (gather)
            tasks = [self._do_query_async(client, q) for q in queries]
            results = await asyncio.gather(*tasks)
        # flatten
        flattened: list[dict[str, Any]] = [item for sub in results for item in sub]
        return self._to_dataframe(flattened)

    # ----------------------------
    # Public methods — Sync
    # ----------------------------
    def fetch_elements_sync(self, norad_ids: Iterable[int]) -> pd.DataFrame:
        """
        Synchronously fetch orbital elements for the provided NORAD IDs.

        Returns a pandas.DataFrame. If all norad_ids were already queried today,
        returns an empty DataFrame.
        """
        requested = list(norad_ids)
        epoch_key = (self.epoch_start, self.epoch_end)
        new_ids = self.filter_new_sync(requested, epoch_key=epoch_key)
        if not new_ids:
            logger.info(
                "SpaceTrack (sync): %d NORAD IDs já consultados para a janela de "
                "época %s..%s hoje (cache diário); usando dados existentes.",
                len(requested),
                self.epoch_start,
                self.epoch_end,
            )
            return pd.DataFrame()

        queries = self._build_queries(new_ids)
        with httpx.Client(timeout=60) as client:
            login_url = f"{self.BASE_URL}/ajaxauth/login"
            msg = f"SpaceTrack query queuing. {queries}"
            logger.info(msg)
            try:
                client.post(
                    login_url,
                    data={"identity": self.username, "password": self.password},
                )
            except Exception as exc:
                logger.error("Sync login failed: %s", exc)
            results = [self._do_query_sync(client, q) for q in queries]
        flattened: list[dict[str, Any]] = [item for sub in results for item in sub]
        return self._to_dataframe(flattened)

    # ----------------------------
    # Utility: convert to DataFrame
    # ----------------------------
    def _to_dataframe(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Convert list-of-dicts to pandas DataFrame. Minimal conversion here:
        user can override or post-process columns as needed.
        """
        if not data:
            return pd.DataFrame()
        _df = pd.DataFrame.from_records(data)
        # Lower-case columns (common expectation)
        _df.columns = [c.lower() for c in _df.columns]
        # Attempt to cast common date fields to datetime
        for col in _df.columns:
            if ("date" in col) and ("decay" not in col):
                with contextlib.suppress(Exception):
                    _df[col] = pd.to_datetime(_df[col], errors="coerce")
                    _df[col] = _df[col].where(_df[col].notna(), None)
        # rename norad_cat_id -> norad_id if exists
        if "norad_cat_id" in _df.columns and "norad_id" not in _df.columns:
            _df = _df.rename(columns={"norad_cat_id": "norad_id"})
        return _df

    # ----------------------------
    # Administrative helpers
    # ----------------------------
    @classmethod
    def clear_registry(cls) -> None:
        """Clear global registry (useful for tests or manual reset)."""
        with cls._registry_lock_thread:
            cls._queried_norad_ids.clear()
            cls._registry_day = date.today()
            logger.info("Global query registry cleared manually.")
