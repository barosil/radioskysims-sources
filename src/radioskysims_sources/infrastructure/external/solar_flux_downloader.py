"""\
Módulo: src/radioskysims/infrastructure/external/solar_flux_downloader.py

Descrição:
    Adapter externo para download e parsing de arquivos SRD/SRS de fluxo solar.
    Fonte: https://downloads.sws.bom.gov.au/wdc/wdc_solradio/data/learmonth/

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import Final

import dask.array as da
import httpx
import pandas as pd
import xarray as xr

from radioskysims_sources.domain.localsources import SolarFlux
from radioskysims_core.logging_config import (
    get_logger,
    log_http_event,
    log_io_event,
)
from radioskysims_sources.infrastructure.ports.localsources import SolarFluxDownloaderPort

logger = logging.getLogger(__name__)
slogger = get_logger(__name__)

DEFAULT_SRS_FREQ_MHZ = [245, 410, 610, 1415, 2695, 4995, 8800, 15400]
SFU_TO_JY = 1.0e4


class SolarFluxDownloader(SolarFluxDownloaderPort):
    """Downloader de fluxo solar em formato SRD/SRS."""

    base_url: str = "https://downloads.sws.bom.gov.au/wdc/wdc_solradio/data/learmonth/SRD"

    def __init__(self, timeout_sec: int = 30) -> None:
        self.timeout_sec = timeout_sec
        self._day_cache: dict[str, pd.DataFrame | None] = {}

    def fetch(self, center_utc: datetime, window_hours: int = 2) -> SolarFlux:
        """Interface síncrona (compatível com o service atual)."""
        return self._run_async(self.fetch_async(center_utc=center_utc, window_hours=window_hours))

    async def fetch_async(self, center_utc: datetime, window_hours: int = 2) -> SolarFlux:
        """Baixa e consolida fluxo solar no intervalo centrado em ``center_utc``."""
        center = center_utc.astimezone(UTC).replace(tzinfo=None)
        delta = timedelta(hours=window_hours / 2)
        start_utc = center - delta
        end_utc = center + delta
        today_utc = datetime.now(UTC).date()

        days = {
            (start_utc - timedelta(days=1)).date(),
            start_utc.date(),
            center.date(),
            end_utc.date(),
            (end_utc + timedelta(days=1)).date(),
        }
        days = {day for day in days if day <= today_utc}

        records: list[pd.DataFrame] = []
        used_url: str | None = None
        missing_days: list = []

        for day in sorted(days):
            cache_key: Final[str] = day.isoformat()
            if cache_key in self._day_cache:
                df_day_cached = self._day_cache[cache_key]
                if df_day_cached is not None and not df_day_cached.empty:
                    records.append(df_day_cached)
                    if used_url is None:
                        used_url = self._build_url(day)
                continue

            missing_days.append(day)

        fetched_by_day = await self._download_days(missing_days)
        for day in missing_days:
            cache_key: Final[str] = day.isoformat()
            url = self._build_url(day)
            raw_text = fetched_by_day.get(day)
            if raw_text is None:
                self._day_cache[cache_key] = None
                logger.warning("Falha ao baixar SRD/SRS do dia %s (%s)", day.isoformat(), url)
                continue

            df_day = self._parse_srs(raw_text, day)
            self._day_cache[cache_key] = df_day
            if not df_day.empty:
                records.append(df_day)
                used_url = url

        if not records:
            msg = "Nenhum dado SRD/SRS disponível para a janela solicitada."
            raise RuntimeError(msg)

        full = pd.concat(records, ignore_index=True)
        full = full[(full["time"] >= start_utc) & (full["time"] <= end_utc)]
        if full.empty:
            msg = "SRD/SRS baixado, mas sem amostras no intervalo solicitado."
            raise RuntimeError(msg)

        full = full.sort_values(["time", "frequency_mhz"])
        flux_table = full.pivot_table(
            index="time",
            columns="frequency_mhz",
            values="flux_jy",
            aggfunc="mean",
        ).sort_index()

        times = flux_table.index.to_numpy(dtype="datetime64[ns]")
        freqs = flux_table.columns.to_numpy(dtype=float)
        values = flux_table.to_numpy(dtype=float)

        values_da = da.from_array(values, chunks="auto")
        dataset = xr.Dataset(
            data_vars={"flux_jy": (("time", "frequency_mhz"), values_da)},
            coords={"time": times, "frequency_mhz": freqs},
            attrs={"units": "Jy", "source": "SRD"},
        )

        return SolarFlux(
            start_utc=start_utc,
            end_utc=end_utc,
            dataset=dataset,
            source_url=used_url,
        )

    async def _download_days(self, days: list) -> dict:
        if not days:
            return {}

        timeout = httpx.Timeout(self.timeout_sec)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [self._download_text_async(client, day) for day in days]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        out: dict = {}
        for day, result in zip(days, results, strict=False):
            if isinstance(result, Exception):
                out[day] = None
            else:
                out[day] = result
        return out

    async def _download_text_async(self, client: httpx.AsyncClient, day_date) -> str:
        url = self._build_url(day_date)
        start = datetime.now(UTC)
        response = await client.get(url)
        response.raise_for_status()
        duration_ms = (datetime.now(UTC) - start).total_seconds() * 1000
        log_http_event(
            slogger,
            "solar_flux_download",
            method="GET",
            url=url,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response.text

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: dict[str, object] = {}

        def _worker() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - caminho excepcional
                result["error"] = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()

        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        return result["value"]

    def _build_url(self, day_date) -> str:
        year = day_date.year
        yy = str(year)[-2:]
        mm = f"{day_date.month:02d}"
        dd = f"{day_date.day:02d}"
        return f"{self.base_url}/{year}/L{yy}{mm}{dd}.SRD"

    def _decode_flux_token_to_jy(self, token: str) -> float | None:
        if "/" in token:
            return None

        if token.isdigit() and len(token) == 4:
            # Formato SRD: abcp => a.bc * 10^p (em SFU)
            a = int(token[0])
            b = int(token[1])
            c = int(token[2])
            p = int(token[3])
            value_sfu = (a + b / 10.0 + c / 100.0) * (10.0**p)
            return value_sfu * SFU_TO_JY

        try:
            value = float(token)
        except ValueError:
            return None
        if value < 0:
            return None

        # Fallback para entradas já em SFU (legado)
        return value * SFU_TO_JY

    def _parse_srs(self, raw_text: str, day_date) -> pd.DataFrame:
        log_io_event(
            slogger,
            "solar_flux_parse_start",
            operation="parse_srd_srs",
            path=day_date.isoformat(),
            bytes_count=len(raw_text.encode("utf-8")),
        )
        in_data = False
        rows: list[dict[str, float | datetime]] = []

        for line in io.StringIO(raw_text):
            s = line.strip()
            if not s:
                continue
            if s.startswith("::"):
                in_data = True
                continue
            if s.startswith(":") and not in_data:
                continue

            # Arquivos SRD podem vir sem cabeçalho/sentinela. Nesse caso,
            # tentamos parsear linhas de dados diretamente.

            tokens = s.split()
            if len(tokens) < 2:
                continue

            # Formato novo (SRD): HHMMSS F245 F410 ... F15400
            # Formato antigo (SRS): HHMM SS F245 F410 ... F15400
            hhmmss_token = tokens[0]
            sec = 0
            values_tokens: list[str]

            if len(hhmmss_token) == 6 and hhmmss_token.isdigit():
                hour = int(hhmmss_token[:2])
                minute = int(hhmmss_token[2:4])
                sec = int(hhmmss_token[4:6])
                values_tokens = tokens[1:]
            elif (
                len(hhmmss_token) == 4
                and hhmmss_token.isdigit()
                and len(tokens) >= 3
                and tokens[1].lstrip("+-").isdigit()
            ):
                hour = int(hhmmss_token[:2])
                minute = int(hhmmss_token[2:4])
                sec = int(tokens[1])
                values_tokens = tokens[2:]
            else:
                continue

            ts = datetime(
                year=day_date.year,
                month=day_date.month,
                day=day_date.day,
                hour=hour,
                minute=minute,
                second=sec,
            )

            for freq_mhz, token in zip(DEFAULT_SRS_FREQ_MHZ, values_tokens, strict=False):
                value_jy = self._decode_flux_token_to_jy(token)
                if value_jy is None:
                    continue
                rows.append(
                    {
                        "time": ts,
                        "frequency_mhz": float(freq_mhz),
                        "flux_jy": value_jy,
                    }
                )

        parsed = pd.DataFrame(rows, columns=["time", "frequency_mhz", "flux_jy"])
        log_io_event(
            slogger,
            "solar_flux_parse_done",
            operation="parse_srd_srs",
            path=day_date.isoformat(),
            bytes_count=int(parsed.memory_usage(index=True, deep=True).sum()),
            rows=len(parsed),
        )
        return parsed
