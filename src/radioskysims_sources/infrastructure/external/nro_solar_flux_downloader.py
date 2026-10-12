"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/infrastructure/external/nro_solar_flux_downloader.py

Descrição:
    Adapter externo para o fluxo solar calibrado (absolute measurements) do
    **Nobeyama Radio Polarimeters (NoRP)** do NAOJ/NRO, em formato XDR
    (``IDL SAVE``). Fonte:
    ``https://solar.nro.nao.ac.jp/norp/xdr/yyyy/mm/norpyymmdd.xdr``.

    O arquivo é um IDL SAVE (XDR) lido com ``scipy.io.readsav`` e contém:

    - ``day``   : data (str, "YYYY-MM-DD");
    - ``tim``   : recarray ``{TIME: i4 (ms do dia, UT), DAY: i2}``, onde
                  ``DAY`` é o número de dias desde 1979-01-01 (UT);
    - ``fi``    : ``(N, nchan)`` fluxo total calibrado (Stokes I) em SFU
                  — as *absolute measurements*;
    - ``fv``    : ``(N, nchan)`` polarização circular (Stokes V) em SFU;
    - ``freq``  : ``(nchan,)`` frequências em GHz (1, 2, 3.75, 9.4, 17, 35);
    - ``fiavg``/``fvavg`` : médias diárias por canal (SFU).

    1 SFU = 1e4 Jy. Preferimos ``fi`` (série temporal) para que a redução
    diária (NaN-aware) seja feita a jusante por ``SolarFlux.daily_mean_grid``.

Copyright BINGO Collaboration
Last modified: 2026-10-12
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
import numpy as np
import pandas as pd
import xarray as xr
from radioskysims_core.logging_config import (
    get_logger,
    log_http_event,
)

from radioskysims_sources.domain.localsources import SolarFlux
from radioskysims_sources.infrastructure.ports.localsources import SolarFluxDownloaderPort

logger = logging.getLogger(__name__)
slogger = get_logger(__name__)

# SFU → Jy (1 SFU = 1e4 Jy)
SFU_TO_JY: float = 1.0e4

# Época do campo ``tim.DAY``: dias desde 1979-01-01 (UT). Verificado contra o
# campo ``day`` dos arquivos (ex.: DAY=17379 ↔ 2026-08-01).
NRO_EPOCH_ISO: Final[str] = "1979-01-01T00:00:00"


class NroSolarFluxDownloader(SolarFluxDownloaderPort):
    """Downloader de fluxo solar NoRP (NRO, formato XDR/IDL SAVE).

    Extrai as *absolute measurements* (``fi``, Stokes I calibrado, em SFU) por
    frequência e devolve um ``SolarFlux`` (grade tempo × frequência em Jy).
    """

    base_url: str = "https://solar.nro.nao.ac.jp/norp/xdr"

    def __init__(
        self,
        timeout_sec: int = 60,
        quantity: str = "I",
        drop_dead_channels: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        timeout_sec : int
            Timeout HTTP (s).
        quantity : str
            ``"I"`` (fluxo total/absolute, ``fi``) ou ``"V"`` (polarização,
            ``fv``).
        drop_dead_channels : bool
            Se ``True``, remove canais sem nenhuma amostra finita no dia
            (ex.: 35 GHz desativado → ``fiavg == 0``).
        """
        self.timeout_sec = timeout_sec
        q = quantity.strip().upper()
        if q not in {"I", "V"}:
            msg = f"quantity deve ser 'I' ou 'V' (recebido {quantity!r})"
            raise ValueError(msg)
        self.quantity = q
        self.drop_dead_channels = drop_dead_channels
        self._day_cache: dict[str, pd.DataFrame | None] = {}

    # ------------------------------------------------------------------
    # Porta pública
    # ------------------------------------------------------------------
    def fetch(self, center_utc: datetime, window_hours: int = 2) -> SolarFlux:
        """Interface síncrona (compatível com o service)."""
        return self._run_async(self.fetch_async(center_utc=center_utc, window_hours=window_hours))

    async def fetch_async(self, center_utc: datetime, window_hours: int = 2) -> SolarFlux:
        """Baixa e consolida o fluxo NoRP na janela centrada em ``center_utc``."""
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
            cached = self._day_cache.get(cache_key, "")
            if cache_key in self._day_cache:
                if cached is not None and not cached.empty:
                    records.append(cached)
                    if used_url is None:
                        used_url = self._build_url(day)
                continue
            missing_days.append(day)

        fetched = await self._download_days(missing_days)
        for day in missing_days:
            cache_key: Final[str] = day.isoformat()
            url = self._build_url(day)
            raw_bytes = fetched.get(day)
            if raw_bytes is None:
                self._day_cache[cache_key] = None
                logger.warning("Falha ao baixar NoRP/XDR do dia %s (%s)", cache_key, url)
                continue

            df_day = self.parse_save(raw_bytes, quantity=self.quantity)
            self._day_cache[cache_key] = df_day
            if not df_day.empty:
                records.append(df_day)
                used_url = url

        if not records:
            msg = "Nenhum dado NoRP/XDR disponível para a janela solicitada."
            raise RuntimeError(msg)

        full = pd.concat(records, ignore_index=True)
        full = full[(full["time"] >= start_utc) & (full["time"] <= end_utc)]
        if full.empty:
            msg = "NoRP/XDR baixado, mas sem amostras no intervalo solicitado."
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
            attrs={
                "units": "Jy",
                "source": "NoRP",
                "quantity": self.quantity,
                "calibration": "absolute",
            },
        )

        return SolarFlux(
            start_utc=start_utc,
            end_utc=end_utc,
            dataset=dataset,
            source_url=used_url,
        )

    # ------------------------------------------------------------------
    # Parsing do IDL SAVE (XDR)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_save(
        source: str | bytes | io.BufferedIOBase,
        *,
        quantity: str = "I",
        drop_dead_channels: bool = True,
    ) -> pd.DataFrame:
        """Lê um IDL SAVE NoRP e devolve DataFrame longo (time/freq/flux).

        Parameters
        ----------
        source : str | bytes | file-like
            Caminho, bytes ou arquivo do ``.xdr``.
        quantity : str
            ``"I"`` → ``fi`` (fluxo total/absolute); ``"V"`` → ``fv``.
        drop_dead_channels : bool
            Remove canais sem amostras finitas.

        Returns
        -------
        pandas.DataFrame
            Colunas ``time`` (datetime64[ns]), ``frequency_mhz``, ``flux_jy``.
        """
        from scipy.io import readsav

        if isinstance(source, str):
            save = readsav(source, verbose=False)
        elif isinstance(source, bytes):
            save = readsav(io.BytesIO(source), verbose=False)
        else:
            save = readsav(source, verbose=False)

        key = "fi" if quantity.strip().upper() == "I" else "fv"
        flux = np.asarray(save[key], dtype=float)
        freq_ghz = np.asarray(save["freq"], dtype=float)
        tim = save["tim"]
        day = np.asarray(tim["DAY"], dtype=np.int64)
        msec = np.asarray(tim["TIME"], dtype=np.int64)

        base = np.datetime64(NRO_EPOCH_ISO)
        times = (base + day.astype("timedelta64[D]") + msec.astype("timedelta64[ms]")).astype(
            "datetime64[ns]"
        )

        freq_mhz = freq_ghz * 1000.0  # GHz → MHz
        n_time, n_chan = flux.shape

        frames: list[pd.DataFrame] = []
        for c in range(n_chan):
            col = flux[:, c]
            if drop_dead_channels and not np.any(np.isfinite(col)):
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "time": times,
                        "frequency_mhz": np.full(n_time, freq_mhz[c], dtype=float),
                        "flux_jy": col * SFU_TO_JY,
                    }
                )
            )

        if not frames:
            return pd.DataFrame(columns=["time", "frequency_mhz", "flux_jy"])
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    async def _download_days(self, days: list) -> dict:
        if not days:
            return {}
        timeout = httpx.Timeout(self.timeout_sec)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [self._download_bytes_async(client, day) for day in days]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        out: dict = {}
        for day, result in zip(days, results, strict=False):
            out[day] = None if isinstance(result, Exception) else result
        return out

    async def _download_bytes_async(self, client: httpx.AsyncClient, day_date) -> bytes:
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
        return response.content

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
        return f"{self.base_url}/{year}/{mm}/norp{yy}{mm}{dd}.xdr"


__all__ = ["NroSolarFluxDownloader", "SFU_TO_JY", "NRO_EPOCH_ISO"]
