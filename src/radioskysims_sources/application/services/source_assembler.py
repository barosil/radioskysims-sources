"""
Módulo: src/radioskysims/application/services/source_assembler.py

Serviço de aplicação responsável por montar o array de SignalSource
para uma simulação, reunindo satélites, fontes locais e fontes de
rádio com seus respectivos tracks e sinais.

Copyright BINGO Collaboration
Last modified: 2026-07-27
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from radioskysims_core.domain.common import Observer, TimeRange

from radioskysims_sources.domain.localsources import SolarFlux
from radioskysims_sources.domain.signal_source import (
    LocalSourceSignalSource,
    RadioSourceSignalSource,
    SatelliteSignalSource,
    SignalSource,
)

log = logging.getLogger(__name__)

# Fluxo solar padrão (SFU @ 2800 MHz, F10.7) usado quando não há dados SRD/SRS
# para a janela solicitada (ex.: planejamento de observações FUTURAS, em que o
# downloader só aceita dias <= hoje). Valor de referência de Sol calmo.
# 1 SFU = 1e4 Jy.
DEFAULT_SOLAR_FLUX_SFU: float = 150.0


def _default_solar_flux(
    time_grid: np.ndarray | None,
    freq_mhz: np.ndarray | None,
) -> SolarFlux | None:
    """SolarFlux constante (Sol calmo) cobrindo o grid completo da simulação.

    O fluxo é constante em tempo e frequência: ``DEFAULT_SOLAR_FLUX_SFU``
    SFU convertido para Jy (1 SFU = 1e4 Jy). Cobre todos os instantes do
    ``time_grid`` — diferente do caminho SRD/SRS, que só cobre ±12 h em
    torno do centro (inadequado para janelas multi-dia).
    """
    if time_grid is None or len(time_grid) == 0 or freq_mhz is None or len(freq_mhz) == 0:
        return None
    times = np.asarray(time_grid).astype("datetime64[ns]")
    freqs = np.asarray(freq_mhz, dtype=float)
    # Eixo de frequência com >= 2 pontos: o interp do xarray degenera (divide
    # por zero → NaN) quando a dimensão interpolada tem um único ponto.
    if freqs.size == 1:
        f0 = freqs[0]
        freqs_axis = np.array([f0 - 1.0, f0, f0 + 1.0])
    else:
        freqs_axis = freqs
    flux_jy = float(DEFAULT_SOLAR_FLUX_SFU) * 1.0e4
    ds = xr.Dataset(
        {
            "flux_jy": (
                ("time", "frequency_mhz"),
                np.full((times.size, freqs_axis.size), flux_jy),
            )
        },
        coords={"time": times, "frequency_mhz": freqs_axis},
    )
    return SolarFlux(
        start_utc=pd.Timestamp(times[0]).to_pydatetime(),
        end_utc=pd.Timestamp(times[-1]).to_pydatetime(),
        dataset=ds,
        source_url="default:quiet-sun-fallback",
    )


class SourceAssembler:
    """Monta o array de SignalSource para uma simulação.

    Orquestra a obtenção de tracks e sinais para cada tipo de fonte:
    - Satélites: usa o fluxo de SatelliteService + SimulationService
    - Fontes locais: usa LocalSourcesService para tracks e SolarFlux
    - Fontes de rádio: usa RadioSourceCatalogService para tracks e fluxo
    """

    def __init__(
        self,
        satellite_assembler: Any | None = None,
        local_sources_service: Any | None = None,
        radio_source_catalog_service: Any | None = None,
        frequency_hz: np.ndarray | None = None,
        logger: Any | None = None,
    ) -> None:
        self._satellite_assembler = satellite_assembler
        self._local_sources_service = local_sources_service
        self._radio_catalog = radio_source_catalog_service
        self._frequency_hz = frequency_hz
        self._logger = logger

    # ──────────────────────────────────────────────
    # Satélites
    # ──────────────────────────────────────────────

    def assemble_satellites(
        self,
        assembled_satellites: list[object],
    ) -> list[SignalSource]:
        """Envolve satélites já montados (SatelliteAssembled) em SignalSource."""
        return [SatelliteSignalSource(sat) for sat in assembled_satellites]

    # ──────────────────────────────────────────────
    # Fontes locais
    # ──────────────────────────────────────────────

    def assemble_local_sources(
        self,
        observer: Observer,
        time_range: TimeRange,
        time_grid: np.ndarray,
        freq_mhz: np.ndarray,
        *,
        sun_flux_mean_window_s: float | None = None,
        beam_fwhm_deg: float | None = None,
    ) -> list[SignalSource]:
        """Monta SignalSource para fontes locais (ex.: Sun).

        Para cada fonte local:
        1. Obtém track (alt-az) via LocalSourceTrackProvider
        2. Obtém fluxo solar via SolarFlux (SRD/SRS)
        3. Cria LocalSourceSignalSource

        Args:
            observer: Observador terrestre.
            time_range: Intervalo temporal.
            time_grid: Grade temporal (datetime64[ns]) da simulação.
            freq_mhz: Grade de frequências em MHz.
            sun_flux_mean_window_s: Largura da janela temporal (s) para a
                média do fluxo solar. Se ``None``, usa o tempo aproximado de
                um trânsito do Sol pelo feixe (``beam_fwhm_deg``).
            beam_fwhm_deg: Largura à meia-potência do feixe (graus), usada
                apenas para o valor padrão da janela quando ela é omitida.

        Returns:
            Lista de SignalSource para fontes locais.
        """
        if self._local_sources_service is None:
            return []

        local_sources = self._local_sources_service.build_local_sources()
        track_sources = [s.name for s in local_sources.sources if s.has_track]
        if not track_sources:
            return []

        # Obtém fluxo solar para o período. A janela cobre TODOS os dias da
        # grade de simulação (com margem de 1 dia) para que a média diária
        # tenha dados reais em cada dia — evita cair no fallback de Sol calmo.
        center_utc = _grid_center_datetime(time_grid)
        window_hours = _solar_window_hours(time_grid)
        solar_flux = None
        try:
            solar_flux = self._local_sources_service.get_solar_flux(
                center_utc=center_utc, window_hours=window_hours
            )
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "solar_flux_fetch_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                log.warning("solar_flux_fetch_failed: %s", exc)
            # Janela futura (ex.: planejamento) sem dados SRD/SRS → usa fluxo
            # padrão de Sol calmo cobrindo TODO o grid da simulação.
            solar_flux = _default_solar_flux(time_grid, freq_mhz)
            if solar_flux is not None:
                if self._logger is not None:
                    self._logger.warning(
                        "solar_flux_default_fallback",
                        solar_flux_sfu=DEFAULT_SOLAR_FLUX_SFU,
                    )
                else:
                    log.warning(
                        "solar_flux_default_fallback: Sol calmo (%s SFU) em todo o grid",
                        DEFAULT_SOLAR_FLUX_SFU,
                    )

        # Avalia grid de fluxo solar (Jy).
        # Média DIÁRIA (NaN-aware) em vez de interpolação hora a hora: o SRD/SRS
        # é medido em instantes UTC que não coincidem com a hora local do
        # observatório — interpolar no tempo compararia instantes de medição do
        # Sol com a grade de observação (extrapolação/NaN nas bordas).
        flux_grid = None
        if solar_flux is not None and len(freq_mhz) > 0:
            try:
                flux_grid = solar_flux.daily_mean_grid(
                    times=time_grid,
                    freqs_mhz=freq_mhz.astype(float),
                )
            except Exception as exc:
                if self._logger is not None:
                    self._logger.warning(
                        "solar_flux_interpolation_failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                else:
                    log.warning("solar_flux_interpolation_failed: %s", exc)

        frequency_hz = freq_mhz * 1.0e6 if freq_mhz is not None else None

        # Segundos desde o início (para localizar a janela temporal no grid)
        time_seconds = None
        if time_grid is not None and len(time_grid) > 0:
            sec = np.asarray(time_grid).astype("datetime64[s]").astype(float)
            time_seconds = sec - sec[0]

        sources: list[SignalSource] = []
        for src_name in track_sources:
            try:
                track = self._local_sources_service.get_local_source_track(
                    source_name=src_name,
                    observer=observer,
                    time_range=time_range,
                )
                # Interpola track para o grid temporal da simulação
                track_ds = track.dataset.assign_coords(
                    time=track.dataset["time"].astype("datetime64[ns]")
                )
                interp_track_ds = track_ds.interp(
                    time=time_grid,
                    kwargs={"fill_value": "extrapolate"},
                )

                sources.append(
                    LocalSourceSignalSource(
                        name=src_name,
                        track=_TrackWrapper(interp_track_ds),
                        flux_jy_grid=flux_grid,
                        frequency_hz=frequency_hz,
                        time_seconds=time_seconds,
                        mean_window_s=sun_flux_mean_window_s,
                        beam_fwhm_deg=beam_fwhm_deg,
                    )
                )
            except Exception as exc:
                if self._logger is not None:
                    self._logger.warning(
                        "local_source_track_failed",
                        source_name=src_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                else:
                    log.warning(
                        "local_source_track_failed for '%s': %s",
                        src_name,
                        exc,
                    )

        return sources

    # ──────────────────────────────────────────────
    # Fontes de rádio
    # ──────────────────────────────────────────────

    def assemble_radio_sources(
        self,
        observer: Observer,
        time_range: TimeRange,
        time_grid: np.ndarray,
        freq_mhz: np.ndarray,
        *,
        ra_center_deg: float = 190.0,
        dec_center_deg: float = 0.0,
        search_radius_deg: float = 20.0,
        min_flux_jy: float = 0.1,
        pointing_track: xr.Dataset | None = None,
        pointing_ra_deg: float | None = None,
        pointing_dec_deg: float | None = None,
        beam_fwhm_deg: float | None = None,
        fwhm_factor: float = 3.0,
    ) -> list[SignalSource]:
        """Monta SignalSource para fontes de rádio (NVSS).

        Para cada fonte de rádio:
        1. Filtra por pointing (RA/Dec) — apenas fontes dentro do feixe
           (N×FWHM) geram tracks/sinais;
        2. Calcula fluxo (Jy) via modelo espectral;
        3. Calcula track (alt-az) via SkyfieldRadioSourceTrackProvider;
        4. Cria RadioSourceSignalSource.

        Quando ``pointing_track`` ou ``pointing_ra_deg/dec_deg`` são
        fornecidos, a região de consulta do catálogo é centralizada no
        pointing e as fontes são pré-filtradas pela menor separação angular
        ao caminho apontado (limiar = ``fwhm_deg * fwhm_factor``). Isso evita
        construir tracks e sinais para fontes irrelevantes à simulação.

        Args:
            observer: Observador terrestre.
            time_range: Intervalo temporal.
            time_grid: Grade temporal (datetime64[ns]) da simulação.
            freq_mhz: Grade de frequências em MHz.
            ra_center_deg: RA central da região (graus, fallback sem pointing).
            dec_center_deg: Dec central da região (graus, fallback).
            search_radius_deg: Raio de busca (graus).
            min_flux_jy: Fluxo mínimo (Jy).
            pointing_track: Varredura de pointing (ra/dec em radianos).
            pointing_ra_deg: RA fixa do pointing (graus).
            pointing_dec_deg: Dec fixa do pointing (graus).
            beam_fwhm_deg: FWHM do feixe (graus) para o limiar N×FWHM.
            fwhm_factor: Multiplicador do FWHM (padrão 3×).

        Returns:
            Lista de SignalSource para fontes de rádio.
        """
        if self._radio_catalog is None:
            return []

        from radioskysims_sources.domain.radiosources.selection import (
            pointing_region,
            select_sources_in_beam,
        )

        # ── Região de consulta centralizada no pointing ──
        _region = pointing_region(
            pointing_track=pointing_track,
            pointing_ra_deg=pointing_ra_deg,
            pointing_dec_deg=pointing_dec_deg,
        )
        _beam_radius = (
            beam_fwhm_deg * fwhm_factor if beam_fwhm_deg is not None and beam_fwhm_deg > 0 else None
        )
        _query_ra = ra_center_deg
        _query_dec = dec_center_deg
        _query_radius = search_radius_deg
        if _region is not None:
            _query_ra, _query_dec, _coverage = _region
            _query_radius = max(search_radius_deg, _coverage + (_beam_radius or 0.0))

        # Lista fontes na região
        sources = self._radio_catalog.list_sources_in_region(
            ra_deg=_query_ra,
            dec_deg=_query_dec,
            radius_deg=_query_radius,
            min_flux_jy=min_flux_jy,
        )
        if not sources:
            if self._logger is not None:
                self._logger.info(
                    "radio_sources_not_found_in_region",
                    ra_deg=_query_ra,
                    dec_deg=_query_dec,
                    radius_deg=_query_radius,
                    min_flux_jy=min_flux_jy,
                )
            else:
                log.info(
                    "Nenhuma fonte de rádio na região RA=%.1f Dec=%.1f r=%.1f° S>%.2f Jy",
                    _query_ra,
                    _query_dec,
                    _query_radius,
                    min_flux_jy,
                )

            return []

        # ── Pré-filtra por pointing (RA/Dec) antes de gerar tracks/sinais ──
        n_before = len(sources)
        sources, _separations = select_sources_in_beam(
            sources,
            pointing_track=pointing_track,
            pointing_ra_deg=pointing_ra_deg,
            pointing_dec_deg=pointing_dec_deg,
            fwhm_deg=beam_fwhm_deg,
            factor=fwhm_factor,
        )
        if not sources:
            beam_radius_deg = _beam_radius if _beam_radius is not None else _query_radius

            if self._logger is not None:
                self._logger.info(
                    "radio_sources_not_found_in_beam",
                    ra_deg=_query_ra,
                    dec_deg=_query_dec,
                    beam_radius_deg=beam_radius_deg,
                )
            else:
                log.info(
                    "Nenhuma fonte de rádio dentro do feixe (pointing) em "
                    "RA=%.1f Dec=%.1f (limiar=%.1f°)",
                    _query_ra,
                    _query_dec,
                    beam_radius_deg,
                )

            return []
        if len(sources) < n_before:
            n_after = len(sources)
            n_discarded = n_before - n_after

            if self._logger is not None:
                self._logger.info(
                    "radio_sources_beam_filter",
                    sources_before=n_before,
                    sources_after=n_after,
                    sources_discarded=n_discarded,
                )
            else:
                log.info(
                    "radio_sources_beam_filter: %d -> %d (descartadas %d fora do feixe)",
                    n_before,
                    n_after,
                    n_discarded,
                )

        if self._logger is not None:
            self._logger.info(
                "radio_sources_assembly_started",
                source_count=len(sources),
            )
        else:
            log.info(
                "Montando SignalSource para %d fontes de rádio",
                len(sources),
            )

        # Calcula fluxo na grade de frequências
        flux_map = self._radio_catalog.evaluate_flux(sources, freq_mhz)
        frequency_hz = freq_mhz * 1.0e6 if freq_mhz is not None else None

        # Calcula tracks para todas as fontes
        try:
            radio_track = self._radio_catalog.compute_tracks(
                sources=sources, observer=observer, time_range=time_range
            )
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "radio_source_track_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                log.warning("radio_source_track_failed: %s", exc)

            return []

        from radioskysims_sources.domain.signal_source_group import RadioSourceGroupSignalSource

        # Converte para SignalSource
        signal_sources: list[SignalSource] = []
        for src in sources:
            flux_jy = flux_map.get(src.id)
            if flux_jy is None:
                continue

            source_track = _extract_single_source_track(radio_track, src.id, time_grid)
            if source_track is None:
                if self._logger is not None:
                    self._logger.debug(
                        "radio_source_track_missing",
                        source_id=src.id,
                    )
                else:
                    log.debug(
                        "Pulando fonte %s — sem dados de track no grid",
                        src.id,
                    )

                continue

            signal_sources.append(
                RadioSourceSignalSource(
                    source_id=src.id,
                    track=source_track,
                    flux_jy_grid=np.asarray(flux_jy, dtype=float),
                    frequency_hz=frequency_hz,
                )
            )

        # Agrupa fontes co-localizadas (mesmo RA/Dec) para colapsar
        # o grafo dask: N fontes na mesma posição viram 1 grupo.
        n_before = len(signal_sources)
        signal_sources = RadioSourceGroupSignalSource.group_by_position(signal_sources)
        n_after = len(signal_sources)
        if n_after < n_before:
            reduction_percent = (1 - n_after / max(n_before, 1)) * 100

            if self._logger is not None:
                self._logger.info(
                    "radio_sources_grouped",
                    sources_before=n_before,
                    sources_after=n_after,
                    reduction_percent=reduction_percent,
                )
            else:
                log.info(
                    "radio_sources_grouped: %d -> %d (%.1f%% redução)",
                    n_before,
                    n_after,
                    reduction_percent,
                )

        return signal_sources


# ═══════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════


def _grid_center_datetime(time_grid: np.ndarray):
    """Retorna datetime UTC no centro do grid temporal."""
    from datetime import UTC, datetime

    if len(time_grid) == 0:
        return datetime.now(UTC)
    mid = len(time_grid) // 2
    dt = time_grid[mid]
    if isinstance(dt, np.datetime64):
        return dt.astype("datetime64[us]").tolist().replace(tzinfo=UTC)
    return datetime.now(UTC)


def _solar_window_hours(time_grid: np.ndarray) -> int:
    """Janela (h) do download de fluxo solar que cobre todo o grid temporal.

    Como o fluxo passa a ser usado por **média diária**, a janela precisa
    conter todos os dias UTC presentes na grade (com margem de 1 dia antes e
    depois). Mínimo de 24 h.
    """
    if time_grid is None or len(time_grid) == 0:
        return 24
    grid = np.asarray(time_grid).astype("datetime64[ns]")
    span_h = float((grid.max() - grid.min()) / np.timedelta64(1, "h"))
    if not np.isfinite(span_h) or span_h < 0:
        return 24
    # +2 dias de margem (um de cada lado), arredondado para cima.
    return int(np.ceil(span_h)) + 48


def _extract_single_source_track(
    radio_track: Any,
    source_id: str,
    time_grid: np.ndarray,
) -> object | None:
    """Extrai track de uma única fonte do RadioSourceTrack multicanal."""
    ds = getattr(radio_track, "dataset", None)
    if ds is None:
        return None
    if "source" not in ds.dims:
        return None
    try:
        subset = ds.sel(source=source_id)
    except (KeyError, ValueError):
        return None

    # Interpola para o grid temporal
    subset = subset.assign_coords(time=subset["time"].astype("datetime64[ns]"))
    interp = subset.interp(
        time=time_grid,
        kwargs={"fill_value": "extrapolate"},
    )
    return _TrackWrapper(interp)


class _TrackWrapper:
    """Wrapper simples que expõe .dataset para adapters SignalSource."""

    def __init__(self, dataset: xr.Dataset) -> None:
        self.dataset = dataset
