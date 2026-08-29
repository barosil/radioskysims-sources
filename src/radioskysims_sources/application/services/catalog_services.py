"""
Módulo: src/radioskysims/application/services/catalog_services.py

Serviço de aplicação para o contexto de catálogos de fontes de rádio.

Orquestra:
    - Ingestão de catálogos via TAP/ADQL (NVSS + SPECFIND via VizieR).
    - Persistência do catálogo em banco PostgreSQL.
    - Cálculo de tracks para fontes fixas.
    - Avaliação de sinal (fluxo) para fontes em frequências arbitrárias.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

import logging

import numpy as np

from radioskysims_sources.application.ports.radiosources import (
    CatalogIngestPort,
    RadioSourcesRepositoryPort,
)
from radioskysims_sources.application.radiosources_use_cases import (
    BuildNvssSpecfindCatalogInput,
    BuildNvssSpecfindCatalogUseCase,
)
from radioskysims_sources.domain.catalog.models import CatalogIngestion, CatalogQuery
from radioskysims_core.domain.common import Observer, TimeRange
from radioskysims_sources.domain.radiosources.models import RadioSource
from radioskysims_sources.domain.radiosources.signals import RadioSourceSignalEvaluator
from radioskysims_sources.domain.radiosources.tracks import RadioSourceTrack
from radioskysims_sources.infrastructure.external.radiosource_track_provider import (
    SkyfieldRadioSourceTrackProvider,
)

log = logging.getLogger(__name__)


class RadioSourceCatalogService:
    """Serviço de aplicação para catálogos de fontes de rádio.

    Encapsula os fluxos de ingestão NVSS+SPECFIND, persistência,
    cálculo de tracks e avaliação de sinal.
    """

    def __init__(
        self,
        ingest_port: CatalogIngestPort,
        repository: RadioSourcesRepositoryPort,
        track_provider: SkyfieldRadioSourceTrackProvider | None = None,
    ) -> None:
        self._ingest_port = ingest_port
        self._repository = repository
        self._track_provider = track_provider or SkyfieldRadioSourceTrackProvider()
        self._use_case = BuildNvssSpecfindCatalogUseCase(
            ingest_port=ingest_port,
            repository=repository,
        )

    # ──────────────────────────────────────────────
    # Ingestão de catálogo
    # ──────────────────────────────────────────────

    def ingest_nvss_specfind(
        self,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        smin_jy: float = 0.1,
    ) -> CatalogIngestion:
        """Ingere catálogo NVSS + SPECFIND para uma região do céu.

        Args:
            ra_min_deg: RA mínima (graus).
            ra_max_deg: RA máxima (graus).
            dec_min_deg: Dec mínima (graus).
            dec_max_deg: Dec máxima (graus).
            smin_jy: Fluxo mínimo em 1.4 GHz (Jy).

        Returns:
            CatalogIngestion com resultado da operação.
        """
        params = BuildNvssSpecfindCatalogInput(
            ra_min_deg=ra_min_deg,
            ra_max_deg=ra_max_deg,
            dec_min_deg=dec_min_deg,
            dec_max_deg=dec_max_deg,
            smin_jy=smin_jy,
        )

        query = CatalogQuery(
            survey="NVSS+SPECFIND",
            query=(
                f"NVSS ({ra_min_deg}≤RA≤{ra_max_deg}, "
                f"{dec_min_deg}≤Dec≤{dec_max_deg}, S1.4>{smin_jy}Jy)"
            ),
            params={
                "ra_min_deg": ra_min_deg,
                "ra_max_deg": ra_max_deg,
                "dec_min_deg": dec_min_deg,
                "dec_max_deg": dec_max_deg,
                "smin_jy": smin_jy,
            },
        )

        try:
            sources = self._use_case.execute(params)
            num_sources = len(sources)
            status = "success" if num_sources > 0 else "partial"
            log.info(
                "NVSS+SPECFIND ingest concluída: %d fontes para região "
                "RA=[%.1f, %.1f], Dec=[%.1f, %.1f], S1.4>%.3f Jy",
                num_sources,
                ra_min_deg,
                ra_max_deg,
                dec_min_deg,
                dec_max_deg,
                smin_jy,
            )
        except Exception as exc:
            log.exception("Falha na ingestão NVSS+SPECFIND: %s", exc)
            return CatalogIngestion(
                catalog_name="nvss_specfind",
                num_sources=0,
                query=query,
                status="failed",
                error=str(exc),
            )

        return CatalogIngestion(
            catalog_name="nvss_specfind",
            num_sources=num_sources,
            query=query,
            status=status,
        )

    # ──────────────────────────────────────────────
    # Consulta
    # ──────────────────────────────────────────────

    def list_sources_in_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float = 5.0,
        min_flux_jy: float | None = None,
    ) -> list[RadioSource]:
        """Retorna fontes persistidas em uma região circular.

        Args:
            ra_deg: RA central (graus).
            dec_deg: Dec central (graus).
            radius_deg: Raio de busca (graus).
            min_flux_jy: Corte opcional de fluxo mínimo (Jy).

        Returns:
            Lista de RadioSource na região.
        """
        return self._repository.list_sources_in_region(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_deg=radius_deg,
            min_flux_jy=min_flux_jy,
        )

    # ──────────────────────────────────────────────
    # Tracks
    # ──────────────────────────────────────────────

    def compute_tracks(
        self,
        sources: list[RadioSource],
        observer: Observer,
        time_range: TimeRange,
    ) -> RadioSourceTrack:
        """Calcula tracks de elevação/azimute para fontes fixas.

        Args:
            sources: Fontes de rádio com coordenadas celestes.
            observer: Observador terrestre.
            time_range: Intervalo temporal.

        Returns:
            RadioSourceTrack com posições ao longo do tempo.
        """
        return self._track_provider.compute_track(
            sources=sources,
            observer=observer,
            time_range=time_range,
        )

    # ──────────────────────────────────────────────
    # Sinais / Fluxo
    # ──────────────────────────────────────────────

    def evaluate_flux(
        self,
        sources: list[RadioSource],
        freqs_mhz: np.ndarray | list[float],
    ) -> dict[str, np.ndarray]:
        """Avalia fluxo (Jy) para cada fonte na grade de frequências.

        Args:
            sources: Lista de fontes.
            freqs_mhz: Array de frequências em MHz.

        Returns:
            Dict {source_id: np.ndarray} com fluxo (Jy).
        """
        return RadioSourceSignalEvaluator.evaluate_flux_grid(sources, freqs_mhz)

    def get_brightest_sources(
        self,
        sources: list[RadioSource],
        freqs_mhz: np.ndarray | list[float],
        top_n: int = 10,
    ) -> list[RadioSource]:
        """Retorna as N fontes mais brilhantes na banda de frequências.

        Args:
            sources: Lista de fontes candidatas.
            freqs_mhz: Grade de frequências (MHz).
            top_n: Número de fontes a retornar.

        Returns:
            Lista das N fontes mais brilhantes (ordenadas por fluxo médio decrescente).
        """
        freqs = np.asarray(freqs_mhz, dtype=float)
        scored: list[tuple[float, RadioSource]] = []
        for src in sources:
            flux_arr = RadioSourceSignalEvaluator.evaluate_flux(src, freqs)
            mean_flux = float(np.mean(flux_arr))
            scored.append((mean_flux, src))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [src for _, src in scored[:top_n]]
