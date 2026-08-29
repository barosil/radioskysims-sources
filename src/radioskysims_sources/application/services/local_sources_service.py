"""\
Módulo: src/radioskysims/application/services/local_sources_service.py

Descrição:
    Serviço de aplicação para o agregado LocalSources com foco em sinal solar.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from radioskysims_core.domain.common import Observer, TimeRange
from radioskysims_sources.domain.localsources.models import (
    LocalSource,
    LocalSources,
    LocalSourceTrack,
    SolarFlux,
)
from radioskysims_sources.application.ports.localsources import (
    LocalSourceTrackProviderPort,
    SolarFluxDownloaderPort,
    SolarFluxRepositoryPort,
)


class LocalSourcesService:
    """Orquestra fontes locais e provê acesso a sinal solar em janela temporal."""

    def __init__(
        self,
        solar_flux_downloader: SolarFluxDownloaderPort,
        solar_flux_repository: SolarFluxRepositoryPort,
        local_source_track_provider: LocalSourceTrackProviderPort | None = None,
        default_window_hours: int = 2,
    ) -> None:
        self._downloader = solar_flux_downloader
        self._repository = solar_flux_repository
        self._track_provider = local_source_track_provider
        self._default_window_hours = default_window_hours

    def build_local_sources(self) -> LocalSources:
        """Retorna agregado inicial de fontes locais (MVP)."""
        return LocalSources(
            sources=[
                LocalSource(name="Sun", kind="sun", has_track=True, has_signal=True),
            ]
        )

    def get_solar_flux(
        self,
        center_utc: datetime,
        window_hours: int | None = None,
    ) -> SolarFlux:
        """Obtém SolarFlux via cache local + fallback de download externo."""
        win = window_hours if window_hours is not None else self._default_window_hours
        center = center_utc.astimezone(UTC).replace(tzinfo=None)
        half = timedelta(hours=win / 2)
        start_utc = center - half
        end_utc = center + half

        cached = self._repository.load_interval(start_utc=start_utc, end_utc=end_utc)
        if cached is not None:
            return cached

        downloaded = self._downloader.fetch(center_utc=center, window_hours=win)
        return self._repository.save(downloaded)

    def get_local_source_track(
        self,
        source_name: str,
        observer: Observer,
        time_range: TimeRange,
    ) -> LocalSourceTrack:
        """Retorna track de uma fonte local no referencial do observador."""
        if self._track_provider is None:
            msg = (
                "LocalSourcesService sem provider de track configurado. "
                "Injete LocalSourceTrackProviderPort (ex.: Skyfield)."
            )
            raise RuntimeError(msg)

        return self._track_provider.compute_track(
            source_name=source_name,
            observer=observer,
            time_range=time_range,
        )
