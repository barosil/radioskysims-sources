"""\
Módulo: src/radioskysims/application/ports/localsources.py

Descrição:
    Ports de aplicacao para o agregado LocalSources,
    com foco no fluxo solar (download, cache e persistência).

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from radioskysims_core.domain.common import Observer, TimeRange
from radioskysims_sources.domain.localsources.models import LocalSourceTrack, SolarFlux


class SolarFluxDownloaderPort(ABC):
    """Porta para obtenção de dados brutos de fluxo solar."""

    @abstractmethod
    def fetch(self, center_utc: datetime, window_hours: int = 2) -> SolarFlux:
        """Obtém um ``SolarFlux`` centrado em ``center_utc``."""
        raise NotImplementedError


class SolarFluxRepositoryPort(ABC):
    """Porta para cache/persistência de ``SolarFlux``."""

    @abstractmethod
    def load_interval(self, start_utc: datetime, end_utc: datetime) -> SolarFlux | None:
        """Retorna cache de SolarFlux que cobre o intervalo, se existir."""
        raise NotImplementedError

    @abstractmethod
    def save(self, solar_flux: SolarFlux) -> SolarFlux:
        """Persiste dados em NetCDF e registra metadados quando houver DB."""
        raise NotImplementedError


class LocalSourceTrackProviderPort(ABC):
    """Porta para cálculo de tracks de fontes locais no referencial do observador."""

    @abstractmethod
    def compute_track(
        self,
        source_name: str,
        observer: Observer,
        time_range: TimeRange,
    ) -> LocalSourceTrack:
        """Calcula track temporal para uma fonte local (ex.: Sun)."""
        raise NotImplementedError
