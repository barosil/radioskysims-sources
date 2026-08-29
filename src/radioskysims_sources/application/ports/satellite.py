"""Contrato de repositorio de satelites da camada de Application."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radioskysims_sources.domain.satellite import Satellite


class SatelliteRepositoryPort(ABC):
    """
    Porta para acesso e curadoria de satélites.
    """

    @abstractmethod
    def query_sats(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        raise NotImplementedError

    @abstractmethod
    def query_constellation(
        self,
        constellations: list[str],
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        raise NotImplementedError

    @abstractmethod
    def update_orbital_elements_sync(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_satellites_from_celestrak(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def verify(self, epoch: datetime | None = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self, kind: str = "all") -> None:
        raise NotImplementedError
