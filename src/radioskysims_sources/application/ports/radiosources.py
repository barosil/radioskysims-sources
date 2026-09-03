"""\
Módulo: src/radioskysims/application/ports/radiosources.py

Ports (interfaces) para operacoes de RadioSources na camada de Application.

Define o contrato entre o dominio de fontes de radio e as implementacoes
concretas de ingestão de catálogos, persistência em banco de dados e
consulta/avaliação de fluxo em frequência.

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from radioskysims_core.domain.common import Observer, TimeRange

from radioskysims_sources.domain.radiosources.models import RadioSource, RadioSourceFluxSample
from radioskysims_sources.domain.radiosources.tracks import RadioSourceTrack


class RadioSourceTrackProviderPort(ABC):
    """Porta para cálculo de tracks de fontes de rádio fixas.

    Abstrai a implementação concreta (Skyfield) para que os casos de uso
    dependam apenas do contrato; o adapter é fornecido pelo composition root.
    """

    @abstractmethod
    def compute_track(
        self,
        sources: list[RadioSource],
        observer: Observer,
        time_range: TimeRange,
    ) -> RadioSourceTrack:
        """Calcula tracks (alt-az) das fontes no referencial do observador."""
        raise NotImplementedError


class CatalogIngestPort(ABC):
    """Porta para ingestão de catálogos externos de fontes de rádio.

    Responsável por abstrair o acesso a serviços TAP/ADQL, arquivos
    locais (FITS, HDF5, CSV, etc.) ou outros provedores remotos,
    retornando dados tabulares que possam ser normalizados em
    entidades de domínio.
    """

    @abstractmethod
    def ingest_from_tap(self, survey: str, adql_query: str) -> Any:
        """Executa uma query TAP/ADQL em um survey e retorna tabela de dados."""
        raise NotImplementedError

    @abstractmethod
    def ingest_from_file(self, path: Path, format: str) -> Any:
        """Lê um catálogo local no formato especificado e retorna tabela de dados."""
        raise NotImplementedError


class RadioSourcesRepositoryPort(ABC):
    """Porta para persistência e consulta de entidades :class:`RadioSource`.

    Normalmente implementada sobre PostgreSQL, mas sem expor detalhes
    de infraestrutura para o domínio ou casos de uso.
    """

    @abstractmethod
    def save_sources(self, sources: list[RadioSource]) -> None:
        """Persiste ou atualiza um conjunto de fontes no repositório."""
        raise NotImplementedError

    @abstractmethod
    def list_sources_in_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        min_flux_jy: float | None = None,
    ) -> list[RadioSource]:
        """Retorna fontes dentro de uma região circular no céu."""
        raise NotImplementedError


class RadioSourcesCatalogPort(ABC):
    """Porta de alto nível para consulta de fontes normalizadas.

    Esta interface é a visão "amigável" que contexts como Simulation
    e TOD devem utilizar para obter fontes relevantes em uma região.
    """

    @abstractmethod
    def query_sources(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        min_flux_jy: float | None = None,
        freq_mhz: float | None = None,
        limit: int | None = None,
    ) -> list[RadioSource]:
        """Consulta fontes em uma região do céu com possíveis cortes em fluxo."""
        raise NotImplementedError


class RadioSourcesFluxPort(ABC):
    """Porta para avaliação de fluxo de fontes em frequências específicas."""

    @abstractmethod
    def evaluate_flux(
        self,
        sources: list[RadioSource],
        freqs_mhz: list[float],
    ) -> list[RadioSourceFluxSample]:
        """Avalia o fluxo das fontes nas frequências solicitadas."""
        raise NotImplementedError


class RadioSourcesSkyFluxPort(ABC):
    """Porta para cálculo de contribuição de fontes em uma linha de visada.

    Usada tipicamente pelo TOD Context para obter a contribuição de
    foreground discreto em uma direção e lista de frequências.
    """

    @abstractmethod
    def evaluate_sky_contribution(
        self,
        theta_deg: float,
        phi_deg: float,
        freqs_mhz: list[float],
        radius_deg: float,
        min_flux_jy: float | None = None,
    ) -> list[RadioSourceFluxSample]:
        """Calcula a contribuição de fontes em torno de (theta_deg, phi_deg).

        Todas as frequências devem ser fornecidas em MHz e ângulos em graus,
        conforme convenção global do projeto.
        """
        raise NotImplementedError
