"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/orchestrators/radio_sources_orchestrator.py

Descrição:
    Fachada de alto nível (UX — Onda 3/R6) para fontes de rádio (catálogo).

    Encapsula o ``RadioSourcesManager`` (simulado/banco/TAP), escondendo o
    import de ``application.services`` e o fallback do banco (aviso — R3).

    Uso (notebook):
        >>> from radioskysims_sources import RadioSourcesOrchestrator
        >>> rs = RadioSourcesOrchestrator.simulated(n_sources=100)
        >>> rs.select(min_flux_jy=1.0).view_mollweide()
        >>> rs.flux_density().view()

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from typing import Any

from radioskysims_sources.application.services.flux_view import FluxDensity
from radioskysims_sources.application.services.radio_sources_manager import (
    RadioSourcesManager,
)

__all__ = ["RadioSourcesOrchestrator"]


class RadioSourcesOrchestrator:
    """Orquestrador/fachada pública de fontes de rádio (catálogo)."""

    def __init__(self, manager: RadioSourcesManager) -> None:
        self._manager = manager

    # ──────────────────────────────────────────────────────────
    # Fábricas
    # ──────────────────────────────────────────────────────────
    @classmethod
    def simulated(
        cls,
        n_sources: int = 40,
        *,
        seed: int | None = 42,
        region: dict[str, float] | None = None,
    ) -> "RadioSourcesOrchestrator":
        """Catálogo sintético (demo, sem rede/banco)."""
        return cls(RadioSourcesManager.simulated(n_sources=n_sources, seed=seed, region=region))

    @classmethod
    def from_db(cls, **kwargs: Any) -> "RadioSourcesOrchestrator":
        """Carrega do PostgreSQL (tabela ``radio_sources``).

        Sem banco disponível ou tabela vazia, cai no catálogo simulado com
        aviso (R3 — nunca em silêncio).
        """
        return cls(RadioSourcesManager.get_radio_sources(**kwargs))

    @classmethod
    def from_tap(cls, **kwargs: Any) -> "RadioSourcesOrchestrator":
        """Ingere um catálogo REAL via TAP/ADQL (VizieR: NVSS + SPECFIND)."""
        return cls(RadioSourcesManager.from_tap(**kwargs))

    # ──────────────────────────────────────────────────────────
    # Acesso
    # ──────────────────────────────────────────────────────────
    @property
    def sources(self) -> list[Any]:
        """Catálogo ativo (seleção atual)."""
        return self._manager.sources

    def __len__(self) -> int:
        return len(self._manager)

    def summary(self) -> str:
        """Resumo em tabela das fontes (para CLI/print)."""
        return self._manager.summary()

    def list_properties(self, *args: Any, **kwargs: Any) -> Any:
        """DataFrame com as propriedades das fontes."""
        return self._manager.list_properties(*args, **kwargs)

    def select(self, **kwargs: Any) -> "RadioSourcesOrchestrator":
        """Seleciona/filtra o catálogo (ids, min_flux_jy, região...)."""
        return RadioSourcesOrchestrator(self._manager.select(**kwargs))

    def flux_density(self, *args: Any, **kwargs: Any) -> FluxDensity:
        """Curvas de fluxo (Jy) por fonte."""
        return self._manager.flux_density(*args, **kwargs)

    # ──────────────────────────────────────────────────────────
    # Visualização (deferida para radioskysims-viz)
    # ──────────────────────────────────────────────────────────
    def view_mollweide(self, *args: Any, **kwargs: Any) -> Any:
        """Mapa RA/Dec (Mollweide) do catálogo."""
        return self._manager.view_mollweide(*args, **kwargs)

    def view_flux_density(self, *args: Any, **kwargs: Any) -> Any:
        """Curvas de fluxo × frequência (plot)."""
        return self._manager.view_flux_density(*args, **kwargs)
