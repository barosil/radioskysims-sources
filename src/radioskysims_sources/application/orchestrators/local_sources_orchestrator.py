"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/orchestrators/local_sources_orchestrator.py

Descrição:
    Fachada de alto nível (UX — Onda 3/R6) para fontes locais (Sol).

    Uso (notebook):
        >>> from radioskysims_sources import LocalSourcesOrchestrator
        >>> sun = LocalSourcesOrchestrator.create_sun()
        >>> sun.view_skychart()          # carta local ALT/AZ
        >>> sun.flux_density().view()    # fluxo × frequência

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from typing import Any

import numpy as np
from radioskysims_core.domain.common import Observer, TimeRange

from radioskysims_sources.application.services.local_sources_manager import (
    LocalSourcesManager,
    Sun,
)
from radioskysims_sources.domain.localsources.models import SolarFlux

__all__ = ["LocalSourcesOrchestrator"]


class LocalSourcesOrchestrator:
    """Orquestrador/fachada pública de fontes locais (Sol).

    Encapsula o ``LocalSourcesManager``/``Sun`` (track via skyfield e fluxo),
    escondendo downloader de fluxo solar e detalhes de domínio.
    """

    def __init__(
        self,
        manager: LocalSourcesManager | None = None,
        *,
        observer: Observer | None = None,
        time_range: TimeRange | None = None,
    ) -> None:
        self.manager = manager or LocalSourcesManager(observer=observer, time_range=time_range)

    # ──────────────────────────────────────────────────────────
    # Fábricas
    # ──────────────────────────────────────────────────────────
    @classmethod
    def create_sun(
        cls,
        *,
        observer: Observer | None = None,
        time_range: TimeRange | None = None,
        freqs_mhz: np.ndarray | list[float] | None = None,
        flux: SolarFlux | Any | None = None,
    ) -> Sun:
        """Retorna o Sol (fonte local) pronto para track/fluxo.

        Args:
            observer: Observador (default: BINGO/RN).
            time_range: Janela temporal (default: 1 dia centrado no Sol).
            freqs_mhz: Grade de frequências (MHz) para o fluxo.
            flux: ``SolarFlux`` real (ex.: download SRD/SRS); ``None`` → Sol
                calmo padrão (~150 SFU).
        """
        if observer is None and time_range is None and flux is None:
            return LocalSourcesManager().sun()
        return Sun.simulated(
            observer=observer,
            time_range=time_range,
            freqs_mhz=freqs_mhz,
            flux=flux,
        )

    # ──────────────────────────────────────────────────────────
    # Acesso via instância
    # ──────────────────────────────────────────────────────────
    def sun(self, **kwargs: Any) -> Sun:
        """Sol a partir do manager desta instância (observer/janela fixos)."""
        return self.manager.sun(**kwargs)

    def __repr__(self) -> str:
        return f"<LocalSourcesOrchestrator: {self.manager!r}>"
