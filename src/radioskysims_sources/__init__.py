"""
Pacote: radioskysims_sources

Fontes de sinal do ecossistema RadioSkySim — satélites GNSS, fontes locais
(Sol/fluxo), fontes de rádio, cosmologia (continuum/HI) e catálogos.
Exporta o protocolo unificado ``SignalSource`` (fronteira sources↔simulation).

Copyright BINGO Collaboration
"""

from __future__ import annotations

__version__ = "0.1.0"

from .application.orchestrators import (
    ContinuumSourceOrchestrator,
    LocalSourcesOrchestrator,
    RadioSourcesOrchestrator,
    SatelliteOrchestrator,
)
from .domain.signal_source import SignalSource

__all__ = [
    "ContinuumSourceOrchestrator",
    "LocalSourcesOrchestrator",
    "RadioSourcesOrchestrator",
    "SatelliteOrchestrator",
    "SignalSource",
]
