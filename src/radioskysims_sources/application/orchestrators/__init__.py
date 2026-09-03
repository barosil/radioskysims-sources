"""\
Pacote: radioskysims_sources.application.orchestrators

Orquestradores/fachadas de alto nível (UX — Onda 3/R6) do contexto de fontes
do RadioSkySim. Escondem bootstrap/repositórios e expõem uma API única:

    SatelliteOrchestrator, LocalSourcesOrchestrator, RadioSourcesOrchestrator,
    ContinuumSourceOrchestrator

Copyright BINGO Collaboration
"""

from radioskysims_sources.application.orchestrators.continuum_source_orchestrator import (
    ContinuumSourceOrchestrator,
)
from radioskysims_sources.application.orchestrators.local_sources_orchestrator import (
    LocalSourcesOrchestrator,
)
from radioskysims_sources.application.orchestrators.radio_sources_orchestrator import (
    RadioSourcesOrchestrator,
)
from radioskysims_sources.application.orchestrators.satellite_orchestrator import (
    SatelliteOrchestrator,
)

__all__ = [
    "ContinuumSourceOrchestrator",
    "LocalSourcesOrchestrator",
    "RadioSourcesOrchestrator",
    "SatelliteOrchestrator",
]
