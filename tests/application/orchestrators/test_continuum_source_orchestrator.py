"""\
Módulo: tests/application/orchestrators/test_continuum_source_orchestrator.py

Descrição:
    Testes da fachada ``ContinuumSourceOrchestrator`` (UX Onda 3/R6) com um
    use case fake — sem PySM3/mapas reais.

Copyright BINGO Collaboration
"""

from __future__ import annotations

from radioskysims_sources import ContinuumSourceOrchestrator
from radioskysims_sources.application.orchestrators.continuum_source_orchestrator import (
    ContinuumSourceOrchestrator as _Orch,
)


class FakeUseCase:
    """Fake de GenerateGalaxyMapsUseCase: registra chamadas."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def execute(self, **kwargs: object) -> str:
        self.kwargs = dict(kwargs)
        return "maps-ok"


def test_galaxy_maps_delega_para_use_case():
    fake = FakeUseCase()
    orch = ContinuumSourceOrchestrator(use_case=fake)  # type: ignore[arg-type]
    result = orch.galaxy_maps(
        source="galaxy",  # type: ignore[arg-type]
        frequency_range="fr",  # type: ignore[arg-type]
        nside=64,
    )
    assert result == "maps-ok"
    assert fake.kwargs == {
        "source": "galaxy",
        "frequency_range": "fr",
        "nside": 64,
        "persist": False,
        "overwrite": False,
    }


def test_from_generator_monta_use_case():
    gen = object()
    orch = _Orch.from_generator(gen)  # type: ignore[arg-type]
    assert orch is not None
    assert "ContinuumSourceOrchestrator" in repr(orch)


def test_default_monta_com_pysm():
    orch = ContinuumSourceOrchestrator.default()
    assert isinstance(orch, ContinuumSourceOrchestrator)
    assert "ContinuumSourceOrchestrator" in repr(orch)
