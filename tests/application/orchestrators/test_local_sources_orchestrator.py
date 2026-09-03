"""\
Módulo: tests/application/orchestrators/test_local_sources_orchestrator.py

Descrição:
    Testes da fachada ``LocalSourcesOrchestrator`` (UX Onda 3/R6).

Copyright BINGO Collaboration
"""

from __future__ import annotations

from radioskysims_sources import LocalSourcesOrchestrator
from radioskysims_sources.application.services.local_sources_manager import Sun


def test_create_sun_default():
    sun = LocalSourcesOrchestrator.create_sun()
    assert isinstance(sun, Sun)
    assert sun.flux is not None
    assert sun.track is not None


def test_create_sun_exposto_na_raiz_do_pacote():
    assert hasattr(LocalSourcesOrchestrator, "create_sun")
