"""\
Módulo: tests/application/orchestrators/test_satellite_orchestrator.py

Descrição:
    Testes da fachada ``SatelliteOrchestrator`` (UX Onda 3/R6) com um
    ``SatelliteManager`` fake — sem banco/rede.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import pytest
from radioskysims_sources import SatelliteOrchestrator
from radioskysims_sources.application.orchestrators.satellite_orchestrator import (
    SatelliteOrchestrator as _Orch,
)


class FakeManager:
    """Fake de SatelliteManager: registra chamadas e devolve marcadores."""

    def __init__(self) -> None:
        self.satellites = ["sat-1", "sat-2"]
        self.created: tuple[list[str], object] | None = None
        self.epoch_seen: object = None

    def create_from_constellations(self, constellations: list[str], *, epoch=None) -> list[str]:
        self.created = (list(constellations), epoch)
        self.epoch_seen = epoch
        return self.satellites

    def build_tracks(self, timerange):  # noqa: ARG002
        return "tracks"

    def build_signals(self, timerange, frequency_range):  # noqa: ARG002
        return "signals"

    def build(self, timerange, frequency_range):  # noqa: ARG002
        return "assembled"

    def get_tle(self):
        return [{"norad_id": 1, "name": "GPS BIIR-1"}]


def test_create_from_constellation_delega_para_manager():
    fake = FakeManager()
    orch = SatelliteOrchestrator.create_from_constellation("GPS", manager=fake)
    assert fake.created == (["GPS"], None)
    assert orch.satellites == ["sat-1", "sat-2"]
    assert orch.get_tle() == [{"norad_id": 1, "name": "GPS BIIR-1"}]


def test_create_aceita_lista_de_constelacoes():
    fake = FakeManager()
    _Orch.create_from_constellation(["GPS", "GALILEO"], manager=fake)
    assert fake.created == (["GPS", "GALILEO"], None)


def test_create_rejeita_constelacao_vazia():
    with pytest.raises(ValueError):
        _Orch.create_from_constellation([], manager=FakeManager())


def test_fluxo_delega_para_manager():
    fake = FakeManager()
    orch = _Orch.from_manager(fake)  # type: ignore[arg-type]
    assert orch.tracks("tr") == "tracks"  # type: ignore[arg-type]
    assert orch.signals("tr", "fr") == "signals"  # type: ignore[arg-type]
    assert orch.build("tr", "fr") == "assembled"  # type: ignore[arg-type]


def test_method_rinex_nao_implementado_ainda():
    fake = FakeManager()
    with pytest.raises(NotImplementedError):
        _Orch.create_from_constellation("GPS", method="RINEX", manager=fake)
    with pytest.raises(NotImplementedError):
        _Orch.from_rinex()
