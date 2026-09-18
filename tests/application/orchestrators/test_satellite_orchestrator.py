"""\
Módulo: tests/application/orchestrators/test_satellite_orchestrator.py

Descrição:
    Testes da fachada ``SatelliteOrchestrator`` (UX Onda 3/R6) com um
    ``SatelliteManager`` fake — sem banco/rede.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import builtins
import importlib

import pytest
from radioskysims_sources import SatelliteOrchestrator
import radioskysims_sources.application.orchestrators.satellite_orchestrator as orchestrator_module
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


def test_create_usa_manager_default_quando_nao_injetado(monkeypatch):
    fake = FakeManager()
    calls = {}

    def fake_default_manager(_cls, *, epoch, settings_file=None):
        calls["args"] = (epoch, settings_file)
        return fake

    monkeypatch.setattr(_Orch, "_default_manager", classmethod(fake_default_manager))

    orch = _Orch.create_from_constellation("GPS", epoch="epoch")

    assert orch._manager is fake
    assert calls["args"] == ("epoch", None)


def test_create_rejeita_constelacao_vazia():
    with pytest.raises(ValueError):
        _Orch.create_from_constellation([], manager=FakeManager())


def test_create_converte_erro_do_manager_em_runtime_error():
    class FailingManager(FakeManager):
        def create_from_constellations(self, constellations, *, epoch=None):
            _ = (constellations, epoch)
            message = "catálogo vazio"
            raise ValueError(message)

    with pytest.raises(RuntimeError, match="catálogo vazio") as caught:
        _Orch.create_from_constellation("GPS", manager=FailingManager())

    assert isinstance(caught.value.__cause__, ValueError)


def test_default_manager_monta_dependencias_e_aplica_settings(monkeypatch):
    calls = {}
    config_loader = importlib.import_module("radioskysims_core.config.config_loader")
    db_engine = importlib.import_module("radioskysims_core.db.engine")
    builders = importlib.import_module("radioskysims_sources.bootstrap.builders")
    psd_module = importlib.import_module(
        "radioskysims_sources.infrastructure.signal.gnsstools_psd"
    )

    def set_settings_file(path):
        calls["settings"] = path

    class FakeDatabase:
        def __init__(self):
            calls["database"] = self

    class FakePsdProvider:
        pass

    def build_repositories(db):
        calls["repositories_db"] = db
        return {"satellite_repository": "sat-repo", "signal_repository": "signal-repo"}

    monkeypatch.setattr(config_loader, "set_settings_file", set_settings_file)
    monkeypatch.setattr(db_engine, "Database", FakeDatabase)
    monkeypatch.setattr(builders, "build_repositories", build_repositories)
    monkeypatch.setattr(psd_module, "GnsstoolsAnalyticPsdProvider", FakePsdProvider)

    class CapturingManager:
        def __init__(self, **kwargs):
            calls["manager"] = kwargs

    monkeypatch.setattr(orchestrator_module, "SatelliteManager", CapturingManager)

    result = _Orch._default_manager(epoch="epoch", settings_file="settings.toml")

    assert isinstance(result, CapturingManager)
    assert calls["settings"] == "settings.toml"
    assert calls["repositories_db"] is calls["database"]
    assert calls["manager"] == {
        "repository": "sat-repo",
        "signal_repository": "signal-repo",
        "epoch": "epoch",
        "psd_provider": calls["manager"]["psd_provider"],
    }
    assert isinstance(calls["manager"]["psd_provider"], FakePsdProvider)


def test_default_manager_informa_falha_de_import(monkeypatch):
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "radioskysims_core.config.config_loader":
            message = "config ausente"
            raise ImportError(message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(RuntimeError, match="config ausente") as caught:
        _Orch._default_manager(epoch=None)

    assert isinstance(caught.value.__cause__, ImportError)


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
