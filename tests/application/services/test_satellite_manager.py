"""\
Módulo: tests/application/services/test_satellite_manager.py

Descrição:
    Testes da facade de alto nível ``SatelliteManager`` (dedupe, TLE,
    blocos de construção e erros de uso) sem dependência de banco.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from radioskysims_core.domain.common import FrequencyRange, TimeRange
import radioskysims_sources.application.services.satellite_manager as manager_module
from radioskysims_sources.application.services.satellite_manager import SatelliteManager
from radioskysims_sources.domain.satellite.factory import SatelliteFactory
from radioskysims_sources.domain.satellite.models import OrbitalElement, Satellite

EPOCH = datetime(2026, 8, 30, 12, 0)


def _sat(norad: int, name: str, epoch: datetime) -> Satellite:
    return Satellite(
        norad_id=norad,
        name=name,
        constellation="GPS",
        epoch=epoch,
        element=OrbitalElement(epoch=epoch, norad_id=norad, tle_line1="1 x", tle_line2="2 x"),
    )


class _FakeRepo:
    """Repositório fake que devolve duplicatas por época (como o catálogo)."""

    def __init__(self, sats: list[Satellite]) -> None:
        self.sats = sats

    def query_sats(self, norad_ids=None, epoch=None) -> list[Satellite]:  # noqa: ARG002
        return self.sats


class _FakeFactory(SatelliteFactory):
    """Factory fake que devolve o mesmo conjunto para qualquer chamada."""

    def __init__(self, sats: list[Satellite], repo=None) -> None:
        super().__init__(repo)
        self.sats = sats

    def create_from_constellations(self, constellations, epoch):  # noqa: ARG002
        return list(self.sats)

    def create_from_norad_ids(self, norad_ids, epoch):  # noqa: ARG002
        return [s for s in self.sats if s.norad_id in norad_ids]


def test_manager_epoch_is_aware():
    mgr = SatelliteManager(factory=_FakeFactory([]), epoch=EPOCH)
    assert mgr.epoch.tzinfo is not None


def test_manager_builds_default_factory_with_repository():
    repository = object()
    mgr = SatelliteManager(repository=repository)
    assert mgr.factory.repository is repository


def test_manager_epoch_prefers_argument_and_uses_now_when_unconfigured():
    mgr = SatelliteManager(factory=_FakeFactory([]))
    requested = datetime(2026, 8, 30, 15, 0, tzinfo=timezone(timedelta(hours=3)))

    resolved = mgr._epoch(requested)
    now = mgr._epoch(None)

    assert resolved == datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    assert now.tzinfo == timezone.utc


def test_manager_dedupe_by_norad_keeps_closest():
    # mesma norad em duas épocas (uma dentro, outra fora do intervalo)
    s1 = _sat(25544, "ISS-A", EPOCH)
    s2 = _sat(25544, "ISS-B", EPOCH + timedelta(days=3))
    s3 = _sat(27663, "GPS-B", EPOCH)
    mgr = SatelliteManager(factory=_FakeFactory([s1, s2, s3]), epoch=EPOCH)
    sats = mgr.create_from_constellations(["GPS"])
    norads = [s.norad_id for s in sats]
    assert norads.count(25544) == 1
    assert sorted(norads) == [25544, 27663]
    kept = next(s for s in sats if s.norad_id == 25544)
    assert kept.name == "ISS-A"  # mais próximo da época


def test_manager_dedupe_replaces_with_later_closer_satellite():
    first = _sat(25544, "ISS-A", EPOCH - timedelta(days=3))
    closest = _sat(25544, "ISS-B", EPOCH + timedelta(days=1))
    epoch = EPOCH.replace(tzinfo=timezone.utc)

    result = SatelliteManager._dedupe([first, closest], epoch)

    assert result == [closest]


def test_manager_create_from_norad_ids_forwards_epoch_and_deduplicates():
    sats = [_sat(1, "SAT-1", EPOCH), _sat(1, "SAT-1-old", EPOCH + timedelta(days=2))]
    factory = _FakeFactory(sats)
    mgr = SatelliteManager(factory=factory, epoch=EPOCH)

    result = mgr.create_from_norad_ids([1])

    assert result == [sats[0]]
    assert mgr.satellites is result


def test_manager_query_sats_forwards_ids_and_epoch():
    sats = [_sat(7, "SAT-7", EPOCH)]
    repository = _FakeRepo(sats)
    mgr = SatelliteManager(factory=_FakeFactory([], repo=repository), epoch=EPOCH)

    result = mgr.query_sats([7])

    assert result == sats
    assert mgr.satellites is result


def test_manager_get_tle():
    mgr = SatelliteManager(factory=_FakeFactory([_sat(25544, "ISS", EPOCH)]), epoch=EPOCH)
    mgr.create_from_constellations(["GPS"])
    tle = mgr.get_tle()
    assert tle[0]["norad_id"] == 25544
    assert tle[0]["tle_line1"] == "1 x"
    assert "epoch" in tle[0]


def test_manager_get_tle_returns_all_domain_fields():
    sat = _sat(25544, "ISS", EPOCH)
    mgr = SatelliteManager(factory=_FakeFactory([sat]))
    mgr.satellites = [sat]

    assert mgr.get_tle() == [
        {
            "norad_id": 25544,
            "name": "ISS",
            "constellation": "GPS",
            "epoch": EPOCH.replace(tzinfo=timezone.utc).isoformat(),
            "tle_line1": "1 x",
            "tle_line2": "2 x",
        }
    ]


def test_manager_build_requires_satellites():
    mgr = SatelliteManager(factory=_FakeFactory([]), epoch=EPOCH)
    tr = TimeRange.from_iso("2026-08-30T10:00:00", "2026-08-30T12:00:00", 120)
    fr = FrequencyRange(min_mhz=1000.0, max_mhz=1250.0, resolution_mhz=0.5)
    with pytest.raises(ValueError, match="Nenhum satélite selecionado"):
        mgr.build(tr, fr)


def test_manager_build_signals_requires_signal_repository():
    sats = [_sat(25544, "ISS", EPOCH)]
    mgr = SatelliteManager(factory=_FakeFactory(sats), epoch=EPOCH)
    mgr.create_from_constellations(["GPS"])
    tr = TimeRange.from_iso("2026-08-30T10:00:00", "2026-08-30T12:00:00", 120)
    fr = FrequencyRange(min_mhz=1000.0, max_mhz=1250.0, resolution_mhz=0.5)
    with pytest.raises(ValueError, match="signal_repository"):
        mgr.build(tr, fr)


def test_manager_block_accepts_explicit_satellites():
    sat = _sat(25544, "ISS", EPOCH)
    timerange = TimeRange.from_iso("2026-08-30T10:00:00", "2026-08-30T12:00:00", 120)
    mgr = SatelliteManager(factory=_FakeFactory([]))

    block = mgr._block(timerange, [sat])

    assert block == {
        float(timerange.start_jd): {
            "timerange": timerange,
            "satellites": [sat],
        }
    }


def test_manager_assembler_wires_signal_dependencies(monkeypatch):
    calls = {}

    class FakeSignalFactory:
        def __init__(self, **kwargs):
            calls["signal"] = kwargs

    class FakeAssembler:
        def __init__(self, **kwargs):
            calls["assembler"] = kwargs

    class FakeTrackFactory:
        pass

    monkeypatch.setattr(manager_module, "SignalFactory", FakeSignalFactory)
    monkeypatch.setattr(manager_module, "SatelliteAssembler", FakeAssembler)

    signal_repository = object()
    psd_provider = object()
    frequency_range = FrequencyRange(min_mhz=1000.0, max_mhz=1250.0, resolution_mhz=0.5)
    mgr = SatelliteManager(
        factory=_FakeFactory([]),
        signal_repository=signal_repository,
        psd_provider=psd_provider,
        track_factory=FakeTrackFactory(),
    )

    assembler = mgr._assembler(frequency_range)

    assert isinstance(assembler, FakeAssembler)
    assert calls["signal"] == {
        "frequency_range": frequency_range,
        "signal_repo": signal_repository,
        "psd_provider": psd_provider,
    }
    assert calls["assembler"]["signal_factory"].__class__ is FakeSignalFactory
    assert isinstance(calls["assembler"]["track_factory"], FakeTrackFactory)


def test_manager_build_tracks_delegates_block():
    calls = {}
    dataset = object()

    class FakeTrackFactory:
        def build(self, block):
            calls["block"] = block
            return SimpleNamespace(dataset=dataset)

    sat = _sat(25544, "ISS", EPOCH)
    timerange = TimeRange.from_iso("2026-08-30T10:00:00", "2026-08-30T12:00:00", 120)
    mgr = SatelliteManager(
        factory=_FakeFactory([]),
        track_factory=FakeTrackFactory(),
    )

    assert mgr.build_tracks(timerange, sats=[sat]) is dataset
    assert list(calls["block"].values())[0]["satellites"] == [sat]


def test_manager_build_signals_delegates_block(monkeypatch):
    calls = {}
    assembled = object()

    class FakeAssembler:
        def build_signals(self, block):
            calls["block"] = block
            return assembled

    monkeypatch.setattr(SatelliteManager, "_assembler", lambda _self, _fr: FakeAssembler())
    sat = _sat(25544, "ISS", EPOCH)
    timerange = TimeRange.from_iso("2026-08-30T10:00:00", "2026-08-30T12:00:00", 120)
    frequency_range = FrequencyRange(min_mhz=1000.0, max_mhz=1250.0, resolution_mhz=0.5)
    mgr = SatelliteManager(factory=_FakeFactory([]))

    assert mgr.build_signals(timerange, frequency_range, sats=[sat]) is assembled
    assert list(calls["block"].values())[0]["satellites"] == [sat]


def test_manager_build_delegates_block(monkeypatch):
    calls = {}
    assembled = object()

    class FakeAssembler:
        def build_signals(self, block):
            calls["block"] = block
            return assembled

    monkeypatch.setattr(SatelliteManager, "_assembler", lambda _self, _fr: FakeAssembler())
    sat = _sat(25544, "ISS", EPOCH)
    timerange = TimeRange.from_iso("2026-08-30T10:00:00", "2026-08-30T12:00:00", 120)
    frequency_range = FrequencyRange(min_mhz=1000.0, max_mhz=1250.0, resolution_mhz=0.5)
    mgr = SatelliteManager(factory=_FakeFactory([]))

    assert mgr.build(timerange, frequency_range, sats=[sat]) is assembled
    assert list(calls["block"].values())[0]["satellites"] == [sat]


def test_manager_query_sats_requires_repo():
    mgr = SatelliteManager(factory=_FakeFactory([]), epoch=EPOCH)
    with pytest.raises(ValueError, match="Repositório não configurado"):
        mgr.query_sats()
