"""\
Módulo: tests/application/services/test_satellite_manager.py

Descrição:
    Testes da facade de alto nível ``SatelliteManager`` (dedupe, TLE,
    blocos de construção e erros de uso) sem dependência de banco.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from radioskysims_core.domain.common import FrequencyRange, TimeRange
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
        element=OrbitalElement(
            epoch=epoch, norad_id=norad, tle_line1="1 x", tle_line2="2 x"
        ),
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


def test_manager_get_tle():
    mgr = SatelliteManager(factory=_FakeFactory([_sat(25544, "ISS", EPOCH)]), epoch=EPOCH)
    mgr.create_from_constellations(["GPS"])
    tle = mgr.get_tle()
    assert tle[0]["norad_id"] == 25544
    assert tle[0]["tle_line1"] == "1 x"
    assert "epoch" in tle[0]


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


def test_manager_query_sats_requires_repo():
    mgr = SatelliteManager(factory=_FakeFactory([]), epoch=EPOCH)
    with pytest.raises(ValueError, match="Repositório não configurado"):
        mgr.query_sats()
