from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from astropy.time import Time

from radioskysims_core.domain.common import TimeRange
from radioskysims_sources.domain.satellite.factory import SatelliteFactory


class FakeElement:
    def __init__(self, epoch):
        self.epoch = epoch


class FakeSatID:
    def __init__(self, norad_id):
        self.norad_id = norad_id
        self.name = f"SAT-{norad_id}"
        self.constellation = "TEST"
        self.elements = [FakeElement(Time("2024-01-01"))]


class FakeSatellite:
    def __init__(self, norad_id, **_):
        self.norad_id = norad_id


def test_factory_build_success(monkeypatch):
    repo = MagicMock()

    # Mock os métodos chamados pela factory
    repo.load_elements.return_value = [FakeSatID(1), FakeSatID(2)]
    repo.load_elements_from_constellation.return_value = []
    repo.reset.return_value = repo
    repo.process.return_value = None
    repo.satellites = [FakeSatID(1), FakeSatID(2)]  # Satélites após processamento

    # Mock TLESelector.select para retornar satélites fake
    def fake_select(satids, epoch_jd):
        return [FakeSatellite(1), FakeSatellite(2)]
    
    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.TLESelector.select",
        fake_select,
    )

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.Satellite",
        FakeSatellite,
    )

    tr = TimeRange(
        start_jd=Time("2024-01-01").jd,
        end_jd=Time("2024-01-03").jd,
        resolution_sec=3600.0,  # 1 hora
    )

    factory = SatelliteFactory()

    result = factory.build(
        tr=tr,
        repository=repo,
        ids=[1, 2],
        interval=1,
    )

    assert len(result) == 2  # 2 slices
    for sats in result.values():
        assert len(sats) == 2


def test_factory_requires_input():
    factory = SatelliteFactory()

    with pytest.raises(ValueError, match="ids ou constellations"):
        factory.build(
            tr=MagicMock(),
            repository=MagicMock(),
        )


def test_factory_no_satids():
    repo = MagicMock()
    repo.query_sats.return_value = []
    repo.query_constellation.return_value = []

    factory = SatelliteFactory()

    with pytest.raises(ValueError, match="Nenhum SatID"):
        factory.build(
            tr=MagicMock(),
            repository=repo,
            ids=[999],
        )
