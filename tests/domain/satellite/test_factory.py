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
    def fake_select(satids, epoch_jd):  # noqa: ARG001
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


def test_create_from_norad_ids_requires_repository():
    factory = SatelliteFactory()

    with pytest.raises(ValueError, match="Repositório não foi configurado"):
        factory.create_from_norad_ids(
            ids=[1],
            epoch=Time("2024-01-01").to_datetime(),
        )


def test_create_from_norad_ids_requires_satids():
    repo = MagicMock()
    repo.load_elements.return_value = []

    factory = SatelliteFactory(repository=repo)

    with pytest.raises(ValueError, match="Nenhum SatID encontrado"):
        factory.create_from_norad_ids(
            ids=[1],
            epoch=Time("2024-01-01").to_datetime(),
        )


def test_create_from_norad_ids_requires_orbital_elements():
    repo = MagicMock()

    sat = FakeSatID(1)
    sat.elements = []

    repo.load_elements.return_value = [sat]

    factory = SatelliteFactory(repository=repo)

    with pytest.raises(ValueError, match="Sem elementos orbitais"):
        factory.create_from_norad_ids(
            ids=[1],
            epoch=Time("2024-01-01").to_datetime(),
        )


def test_create_from_norad_ids_success(monkeypatch):
    repo = MagicMock()
    repo.load_elements.return_value = [
        FakeSatID(1),
        FakeSatID(2),
    ]

    selected = [
        {"norad_id": 1},
        {"norad_id": 2},
    ]

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.TLESelector.select",
        lambda _satids, _epoch_jd: selected
    )

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.Satellite.model_validate",
        lambda data: FakeSatellite(data["norad_id"]),
    )

    factory = SatelliteFactory(repository=repo)

    result = factory.create_from_norad_ids(
        ids=[1, 2],
        epoch=Time("2024-01-01").to_datetime(),
    )

    assert [sat.norad_id for sat in result] == [1, 2]
    repo.load_elements.assert_called_once_with([1, 2])


def test_create_from_norad_ids_ignores_invalid_satellite(monkeypatch):
    repo = MagicMock()
    repo.load_elements.return_value = [
        FakeSatID(1),
        FakeSatID(2),
    ]

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.TLESelector.select",
        lambda _satids, _epoch_jd: [
            {"norad_id": 1},
            {"norad_id": 2},
        ],
    )

    def fake_validate(data):
        if data["norad_id"] == 1:
             msg = "invalid"
             raise ValueError(msg)
        return FakeSatellite(data["norad_id"])

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.Satellite.model_validate",
        fake_validate,
    )

    factory = SatelliteFactory(repository=repo)

    result = factory.create_from_norad_ids(
        ids=[1, 2],
        epoch=Time("2024-01-01").to_datetime(),
    )

    assert [sat.norad_id for sat in result] == [2]


def test_create_from_norad_ids_fails_when_all_selected_are_invalid(
    monkeypatch,
):
    repo = MagicMock()
    repo.load_elements.return_value = [FakeSatID(1)]

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.TLESelector.select",
        lambda _satids, _epoch_jd: [{"norad_id": 1}],
    )

    def invalid_satellite(_data):
        msg = "invalid"
        raise ValueError(msg)


    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.Satellite.model_validate",
        invalid_satellite,
    )

    factory = SatelliteFactory(repository=repo)

    with pytest.raises(ValueError, match="Nenhum satélite válido"):
        factory.create_from_norad_ids(
            ids=[1],
            epoch=Time("2024-01-01").to_datetime(),
        )


def test_create_from_constellations_requires_repository():
    factory = SatelliteFactory()

    with pytest.raises(ValueError, match="Repositório não foi configurado"):
        factory.create_from_constellations(
            constellations=["GPS"],
            epoch=Time("2024-01-01").to_datetime(),
        )


def test_create_from_constellations_requires_satids():
    repo = MagicMock()
    repo.load_elements_from_constellation.return_value = []

    factory = SatelliteFactory(repository=repo)

    with pytest.raises(
        ValueError,
        match="Nenhum SatID encontrado para constelações",
    ):
        factory.create_from_constellations(
            constellations=["GPS"],
            epoch=Time("2024-01-01").to_datetime(),
        )


def test_create_from_constellations_success(monkeypatch):
    repo = MagicMock()

    gps = FakeSatID(1)
    galileo = FakeSatID(2)

    repo.load_elements_from_constellation.side_effect = [
        [gps],
        [galileo],
    ]

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.TLESelector.select",
        lambda satids, _epoch_jd: [
            {"norad_id": sat.norad_id}
            for sat in satids
        ],
    )

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.Satellite.model_validate",
        lambda data: FakeSatellite(data["norad_id"]),
    )

    factory = SatelliteFactory(repository=repo)

    result = factory.create_from_constellations(
        constellations=["GPS", "GALILEO"],
        epoch=Time("2024-01-01").to_datetime(),
    )

    assert [sat.norad_id for sat in result] == [1, 2]

    assert repo.load_elements_from_constellation.call_args_list == [
        ((["GPS"],),),
        ((["GALILEO"],),),
    ]


def test_create_from_constellations_fails_when_all_selected_invalid(
    monkeypatch,
):
    repo = MagicMock()
    repo.load_elements_from_constellation.return_value = [FakeSatID(1)]

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.TLESelector.select",
        lambda _satids, _epoch_jd: [{"norad_id": 1}],
    )

    def invalid_satellite(_data):
        msg = "invalid"
        raise ValueError(msg)


    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.factory.Satellite.model_validate",
        invalid_satellite,
    )

    factory = SatelliteFactory(repository=repo)

    with pytest.raises(ValueError, match="Nenhum satélite válido"):
        factory.create_from_constellations(
            constellations=["GPS"],
            epoch=Time("2024-01-01").to_datetime(),
        )
