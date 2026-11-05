from __future__ import annotations

from datetime import timezone

import pytest
from astropy.time import Time

import radioskysims_sources.domain.satellite.selector as selector_module

from radioskysims_sources.domain.satellite.selector import TLESelector


class FakeElement:
    def __init__(self, epoch):
        self.epoch = epoch


class FakeSatID:
    def __init__(self, norad_id, elements):
        self.norad_id = norad_id
        self.name = f"SAT-{norad_id}"
        self.constellation = "TEST"
        self.elements = elements


@pytest.mark.usefixtures("stable_idxmin")
def test_select_closest_tle():
    epoch_jd = Time("2024-01-10").jd

    sat = FakeSatID(
        norad_id=123,
        elements=[
            FakeElement(Time("2024-01-01")),
            FakeElement(Time("2024-01-09")),  # mais próximo
            FakeElement(Time("2024-01-20")),
        ],
    )

    selected = TLESelector.select([sat], epoch_jd)

    assert len(selected) == 1
    assert selected[0]["norad_id"] == 123
    assert selected[0]["element"].epoch == Time("2024-01-09")


class DictElement(dict):
    """Dict compatível com o acesso elem.epoch feito pelo selector."""

    @property
    def epoch(self):
        return self["epoch"]


class ModelDumpElement:
    def __init__(self, epoch):
        self.epoch = epoch
        self.norad_id = 456

    def model_dump(self):
        return {
            "epoch": self.epoch,
            "norad_id": self.norad_id,
        }


class AttributeElement:
    def __init__(self, epoch):
        self.epoch = epoch
        self.norad_id = 789
        self.tle_line1 = "LINE-1"

    def __dir__(self):
        return [
            "epoch",
            "norad_id",
            "tle_line1",
            "broken_attribute",
        ]

    @property
    def broken_attribute(self):
        message = "unavailable"
        raise RuntimeError(message)


@pytest.fixture
def stable_idxmin(monkeypatch):
    class FakeIndexResult:
        def __init__(self, values):
            self.values = values

        def to_numpy(self):
            return self.values

    class FakeSelectedColumn:
        def __init__(self, frame):
            self.frame = frame

        def idxmin(self):
            best = {}

            for index, row in enumerate(self.frame.rows):
                norad_id = row["norad_id"]
                delta_epoch = row["delta_epoch"]

                current = best.get(norad_id)

                if current is None or delta_epoch < current[1]:
                    best[norad_id] = (
                        index,
                        delta_epoch,
                    )

            return FakeIndexResult([index for index, _delta in best.values()])

    class FakeGroupBy:
        def __init__(self, frame):
            self.frame = frame

        def __getitem__(self, key):
            assert key == "delta_epoch"
            return FakeSelectedColumn(self.frame)

    class FakeLoc:
        def __init__(self, frame):
            self.frame = frame

        def __getitem__(self, indices):
            return FakeDataFrame([self.frame.rows[index] for index in indices])

    class FakeDataFrame:
        def __init__(self, rows):
            self.rows = list(rows)

        def groupby(self, key):
            assert key == "norad_id"
            return FakeGroupBy(self)

        def __getitem__(self, columns):
            return FakeDataFrame([{column: row[column] for column in columns} for row in self.rows])

        @property
        def loc(self):
            return FakeLoc(self)

        def to_dict(self, orient):
            assert orient == "records"
            return list(self.rows)

    monkeypatch.setattr(
        selector_module.pd,
        "DataFrame",
        FakeDataFrame,
    )


@pytest.mark.usefixtures("stable_idxmin")
def test_select_converts_dict_element_to_orbital_element():
    epoch = Time("2024-01-09")

    element = DictElement(
        epoch=epoch,
        norad_id=123,
        tle_line1="LINE-1",
        tle_line2="LINE-2",
    )
    sat = FakeSatID(
        norad_id=123,
        elements=[element],
    )

    selected = TLESelector.select(
        [sat],
        Time("2024-01-10").jd,
    )

    orbital = selected[0]["element"]

    from radioskysims_sources.domain.satellite.models import OrbitalElement

    assert isinstance(orbital, OrbitalElement)
    assert orbital.norad_id == 123
    assert orbital.tle_line1 == "LINE-1"
    assert orbital.tle_line2 == "LINE-2"
    assert orbital.epoch == epoch.to_datetime(timezone=timezone.utc)


@pytest.mark.usefixtures("stable_idxmin")
def test_select_converts_model_dump_element_to_orbital_element():
    epoch = Time("2024-01-09")

    sat = FakeSatID(
        norad_id=456,
        elements=[
            ModelDumpElement(epoch),
        ],
    )

    selected = TLESelector.select(
        [sat],
        Time("2024-01-10").jd,
    )

    orbital = selected[0]["element"]

    from radioskysims_sources.domain.satellite.models import OrbitalElement

    assert isinstance(orbital, OrbitalElement)
    assert orbital.norad_id == 456
    assert orbital.epoch == epoch.to_datetime(timezone=timezone.utc)


@pytest.mark.usefixtures("stable_idxmin")
def test_select_extracts_attributes_and_ignores_unreadable_attribute():
    epoch = Time("2024-01-09")

    sat = FakeSatID(
        norad_id=789,
        elements=[
            AttributeElement(epoch),
        ],
    )

    selected = TLESelector.select(
        [sat],
        Time("2024-01-10").jd,
    )

    orbital = selected[0]["element"]

    from radioskysims_sources.domain.satellite.models import OrbitalElement

    assert isinstance(orbital, OrbitalElement)
    assert orbital.norad_id == 789
    assert orbital.tle_line1 == "LINE-1"
    assert orbital.epoch == epoch.to_datetime(timezone=timezone.utc)


@pytest.mark.usefixtures("stable_idxmin")
def test_select_preserves_existing_orbital_element():
    from radioskysims_sources.domain.satellite.models import OrbitalElement

    epoch = Time("2024-01-09").to_datetime(
        timezone=timezone.utc,
    )

    element = OrbitalElement(
        epoch=epoch,
        norad_id=321,
        tle_line1="LINE-1",
        tle_line2="LINE-2",
    )

    sat = FakeSatID(
        norad_id=321,
        elements=[element],
    )

    selected = TLESelector.select(
        [sat],
        Time("2024-01-10").jd,
    )

    assert selected[0]["element"] is element
    assert selected[0]["epoch"] == epoch
