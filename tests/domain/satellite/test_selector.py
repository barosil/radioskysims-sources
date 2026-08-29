from __future__ import annotations

from astropy.time import Time

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
