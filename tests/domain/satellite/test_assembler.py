from __future__ import annotations

from radioskysims_sources.domain.satellite.assembler import SatelliteAssembler


class DummySatellite:
    def __init__(self, norad_id, **_kwargs):
        self.norad_id = norad_id


def test_assemble_valid(monkeypatch):
    # monkeypatch do modelo real
    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.assembler.Satellite",
        DummySatellite,
    )

    params = [{"norad_id": 1}, {"norad_id": 2}]

    sats = SatelliteAssembler.assemble(params)

    assert len(sats) == 2
    assert {s.norad_id for s in sats} == {1, 2}


def test_assemble_invalid(monkeypatch):
    def raising_satellite(**_):
        msg = "invalid"
        raise ValueError(msg)

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.assembler.Satellite",
        raising_satellite,
    )

    sats = SatelliteAssembler.assemble([{"norad_id": 99}])

    assert sats == []
