"""\
Módulo: tests/domain/satellite/test_models.py

Descrição:
    Testes de timezone-aware e da interface UX de ``SatelliteAssembled``
    (``__repr__``, ``summary``, ``select_Constellation``, ``select_Satellite``).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr
from radioskysims_sources.domain.satellite.models import (
    OrbitalElement,
    Satellite,
    SatelliteAssembled,
    as_utc,
)


def _elem(norad: int, epoch: datetime) -> OrbitalElement:
    return OrbitalElement(epoch=epoch, norad_id=norad, tle_line1="1 x", tle_line2="2 x")


# ----------------------------------------------------------------------
# Timezone-aware
# ----------------------------------------------------------------------
def test_as_utc_naive():
    dt = as_utc(datetime(2026, 8, 30, 12, 0))
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0


def test_as_utc_aware_keeps_instant():
    dt = as_utc(datetime(2026, 8, 30, 9, 0, tzinfo=timezone.utc))
    assert dt.isoformat() == "2026-08-30T09:00:00+00:00"


def test_orbital_element_epoch_is_aware():
    elem = _elem(1, datetime(2026, 8, 30, 12, 0))
    assert elem.epoch.tzinfo is not None


def test_satellite_epoch_is_aware():
    epoch = datetime(2026, 8, 30, 12, 0)
    sat = Satellite(norad_id=1, name="S1", epoch=epoch, element=_elem(1, epoch))
    assert sat.epoch.tzinfo is not None
    assert sat.element.epoch.tzinfo is not None


def test_satellite_epoch_interval_valid():
    epoch = datetime(2026, 8, 30, 12, 0)
    Satellite(norad_id=1, epoch=epoch, element=_elem(1, epoch))


# ----------------------------------------------------------------------
# PRN — mapeamento NORAD ↔ Nome ↔ PRN (ponte TLE ↔ RINEX)
# ----------------------------------------------------------------------
def test_satellite_prn_default_none():
    epoch = datetime(2026, 8, 30, 12, 0)
    sat = Satellite(norad_id=1, name="S1", epoch=epoch, element=_elem(1, epoch))
    assert sat.prn is None
    assert sat.sv is None


def test_satellite_prn_roundtrip_and_sv():
    epoch = datetime(2026, 8, 30, 12, 0)
    sat = Satellite(
        norad_id=26407,
        name="GPS BIIR-5 (PRN 22)",
        constellation="GPS",
        prn=22,
        epoch=epoch,
        element=_elem(26407, epoch),
    )
    assert sat.prn == 22
    assert sat.sv == "G22"


def test_satellite_sv_none_for_unknown_constellation():
    epoch = datetime(2026, 8, 30, 12, 0)
    sat = Satellite(
        norad_id=1,
        name="X",
        constellation="STARLINK",
        prn=1,
        epoch=epoch,
        element=_elem(1, epoch),
    )
    assert sat.sv is None


# ----------------------------------------------------------------------
# SatelliteAssembled — seleção e representação
# ----------------------------------------------------------------------
def _assembled() -> SatelliteAssembled:
    tracks = xr.Dataset(
        {"positions": (("satellite", "time", "xyz"), np.zeros((3, 4, 3)))},
        coords={
            "satellite": [1, 2, 3],
            "time": np.arange(4.0),
            "xyz": ["x", "y", "z"],
            "names": ("satellite", ["A", "B", "C"]),
            "constellation": ("satellite", ["GPS", "GPS", "GALILEO"]),
        },
    )
    signals = xr.Dataset(
        {"PSD": (("satellite", "freq"), np.zeros((5, 10)))},
        coords={
            "satellite": [1, 1, 1, 2, 2],
            "freq": np.linspace(1000, 1250, 10),
            "fc": ("satellite", [1575.42, 1227.6, 1176.45, 1575.42, 1227.6]),
            "modulation": ("satellite", ["BPSK(1)"] * 5),
        },
    )
    return SatelliteAssembled(name=["A", "B", "C"], tracks=tracks, signals=signals)


def test_assembled_repr_mentions_sizes():
    a = _assembled()
    r = repr(a)
    assert "3 satélites" in r
    assert "tracks" in r and "signals" in r


def test_assembled_summary_groups_constellation():
    a = _assembled()
    s = a.summary()
    assert "GPS=2" in s and "GALILEO=1" in s


def test_select_constellation_filters_tracks_and_signals():
    a = _assembled()
    gps = a.select_Constellation("GPS")
    assert gps.norad_ids == [1, 2]
    assert gps.signals.sizes["satellite"] == 5  # 3 sinais (sat 1) + 2 (sat 2)


def test_select_satellite_by_norad():
    a = _assembled()
    one = a.select_Satellite(noradID=1)
    assert one.norad_ids == [1]


def test_select_satellite_by_name():
    a = _assembled()
    one = a.select_Satellite(name="C")
    assert one.norad_ids == [3]


def test_select_constellation_unknown_returns_empty():
    a = _assembled()
    empty = a.select_Constellation("BEIDOU")
    assert empty.norad_ids == []
