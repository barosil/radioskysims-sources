"""\
Módulo: tests/domain/satellite/test_models.py

Descrição:
    Testes de timezone-aware e da interface UX de ``SatelliteAssembled``
    (``__repr__``, ``summary``, ``select_Constellation``, ``select_Satellite``).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import ModuleType
import sys

import numpy as np
import pytest
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


def test_as_utc_returns_none_unchanged():
    assert as_utc(None) is None


def test_as_utc_converts_non_utc_offset():
    dt = as_utc(
        datetime(2026, 8, 30, 15, 0, tzinfo=timezone(timedelta(hours=3)))
    )
    assert dt == datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


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


def test_satellite_epoch_interval_rejects_large_difference():
    epoch = datetime(2026, 8, 30, 12, 0)
    with pytest.raises(ValueError, match="excede o limite de 5 dias"):
        Satellite(
            norad_id=1,
            epoch=epoch,
            element=_elem(1, datetime(2026, 9, 6, 12, 0)),
        )


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


def test_satellite_sv_none_without_constellation():
    epoch = datetime(2026, 8, 30, 12, 0)
    sat = Satellite(
        norad_id=1,
        name="X",
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
    assert str(a) == r


def test_assembled_repr_truncates_long_name_list():
    assembled = _assembled()
    assembled.name = [f"SAT-{index}" for index in range(7)]
    representation = repr(assembled)
    assert "SAT-0, SAT-1, SAT-2, SAT-3, SAT-4, … (+2)" in representation


def test_assembled_summary_groups_constellation():
    a = _assembled()
    s = a.summary()
    assert "GPS=2" in s and "GALILEO=1" in s


def test_assembled_properties_and_summary_without_constellation():
    tracks = xr.Dataset(
        {"theta": (("satellite",), [1.0, 2.0]), "range": (("satellite",), [3.0, 4.0])},
        coords={"satellite": [10, 20]},
    )
    signals = xr.Dataset(
        {"PSD": (("satellite", "freq"), np.ones((2, 1)))},
        coords={"satellite": [10, 20], "freq": [100.0]},
    )
    assembled = SatelliteAssembled(name="single", tracks=tracks, signals=signals)

    assert assembled.norad_ids == [10, 20]
    assert assembled.constellations == []
    assert assembled.summary() == repr(assembled)
    assert assembled.angle().values.tolist() == [1.0, 2.0]
    assert assembled.range().values.tolist() == [3.0, 4.0]
    assert assembled.psd().shape == (2, 1)
    assert _assembled().constellations == ["GALILEO", "GPS"]


def test_select_constellation_filters_tracks_and_signals():
    a = _assembled()
    gps = a.select_Constellation("GPS")
    assert gps.norad_ids == [1, 2]
    assert gps.signals.sizes["satellite"] == 5  # 3 sinais (sat 1) + 2 (sat 2)


def test_select_constellation_accepts_case_insensitive_list():
    selected = _assembled().select_Constellation(["galileo"])
    assert selected.norad_ids == [3]


def test_select_constellation_requires_coordinate():
    tracks = xr.Dataset(coords={"satellite": [1]})
    signals = xr.Dataset(coords={"satellite": [1]})
    assembled = SatelliteAssembled(name="A", tracks=tracks, signals=signals)

    with pytest.raises(ValueError, match="não tem a coordenada 'constellation'"):
        assembled.select_Constellation("GPS")


def test_select_satellite_by_norad():
    a = _assembled()
    one = a.select_Satellite(noradID=1)
    assert one.norad_ids == [1]


def test_select_satellite_combines_norad_and_name_lists():
    selected = _assembled().select_Satellite(noradID=[1, 2], name=["B", "C"])
    assert selected.norad_ids == [2]


def test_select_satellite_by_name():
    a = _assembled()
    one = a.select_Satellite(name="C")
    assert one.norad_ids == [3]


def test_select_constellation_unknown_returns_empty():
    a = _assembled()
    empty = a.select_Constellation("BEIDOU")
    assert empty.norad_ids == []


def test_subset_without_names_coordinate_returns_empty_names():
    tracks = xr.Dataset(
        {"theta": (("satellite",), [1.0])},
        coords={"satellite": [1]},
    )
    signals = xr.Dataset(
        {"PSD": (("satellite", "freq"), [[1.0]])},
        coords={"satellite": [1], "freq": [100.0]},
    )
    assembled = SatelliteAssembled(name="A", tracks=tracks, signals=signals)

    selected = assembled._subset(np.array([True]))
    assert selected.name == []


def test_plot_tracks_forwards_dataset_and_options(monkeypatch):
    calls = []
    satellite_module = ModuleType("radioskysims_viz.satellite")

    def fake_tracks_mollweide(*args, **kwargs):
        calls.append((args, kwargs))
        return "figure"

    satellite_module.tracks_mollweide = fake_tracks_mollweide
    viz_module = ModuleType("radioskysims_viz")
    viz_module.satellite = satellite_module
    monkeypatch.setitem(sys.modules, "radioskysims_viz", viz_module)
    monkeypatch.setitem(sys.modules, "radioskysims_viz.satellite", satellite_module)

    assembled = _assembled()
    result = assembled.plot_tracks(smooth=False, kde=True, tracks=False, marker="x")

    assert result == "figure"
    assert calls == [
        (
            (assembled.tracks,),
            {
                "smooth": False,
                "kde": True,
                "tracks": False,
                "figsize": (9.0, 5.0),
                "marker": "x",
            },
        )
    ]


@pytest.mark.parametrize("satellites", ["GPS", [1], None])
def test_plot_signals_selects_target_and_forwards_names(monkeypatch, satellites):
    calls = []
    satellite_module = ModuleType("radioskysims_viz.satellite")

    def fake_signals_plot(signals, **kwargs):
        calls.append((signals, kwargs))
        return "plot"

    satellite_module.signals_plot = fake_signals_plot
    viz_module = ModuleType("radioskysims_viz")
    viz_module.satellite = satellite_module
    monkeypatch.setitem(sys.modules, "radioskysims_viz", viz_module)
    monkeypatch.setitem(sys.modules, "radioskysims_viz.satellite", satellite_module)

    result = _assembled().plot_signals(satellites=satellites, style="line")

    assert result == "plot"
    signals, kwargs = calls[0]
    expected_size = 5 if satellites in ("GPS", None) else 3
    assert signals.sizes["satellite"] == expected_size
    assert kwargs["norad_names"]
    assert kwargs["style"] == "line"
