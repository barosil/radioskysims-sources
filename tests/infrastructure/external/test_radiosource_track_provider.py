from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from radioskysims_sources.domain.radiosources.models import RadioSource, SpectralModel
from radioskysims_sources.infrastructure.external.radiosource_track_provider import (
    SkyfieldRadioSourceTrackProvider,
)


def _source(source_id: str, ra_deg: float, dec_deg: float) -> RadioSource:
    return RadioSource(
        id=source_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        spectral_model=SpectralModel(
            nu0_mhz=1000.0,
            s0_jy=1.0,
            alpha=0.0,
        ),
    )


class _FakeTimes:
    tt = np.array([2460000.0, 2460000.5])


class _FakeObserver:
    def to_skyfield(self):
        return "observer-location", None


class _FakeAngle:
    def __init__(self, radians):
        self.radians = np.asarray(radians)


class _FakeDistance:
    m = np.array([100.0, 200.0])


class _FakeApparent:
    def altaz(self):
        return (
            _FakeAngle([0.5, 0.6]),
            _FakeAngle([1.0, 1.1]),
            _FakeDistance(),
        )


class _FakeObserved:
    def observe(self, _star):
        return self

    def apparent(self):
        return _FakeApparent()


class _FakeEarth:
    def __add__(self, observer_location):
        assert observer_location == "observer-location"
        return self

    def at(self, times):
        assert times is fake_times
        return _FakeObserved()


fake_times = _FakeTimes()


def _time_range():
    return SimpleNamespace(
        start_jd=2460000.0,
        end_jd=2460000.5,
        to_skyfield=lambda: fake_times,
    )


def test_compute_track_returns_empty_track_for_no_sources(monkeypatch):
    monkeypatch.setattr(
        "radioskysims_sources.infrastructure.external.radiosource_track_provider.load_ephemeris",
        lambda: {"earth": object()},
    )
    provider = SkyfieldRadioSourceTrackProvider()

    result = provider.compute_track([], _FakeObserver(), _time_range())

    assert result.source_ids == []
    assert result.dataset.sizes == {"source": 0, "time": 0}
    assert result.dataset.data_vars == {}
    assert result.time_range[0] < result.time_range[1]


def test_compute_track_builds_altaz_dataset_for_each_source(monkeypatch):
    stars = []

    class FakeStar:
        def __init__(self, **kwargs):
            stars.append(kwargs)

    monkeypatch.setattr(
        "radioskysims_sources.infrastructure.external.radiosource_track_provider.load_ephemeris",
        lambda: {"earth": _FakeEarth()},
    )
    monkeypatch.setattr(
        "radioskysims_sources.infrastructure.external.radiosource_track_provider.Star",
        FakeStar,
    )
    provider = SkyfieldRadioSourceTrackProvider()
    sources = [
        _source("SRC-1", 30.0, -10.0),
        _source("SRC-2", 45.0, 20.0),
    ]

    result = provider.compute_track(sources, _FakeObserver(), _time_range())

    assert result.source_ids == ["SRC-1", "SRC-2"]
    assert result.dataset.attrs == {"provider": "skyfield"}
    assert result.dataset.sizes["source"] == 2
    assert result.dataset.sizes["time"] == 2
    assert set(result.dataset.data_vars) == {
        "theta_rad",
        "az_rad",
        "el_rad",
        "range_m",
    }
    np.testing.assert_allclose(result.dataset["theta_rad"].values[0], [np.pi / 2 - 0.5, np.pi / 2 - 0.6])
    np.testing.assert_allclose(result.dataset["az_rad"].values[1], [1.0, 1.1])
    np.testing.assert_allclose(result.dataset["el_rad"].values[0], [0.5, 0.6])
    np.testing.assert_allclose(result.dataset["range_m"].values[1], [100.0, 200.0])
    assert stars == [
        {"ra_hours": 2.0, "dec_degrees": -10.0},
        {"ra_hours": 3.0, "dec_degrees": 20.0},
    ]
    assert result.time_range == (
        str(result.dataset.time.values[0]),
        str(result.dataset.time.values[-1]),
    )
