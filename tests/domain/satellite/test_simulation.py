from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from radioskysims_sources.domain.satellite.simulation import Simulation


def _time_range() -> SimpleNamespace:
    return SimpleNamespace(start="2026-08-30T10:00:00", end="2026-08-30T12:00:00")


def _signals() -> xr.Dataset:
    return xr.Dataset({"PSD": (("satellite",), np.array([1.0]))})


class GeometryProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], object]] = []

    def compute(self, satellites, time_range):
        self.calls.append((satellites, time_range))
        return {"geometry": "computed"}


class SignalFactory:
    def __init__(self, result: xr.Dataset) -> None:
        self.result = result
        self.calls: list[tuple[list[object], object, object]] = []

    def simulate(self, satellites, geometry, time_range):
        self.calls.append((satellites, geometry, time_range))
        return self.result


def test_run_computes_geometry_then_returns_simulated_signals(caplog):
    caplog.set_level(logging.INFO)
    satellites = [object(), object()]
    time_range = _time_range()
    geometry_provider = GeometryProvider()
    signal_factory = SignalFactory(_signals())
    simulation = Simulation(geometry_provider, signal_factory)

    result = simulation.run(satellites, time_range)

    assert result is signal_factory.result
    assert geometry_provider.calls == [(satellites, time_range)]
    assert signal_factory.calls == [(satellites, {"geometry": "computed"}, time_range)]
    assert "satellite_simulation_start" in caplog.text
    assert "satellite_geometry_computed" in caplog.text
    assert "satellite_simulation_success" in caplog.text


def test_run_reraises_geometry_error_and_logs_failure(caplog):
    error = RuntimeError("geometry unavailable")

    class FailingGeometryProvider:
        def compute(self, satellites, time_range):  # noqa: ARG002
            raise error

    signal_factory = SignalFactory(_signals())
    simulation = Simulation(FailingGeometryProvider(), signal_factory)

    with pytest.raises(RuntimeError, match="geometry unavailable"):
        simulation.run([object()], _time_range())

    assert signal_factory.calls == []
    assert "satellite_simulation_failed" in caplog.text
