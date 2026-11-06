"""Testes da mediana diária (NaN-aware) do fluxo solar (item 3).

Módulo: tests/domain/test_solar_flux_daily_mean.py

Verifica que ``SolarFlux.daily_mean_grid`` usa a mediana por dia UTC ignorando
``NaN`` (em vez de interpolar hora a hora) e cai para a mediana geral em dias
sem cobertura.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
from radioskysims_sources.domain.localsources import SolarFlux


def _solar_flux_with_nan() -> SolarFlux:
    times = np.array(
        [
            "2026-08-01T00:00:00",
            "2026-08-01T12:00:00",
            "2026-08-02T00:00:00",
            "2026-08-02T12:00:00",
        ],
        dtype="datetime64[ns]",
    )
    freqs = np.array([1000.0, 2000.0])
    # Dia 1: amostras finitas. Dia 2: 2ª amostra NaN (deve ser ignorada).
    values = np.array(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
            [np.nan, np.nan],
        ]
    )
    ds = xr.Dataset(
        {"flux_jy": (("time", "frequency_mhz"), values)},
        coords={"time": times, "frequency_mhz": freqs},
    )
    return SolarFlux(
        start_utc=datetime(2026, 8, 1, tzinfo=UTC),
        end_utc=datetime(2026, 8, 3, tzinfo=UTC),
        dataset=ds,
    )


def test_daily_mean_ignores_nan_and_groups_by_utc_day():
    sf = _solar_flux_with_nan()
    times = np.array(
        [
            "2026-08-01T06:00:00",
            "2026-08-02T06:00:00",
        ],
        dtype="datetime64[ns]",
    )
    grid = sf.daily_mean_grid(times=times, freqs_mhz=np.array([1000.0, 2000.0]))

    # Dia 1: mediana([10,20]) = 15 ; mediana([100,200]) = 150
    assert grid[0] == pytest.approx([15.0, 150.0])
    # Dia 2: NaN ignorado → mediana([30]) = 30 ; 300
    assert grid[1] == pytest.approx([30.0, 300.0])


def test_daily_mean_falls_back_to_overall_for_uncovered_day():
    sf = _solar_flux_with_nan()
    times = np.array(["2026-08-05T06:00:00"], dtype="datetime64[ns]")

    grid = sf.daily_mean_grid(times=times, freqs_mhz=np.array([1000.0, 2000.0]))

    # Sem cobertura → mediana geral (ignorando NaN): mediana([10,20,30])=20, mediana([100,200,300])=200
    assert grid[0] == pytest.approx([20.0, 200.0])


def test_daily_mean_interpolates_frequency_only():
    sf = _solar_flux_with_nan()
    times = np.array(["2026-08-01T06:00:00"], dtype="datetime64[ns]")

    grid = sf.daily_mean_grid(times=times, freqs_mhz=np.array([1500.0]))

    # Interpolação linear em frequência: 15 + (150-15)/2 = 82.5
    assert grid[0, 0] == pytest.approx(82.5)
