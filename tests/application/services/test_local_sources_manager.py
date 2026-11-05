"""\
Módulo: tests/application/services/test_local_sources_manager.py

Descrição:
    Testes do facade ``LocalSourcesManager`` / ``Sun`` (fluxo + track).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from radioskysims_sources.application.services import LocalSourcesManager, Sun
from radioskysims_sources.application.services.flux_view import FluxDensity
from radioskysims_sources.domain.localsources.models import SolarFlux


@pytest.fixture(scope="module")
def sun() -> Sun:
    """Sol simulado (track via skyfield) — calculado uma única vez."""
    return Sun.simulated()


def test_sun_simulated_flux_default(sun: Sun):
    assert isinstance(sun.flux, SolarFlux)
    fd = sun.flux_density()
    assert isinstance(fd, FluxDensity)
    assert "Sun" in fd.flux_by_source
    # Sol calmo ~150 SFU = 1.5e6 Jy
    assert np.allclose(fd.total, 1.5e6, rtol=1e-6)


def _burst_flux() -> SolarFlux:
    """SolarFlux sintético: banda BINGO + burst enorme fora dela (15400 MHz)."""
    times = np.array(["2026-08-01T00:00:00", "2026-08-01T01:00:00"], dtype="datetime64[ns]")
    freqs = np.array([245.0, 610.0, 1415.0, 2695.0, 8800.0, 15400.0])
    burst = np.array([1e5, 5e5, 8e5, 1e6, 2e6, 46.5e6])  # burst de ~1e7 em 15400
    flux = np.array([burst, burst])
    ds = xr.Dataset(
        {"flux_jy": (("time", "frequency_mhz"), flux)},
        coords={"time": times, "frequency_mhz": freqs},
    )
    return SolarFlux(
        start_utc=datetime(2026, 8, 1),
        end_utc=datetime(2026, 8, 1, 1, 0),
        dataset=ds,
        source_url="test:burst",
    )


def test_sun_flux_density_defaults_to_bingo_band():
    """Curva padrão fica na banda do BINGO, excluindo bursts fora da banda."""
    sun = Sun.simulated(flux=_burst_flux())
    fd = sun.flux_density()
    # banda do BINGO (1000–1500 MHz), não o range completo do SRD (245–15400)
    assert fd.freqs_mhz.min() >= 1000.0
    assert fd.freqs_mhz.max() <= 1500.0
    # sem o burst de ~1e7 Jy em 15400 MHz — linha visível na escala
    assert np.nanmax(fd.total) < 1e7
    assert np.nanmax(fd.total) > 1e5


def test_sun_simulated_track_vars(sun: Sun):
    assert sun.track is not None
    for var in ["az_rad", "el_rad", "theta_rad", "range_m", "ra_rad", "dec_rad"]:
        assert var in sun.track.dataset.data_vars
    assert sun.track.dataset.sizes["time"] > 1


def test_sun_position_at(sun: Sun):
    times = sun.track.dataset["time"].values
    t = times[len(times) // 2]
    metrics = sun.position_at(t)
    assert set(metrics) == {"theta_rad", "az_rad", "el_rad", "range_m"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_local_sources_manager_sun():
    mgr = LocalSourcesManager()
    sun = mgr.sun()
    assert isinstance(sun, Sun)
    assert sun.track is not None
    assert sun.flux is not None
