from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from radioskysims_sources.domain.signal_source import (
    LocalSourceSignalSource,
    RadioSourceSignalSource,
    SatelliteSignalSource,
    _flux_density_windowed_mean,
    _sun_transit_window_s,
    compute_theta_boresight,
)


def _normalized_tracks() -> xr.Dataset:
    return xr.Dataset(
        {
            "theta": ("time", [0.1, 0.2]),
            "az": ("time", [0.3, 0.4]),
            "el": ("time", [0.5, 0.6]),
            "range": ("time", [1.0e6, 1.1e6]),
        }
    )


def _suffixed_tracks() -> xr.Dataset:
    return xr.Dataset(
        {
            "theta_rad": ("time", [0.1, 0.2]),
            "az_rad": ("time", [0.3, 0.4]),
            "el_rad": ("time", [0.5, 0.6]),
            "range_m": ("time", [1.0e6, 1.1e6]),
        }
    )


def test_satellite_signal_source_exposes_adapter_contract() -> None:
    expected_psd = xr.DataArray(
        [1.0, 2.0],
        dims="frequency",
        coords={"frequency": [1.0e9, 1.1e9]},
        attrs={"units": "W/Hz"},
    )
    satellite = SimpleNamespace(
        name="GPS-1",
        tracks=_normalized_tracks(),
        psd=lambda: expected_psd,
    )

    source = SatelliteSignalSource(satellite)

    assert source.name == "GPS-1"
    assert source.source_type == "satellite"
    assert source.tracks is satellite.tracks
    xr.testing.assert_identical(source.psd(), expected_psd)


def test_satellite_signal_source_uses_default_name() -> None:
    source = SatelliteSignalSource(SimpleNamespace(tracks=_normalized_tracks()))

    assert source.name == "unnamed_satellite"


def test_satellite_signal_source_requires_tracks() -> None:
    source = SatelliteSignalSource(SimpleNamespace(name="GPS-1"))

    with pytest.raises(AttributeError, match="não possui tracks"):
        _ = source.tracks


def test_satellite_signal_source_requires_psd_method() -> None:
    source = SatelliteSignalSource(SimpleNamespace(name="GPS-1", tracks=_normalized_tracks()))

    with pytest.raises(AttributeError, match=r"não possui método psd\(\)"):
        source.psd()


def test_local_source_tracks_normalize_variable_names() -> None:
    source = LocalSourceSignalSource(
        name="Sun",
        track=SimpleNamespace(dataset=_suffixed_tracks()),
    )

    result = source.tracks

    assert source.name == "Sun"
    assert source.source_type == "localsource"
    assert {"theta", "az", "el", "range"} <= set(result.data_vars)


def test_local_source_tracks_require_dataset() -> None:
    source = LocalSourceSignalSource(name="Sun", track=SimpleNamespace())

    with pytest.raises(AttributeError, match="não possui dataset"):
        _ = source.tracks


def test_local_source_tracks_require_all_variables() -> None:
    incomplete = xr.Dataset({"theta": ("time", [0.1])})
    source = LocalSourceSignalSource(
        name="Sun",
        track=SimpleNamespace(dataset=incomplete),
    )

    with pytest.raises(ValueError, match="sem variáveis"):
        _ = source.tracks


def test_local_source_psd_without_flux_returns_zero_fallback() -> None:
    source = LocalSourceSignalSource(
        name="Sun",
        track=SimpleNamespace(dataset=_normalized_tracks()),
    )

    result = source.psd()

    np.testing.assert_allclose(result.values, [0.0])
    np.testing.assert_allclose(result.frequency.values, [1.0])
    assert result.attrs["units"] == "W/m²/Hz"


def test_local_source_psd_converts_jansky_to_flux_density() -> None:
    source = LocalSourceSignalSource(
        name="Sun",
        track=SimpleNamespace(dataset=_normalized_tracks()),
        flux_jy_grid=np.array([10.0, 20.0]),
        frequency_hz=np.array([1.0e9, 1.1e9]),
    )

    result = source.psd()

    np.testing.assert_allclose(result.values, [1.0e-25, 2.0e-25])
    np.testing.assert_allclose(result.frequency.values, [1.0e9, 1.1e9])
    assert result.attrs["units"] == "W/m²/Hz"


def test_radio_source_tracks_normalize_variable_names() -> None:
    source = RadioSourceSignalSource(
        source_id="NVSS-1",
        track=SimpleNamespace(dataset=_suffixed_tracks()),
    )

    result = source.tracks

    assert source.name == "NVSS-1"
    assert source.source_type == "radiosource"
    assert {"theta", "az", "el", "range"} <= set(result.data_vars)


def test_radio_source_tracks_require_dataset() -> None:
    source = RadioSourceSignalSource(
        source_id="NVSS-1",
        track=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match="não possui dataset"):
        _ = source.tracks


def test_radio_source_tracks_require_all_variables() -> None:
    source = RadioSourceSignalSource(
        source_id="NVSS-1",
        track=SimpleNamespace(dataset=xr.Dataset({"theta": ("time", [0.1])})),
    )

    with pytest.raises(ValueError, match="sem variáveis"):
        _ = source.tracks


def test_radio_source_psd_converts_jansky_to_flux_density() -> None:
    source = RadioSourceSignalSource(
        source_id="NVSS-1",
        track=SimpleNamespace(dataset=_normalized_tracks()),
        flux_jy_grid=np.array([2.0, 3.0]),
        frequency_hz=np.array([900.0e6, 1.0e9]),
    )

    result = source.psd()

    np.testing.assert_allclose(result.values, [2.0e-26, 3.0e-26])
    assert result.attrs["units"] == "W/m²/Hz"


def test_radio_source_psd_without_flux_returns_zero_fallback() -> None:
    source = RadioSourceSignalSource(
        source_id="NVSS-1",
        track=SimpleNamespace(dataset=_normalized_tracks()),
    )

    result = source.psd()

    np.testing.assert_allclose(result.values, [0.0])
    assert result.attrs["units"] == "W/m²/Hz"


def test_sun_transit_window_uses_fwhm_and_declination() -> None:
    assert _sun_transit_window_s(15.0, 0.0) == pytest.approx(3600.0)
    assert _sun_transit_window_s(15.0, 60.0) == pytest.approx(7200.0)


@pytest.mark.parametrize("fwhm", [0.0, -1.0, float("nan")])
def test_sun_transit_window_rejects_invalid_fwhm(fwhm: float) -> None:
    assert np.isnan(_sun_transit_window_s(fwhm))


def test_flux_density_windowed_mean_preserves_one_dimensional_flux() -> None:
    flux = np.array([1.0, 2.0, 3.0])

    result = _flux_density_windowed_mean(flux)

    np.testing.assert_allclose(result, flux)


def test_flux_density_windowed_mean_uses_total_mean_without_window() -> None:
    flux = np.array(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [5.0, 50.0],
        ]
    )

    result = _flux_density_windowed_mean(flux)

    np.testing.assert_allclose(result, [3.0, 30.0])


def test_compute_theta_boresight_fixed_radec_zero_separation() -> None:
    tracks = xr.Dataset(
        {
            "ra": ("time", np.deg2rad([10.0, 10.0])),
            "dec": ("time", np.deg2rad([-20.0, -20.0])),
        }
    )

    result = compute_theta_boresight(
        tracks,
        pointing_ra_deg=10.0,
        pointing_dec_deg=-20.0,
    )

    np.testing.assert_allclose(result.values, 0.0, atol=1.0e-7)
    assert result.attrs["units"] == "rad"


def test_compute_theta_boresight_altaz_zero_separation() -> None:
    tracks = xr.Dataset(
        {
            "el": ("time", np.deg2rad([45.0, 45.0])),
            "az": ("time", np.deg2rad([30.0, 30.0])),
        }
    )

    result = compute_theta_boresight(
        tracks,
        pointing_alt_deg=45.0,
        pointing_az_deg=30.0,
    )

    np.testing.assert_allclose(result.values, 0.0, atol=1.0e-7)


def test_compute_theta_boresight_requires_pointing() -> None:
    tracks = xr.Dataset(
        {
            "ra": ("time", [0.0]),
            "dec": ("time", [0.0]),
        }
    )

    with pytest.raises(ValueError, match="requer RA/Dec ou alt-az"):
        compute_theta_boresight(tracks)
