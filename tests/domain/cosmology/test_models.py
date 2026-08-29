import numpy as np
import pytest

from radioskysims_sources.domain.cosmology import (
    CosmologyParameters,
    RedshiftRange,
)


def test_cosmology_parameters_compute_omega_m() -> None:
    cosmology = CosmologyParameters(
        omega_c=0.25,
        omega_b=0.05,
        h=0.67,
        sigma8=0.81,
        n_s=0.965,
    )

    assert cosmology.omega_m == pytest.approx(0.30)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("omega_c", 0.0),
        ("omega_b", 0.0),
        ("h", 0.0),
        ("sigma8", 0.0),
        ("n_s", 0.0),
    ],
)
def test_cosmology_parameters_require_positive_values(
    field: str,
    value: float,
) -> None:
    kwargs = {
        "omega_c": 0.25,
        "omega_b": 0.05,
        "h": 0.67,
        "sigma8": 0.81,
        "n_s": 0.965,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        CosmologyParameters(**kwargs)


def test_redshift_range_generates_samples() -> None:
    redshift_range = RedshiftRange(
        min_z=0.1,
        max_z=0.3,
        resolution_z=0.1,
    )

    np.testing.assert_allclose(
        redshift_range.value,
        np.array([0.1, 0.2, 0.3]),
    )


def test_redshift_range_single_value() -> None:
    redshift_range = RedshiftRange(
        min_z=0.2,
        max_z=0.2,
        resolution_z=0.0,
    )

    np.testing.assert_allclose(
        redshift_range.value,
        np.array([0.2]),
    )


def test_redshift_range_rejects_negative_redshift() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        RedshiftRange(
            min_z=-0.1,
            max_z=0.2,
            resolution_z=0.1,
        )


def test_redshift_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        RedshiftRange(
            min_z=0.4,
            max_z=0.2,
            resolution_z=0.1,
        )


def test_matter_power_spectrum_creation() -> None:
    from radioskysims_sources.domain.cosmology import MatterPowerSpectrum

    spectrum = MatterPowerSpectrum(
        k_mpc=np.array([0.01, 0.1, 1.0]),
        power_mpc3=np.array([100.0, 50.0, 5.0]),
        redshift=0.3,
    )

    np.testing.assert_allclose(
        spectrum.k_mpc,
        np.array([0.01, 0.1, 1.0]),
    )

    np.testing.assert_allclose(
        spectrum.power_mpc3,
        np.array([100.0, 50.0, 5.0]),
    )

    assert spectrum.redshift == 0.3


def test_matter_power_spectrum_requires_matching_shapes() -> None:
    from radioskysims_sources.domain.cosmology import MatterPowerSpectrum

    with pytest.raises(ValueError, match="same shape"):
        MatterPowerSpectrum(
            k_mpc=np.array([0.01, 0.1]),
            power_mpc3=np.array([100.0]),
            redshift=0.3,
        )


def test_matter_power_spectrum_rejects_negative_power() -> None:
    from radioskysims_sources.domain.cosmology import MatterPowerSpectrum

    with pytest.raises(ValueError, match="cannot be negative"):
        MatterPowerSpectrum(
            k_mpc=np.array([0.01, 0.1]),
            power_mpc3=np.array([100.0, -1.0]),
            redshift=0.3,
        )


def test_redshift_window_creation() -> None:
    from radioskysims_sources.domain.cosmology import RedshiftWindow

    window = RedshiftWindow(
        redshift=np.array([0.2, 0.3, 0.4]),
        weight=np.array([0.2, 1.0, 0.2]),
    )

    assert window.min_z == pytest.approx(0.2)
    assert window.max_z == pytest.approx(0.4)


def test_redshift_window_rejects_empty_selection() -> None:
    from radioskysims_sources.domain.cosmology import RedshiftWindow

    with pytest.raises(ValueError, match="positive weight"):
        RedshiftWindow(
            redshift=np.array([0.2, 0.3]),
            weight=np.array([0.0, 0.0]),
        )


def test_angular_power_spectrum_creation() -> None:
    from radioskysims_sources.domain.cosmology import AngularPowerSpectrum

    spectrum = AngularPowerSpectrum(
        ell=np.array([2, 3, 4, 5]),
        cls=np.array([1.0e-5, 8.0e-6, 6.0e-6, 5.0e-6]),
        redshift_min=0.2,
        redshift_max=0.3,
    )

    np.testing.assert_array_equal(
        spectrum.ell,
        np.array([2, 3, 4, 5]),
    )

    assert spectrum.redshift_min == pytest.approx(0.2)
    assert spectrum.redshift_max == pytest.approx(0.3)


def test_angular_power_spectrum_rejects_negative_cl() -> None:
    from radioskysims_sources.domain.cosmology import AngularPowerSpectrum

    with pytest.raises(ValueError, match="cannot be negative"):
        AngularPowerSpectrum(
            ell=np.array([2, 3]),
            cls=np.array([1.0e-5, -1.0e-5]),
            redshift_min=0.2,
            redshift_max=0.3,
        )
