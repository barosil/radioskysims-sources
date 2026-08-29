import numpy as np
import pytest

from radioskysims_sources.domain.cosmology import (
    CosmologyParameters,
    dimensionless_hubble_parameter,
    hi_bias,
    hi_temperature_fluctuation_uk,
    mean_hi_brightness_temperature_uk,
)


@pytest.fixture
def cosmology() -> CosmologyParameters:
    return CosmologyParameters(
        omega_c=0.25,
        omega_b=0.05,
        h=0.67,
        sigma8=0.81,
        n_s=0.965,
    )


def test_dimensionless_hubble_parameter_at_z_zero(
    cosmology: CosmologyParameters,
) -> None:
    assert dimensionless_hubble_parameter(
        0.0,
        cosmology,
    ) == pytest.approx(1.0)


def test_dimensionless_hubble_parameter_increases_with_redshift(
    cosmology: CosmologyParameters,
) -> None:
    e0 = dimensionless_hubble_parameter(0.0, cosmology)
    e1 = dimensionless_hubble_parameter(1.0, cosmology)

    assert e1 > e0


def test_dimensionless_hubble_parameter_rejects_negative_redshift(
    cosmology: CosmologyParameters,
) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        dimensionless_hubble_parameter(
            -0.1,
            cosmology,
        )


def test_mean_hi_brightness_temperature_is_positive(
    cosmology: CosmologyParameters,
) -> None:
    temperature = mean_hi_brightness_temperature_uk(
        0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
    )

    assert np.isfinite(temperature)
    assert temperature > 0


def test_mean_hi_brightness_temperature_matches_formula_at_z_zero(
    cosmology: CosmologyParameters,
) -> None:
    omega_hi = 4.0e-4

    temperature = mean_hi_brightness_temperature_uk(
        0.0,
        cosmology=cosmology,
        omega_hi=omega_hi,
    )

    expected = (
        566.0
        * cosmology.h
        * (omega_hi / 0.003)
    )

    assert temperature == pytest.approx(expected)


def test_mean_hi_brightness_temperature_supports_redshift_array(
    cosmology: CosmologyParameters,
) -> None:
    redshift = np.array([0.1, 0.2, 0.3, 0.4])

    temperature = mean_hi_brightness_temperature_uk(
        redshift,
        cosmology=cosmology,
        omega_hi=4.0e-4,
    )

    assert temperature.shape == redshift.shape
    assert np.all(np.isfinite(temperature))
    assert np.all(temperature > 0)


def test_mean_hi_brightness_temperature_scales_with_omega_hi(
    cosmology: CosmologyParameters,
) -> None:
    low = mean_hi_brightness_temperature_uk(
        0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
    )

    high = mean_hi_brightness_temperature_uk(
        0.3,
        cosmology=cosmology,
        omega_hi=8.0e-4,
    )

    assert high == pytest.approx(2.0 * low)


def test_mean_hi_brightness_temperature_rejects_non_positive_omega_hi(
    cosmology: CosmologyParameters,
) -> None:
    with pytest.raises(ValueError, match="omega_hi must be positive"):
        mean_hi_brightness_temperature_uk(
            0.3,
            cosmology=cosmology,
            omega_hi=0.0,
        )


def test_hi_bias_constant_model() -> None:
    redshift = np.array([0.1, 0.2, 0.3])

    bias = hi_bias(
        redshift,
        bias_0=1.5,
    )

    np.testing.assert_allclose(
        bias,
        np.array([1.5, 1.5, 1.5]),
    )


def test_hi_bias_rejects_non_positive_bias() -> None:
    with pytest.raises(ValueError, match="bias_0 must be positive"):
        hi_bias(
            0.3,
            bias_0=0.0,
        )


def test_hi_temperature_fluctuation_zero_for_zero_overdensity(
    cosmology: CosmologyParameters,
) -> None:
    fluctuation = hi_temperature_fluctuation_uk(
        0.0,
        redshift=0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
        bias=1.0,
    )

    assert fluctuation == pytest.approx(0.0)


def test_hi_temperature_fluctuation_scales_linearly_with_overdensity(
    cosmology: CosmologyParameters,
) -> None:
    first = hi_temperature_fluctuation_uk(
        0.1,
        redshift=0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
        bias=1.0,
    )

    second = hi_temperature_fluctuation_uk(
        0.2,
        redshift=0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
        bias=1.0,
    )

    assert second == pytest.approx(2.0 * first)


def test_hi_temperature_fluctuation_scales_linearly_with_bias(
    cosmology: CosmologyParameters,
) -> None:
    unbiased = hi_temperature_fluctuation_uk(
        0.1,
        redshift=0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
        bias=1.0,
    )

    biased = hi_temperature_fluctuation_uk(
        0.1,
        redshift=0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
        bias=2.0,
    )

    assert biased == pytest.approx(2.0 * unbiased)


def test_hi_temperature_fluctuation_supports_map_array(
    cosmology: CosmologyParameters,
) -> None:
    density_map = np.array(
        [-0.2, -0.1, 0.0, 0.1, 0.2]
    )

    fluctuation = hi_temperature_fluctuation_uk(
        density_map,
        redshift=0.3,
        cosmology=cosmology,
        omega_hi=4.0e-4,
        bias=1.2,
    )

    assert fluctuation.shape == density_map.shape
    assert np.all(np.isfinite(fluctuation))

    assert fluctuation[0] < 0
    assert fluctuation[2] == pytest.approx(0.0)
    assert fluctuation[-1] > 0
