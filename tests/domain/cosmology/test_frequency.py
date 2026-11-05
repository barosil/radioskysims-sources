import numpy as np
import pytest

from radioskysims_core.domain.common.models import FrequencyRange
from radioskysims_sources.domain.cosmology import (
    HI_21CM_REST_FREQUENCY_MHZ,
    frequency_range_to_redshift,
    frequency_range_to_redshift_range,
    frequency_to_redshift,
    redshift_to_frequency,
)


def test_rest_frequency_corresponds_to_zero_redshift() -> None:
    redshift = frequency_to_redshift(HI_21CM_REST_FREQUENCY_MHZ)

    assert redshift == pytest.approx(0.0)


def test_frequency_to_redshift_scalar() -> None:
    redshift = frequency_to_redshift(1000.0)

    expected = HI_21CM_REST_FREQUENCY_MHZ / 1000.0 - 1.0

    assert redshift == pytest.approx(expected)


def test_redshift_to_frequency_scalar() -> None:
    frequency = redshift_to_frequency(0.42)

    expected = HI_21CM_REST_FREQUENCY_MHZ / 1.42

    assert frequency == pytest.approx(expected)


def test_frequency_redshift_round_trip() -> None:
    frequencies = np.array([980.0, 1000.0, 1100.0, 1200.0, 1260.0])

    redshifts = frequency_to_redshift(frequencies)
    recovered = redshift_to_frequency(redshifts)

    np.testing.assert_allclose(
        recovered,
        frequencies,
        rtol=1e-12,
    )


def test_frequency_range_generates_redshift_per_channel() -> None:
    frequency_range = FrequencyRange(
        min_mhz=980.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    redshifts = frequency_range_to_redshift(frequency_range)

    assert redshifts.shape == frequency_range.value.shape
    assert np.all(np.diff(redshifts) < 0)


def test_bingo_frequency_limits_map_to_expected_redshift_range() -> None:
    frequency_range = FrequencyRange(
        min_mhz=980.0,
        max_mhz=1260.0,
        resolution_mhz=10.0,
    )

    redshift_range = frequency_range_to_redshift_range(frequency_range)

    assert redshift_range.min_z == pytest.approx(HI_21CM_REST_FREQUENCY_MHZ / 1260.0 - 1.0)
    assert redshift_range.max_z == pytest.approx(HI_21CM_REST_FREQUENCY_MHZ / 980.0 - 1.0)


@pytest.mark.parametrize(
    "frequency",
    [0.0, -1.0, -100.0],
)
def test_frequency_to_redshift_rejects_non_positive_frequency(
    frequency: float,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        frequency_to_redshift(frequency)


def test_frequency_above_rest_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        frequency_to_redshift(HI_21CM_REST_FREQUENCY_MHZ + 1.0)


def test_negative_redshift_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        redshift_to_frequency(-0.1)
