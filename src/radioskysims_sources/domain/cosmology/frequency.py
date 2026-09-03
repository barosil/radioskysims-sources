"""
Frequency/redshift conversions for the neutral-hydrogen 21-cm line.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from radioskysims_core.domain.common.models import FrequencyRange
from radioskysims_sources.domain.cosmology.models import RedshiftRange


HI_21CM_REST_FREQUENCY_MHZ: Final[float] = 1420.405751


def frequency_to_redshift(
    frequency_mhz: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert observed 21-cm frequency in MHz to cosmological redshift.

    Uses

        z = nu_rest / nu_obs - 1

    Parameters
    ----------
    frequency_mhz:
        Observed frequency or array of frequencies in MHz.

    Returns
    -------
    float or numpy.ndarray
        Corresponding cosmological redshift.
    """
    frequency = np.asarray(frequency_mhz, dtype=float)

    if np.any(frequency <= 0):
        msg = "Observed frequency must be positive."
        raise ValueError(msg)

    redshift = HI_21CM_REST_FREQUENCY_MHZ / frequency - 1.0

    if np.any(redshift < 0):
        msg = (
            "Observed HI frequency cannot exceed the 21-cm rest frequency "
            f"({HI_21CM_REST_FREQUENCY_MHZ} MHz)."
        )
        raise ValueError(msg)

    if np.ndim(frequency_mhz) == 0:
        return float(redshift)

    return redshift


def redshift_to_frequency(
    redshift: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert cosmological redshift to observed 21-cm frequency in MHz.

    Uses

        nu_obs = nu_rest / (1 + z)
    """
    z = np.asarray(redshift, dtype=float)

    if np.any(z < 0):
        msg = "Redshift cannot be negative."
        raise ValueError(msg)

    frequency = HI_21CM_REST_FREQUENCY_MHZ / (1.0 + z)

    if np.ndim(redshift) == 0:
        return float(frequency)

    return frequency


def frequency_range_to_redshift(
    frequency_range: FrequencyRange,
) -> np.ndarray:
    """
    Convert every frequency channel in a FrequencyRange to redshift.

    Notes
    -----
    Increasing observed frequency corresponds to decreasing redshift.
    Consequently, the returned array follows the frequency-channel order and
    is normally decreasing.
    """
    return np.asarray(
        frequency_to_redshift(frequency_range.value),
        dtype=float,
    )


def frequency_range_to_redshift_range(
    frequency_range: FrequencyRange,
) -> RedshiftRange:
    """
    Convert a FrequencyRange into its equivalent redshift limits.

    Because frequency and redshift are inversely related:

    - max frequency -> min redshift
    - min frequency -> max redshift

    The resolution is defined here as the smallest absolute separation between
    adjacent redshift samples. Since equally spaced frequency channels are not
    equally spaced in redshift, this object represents the interval bounds and
    a nominal redshift resolution rather than the exact channel coordinates.
    """
    redshifts = frequency_range_to_redshift(frequency_range)

    min_z = float(np.min(redshifts))
    max_z = float(np.max(redshifts))

    if redshifts.size == 1:
        resolution_z = 0.0
    else:
        resolution_z = float(np.min(np.abs(np.diff(np.sort(redshifts)))))

    return RedshiftRange(
        min_z=min_z,
        max_z=max_z,
        resolution_z=resolution_z,
    )
