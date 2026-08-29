"""
Neutral-hydrogen 21-cm brightness-temperature domain models.

The functions in this module are pure physical-domain calculations and do not
depend on PyCCL, Healpy or other infrastructure libraries.
"""

from __future__ import annotations

import numpy as np

from radioskysims_sources.domain.cosmology.models import CosmologyParameters


def dimensionless_hubble_parameter(
    redshift: float | np.ndarray,
    cosmology: CosmologyParameters,
) -> float | np.ndarray:
    """
    Compute E(z) = H(z) / H0 for the initial flat Lambda-CDM model.

    The current domain model assumes:

        Omega_Lambda = 1 - Omega_m

    and neglects radiation at the low redshifts relevant to BINGO.
    """
    z = np.asarray(redshift, dtype=float)

    if np.any(z < 0):
        raise ValueError("redshift cannot be negative.")

    omega_m = cosmology.omega_m
    omega_lambda = 1.0 - omega_m

    ez = np.sqrt(
        omega_m * (1.0 + z) ** 3
        + omega_lambda
    )

    if np.ndim(redshift) == 0:
        return float(ez)

    return ez


def mean_hi_brightness_temperature_uk(
    redshift: float | np.ndarray,
    *,
    cosmology: CosmologyParameters,
    omega_hi: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the mean post-reionization HI brightness temperature in microkelvin.

    Uses the commonly adopted intensity-mapping relation

        T_bar(z) =
            566 * h
            * [H0 / H(z)]
            * [Omega_HI(z) / 0.003]
            * (1 + z)^2  microK

    where H0 / H(z) = 1 / E(z).

    Parameters
    ----------
    redshift:
        Cosmological redshift.
    cosmology:
        Cosmological parameters.
    omega_hi:
        Neutral-hydrogen density parameter Omega_HI(z).

    Returns
    -------
    float or numpy.ndarray
        Mean HI brightness temperature in microkelvin.
    """
    z = np.asarray(redshift, dtype=float)
    omega = np.asarray(omega_hi, dtype=float)

    if np.any(z < 0):
        raise ValueError("redshift cannot be negative.")

    if np.any(omega <= 0):
        raise ValueError("omega_hi must be positive.")

    try:
        z, omega = np.broadcast_arrays(z, omega)
    except ValueError as exc:
        raise ValueError(
            "redshift and omega_hi must be broadcast-compatible."
        ) from exc

    ez = np.asarray(
        dimensionless_hubble_parameter(z, cosmology),
        dtype=float,
    )

    temperature_uk = (
        566.0
        * cosmology.h
        * (omega / 0.003)
        * (1.0 + z) ** 2
        / ez
    )

    if np.ndim(redshift) == 0 and np.ndim(omega_hi) == 0:
        return float(temperature_uk)

    return np.asarray(temperature_uk, dtype=float)


def hi_bias(
    redshift: float | np.ndarray,
    *,
    bias_0: float = 1.0,
) -> float | np.ndarray:
    """
    Return the initial constant linear HI bias model.

    This is intentionally simple. A redshift-dependent b_HI(z) model can later
    replace it without changing the rest of the simulation pipeline.
    """
    z = np.asarray(redshift, dtype=float)

    if np.any(z < 0):
        raise ValueError("redshift cannot be negative.")

    if bias_0 <= 0:
        raise ValueError("bias_0 must be positive.")

    result = np.full_like(
        z,
        fill_value=float(bias_0),
        dtype=float,
    )

    if np.ndim(redshift) == 0:
        return float(result)

    return result


def hi_temperature_fluctuation_uk(
    matter_overdensity: float | np.ndarray,
    *,
    redshift: float | np.ndarray,
    cosmology: CosmologyParameters,
    omega_hi: float | np.ndarray,
    bias: float | np.ndarray = 1.0,
) -> float | np.ndarray:
    """
    Convert matter overdensity into HI brightness-temperature fluctuation.

    Linear approximation:

        delta_T_HI =
            T_bar_HI(z) * b_HI(z) * delta_m

    Parameters
    ----------
    matter_overdensity:
        Matter overdensity delta_m.
    redshift:
        Cosmological redshift.
    cosmology:
        Cosmological parameters.
    omega_hi:
        Neutral-hydrogen density parameter.
    bias:
        HI linear bias.

    Returns
    -------
    float or numpy.ndarray
        Brightness-temperature fluctuation in microkelvin.
    """
    delta_m = np.asarray(matter_overdensity, dtype=float)
    z = np.asarray(redshift, dtype=float)
    omega = np.asarray(omega_hi, dtype=float)
    bias_array = np.asarray(bias, dtype=float)

    if np.any(bias_array <= 0):
        raise ValueError("bias must be positive.")

    mean_temperature = np.asarray(
        mean_hi_brightness_temperature_uk(
            z,
            cosmology=cosmology,
            omega_hi=omega,
        ),
        dtype=float,
    )

    try:
        delta_m, mean_temperature, bias_array = np.broadcast_arrays(
            delta_m,
            mean_temperature,
            bias_array,
        )
    except ValueError as exc:
        raise ValueError(
            "matter_overdensity, redshift, omega_hi and bias "
            "must be broadcast-compatible."
        ) from exc

    fluctuation = mean_temperature * bias_array * delta_m

    if (
        np.ndim(matter_overdensity) == 0
        and np.ndim(redshift) == 0
        and np.ndim(omega_hi) == 0
        and np.ndim(bias) == 0
    ):
        return float(fluctuation)

    return np.asarray(fluctuation, dtype=float)
