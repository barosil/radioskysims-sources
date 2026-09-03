"""
PyCCL adapter for cosmological calculations.

This module belongs to Infrastructure because it depends directly on PyCCL.
Domain and Application layers remain independent from the concrete cosmology
backend.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from radioskysims_sources.application.ports.cosmology import (
    AngularPowerSpectrumPort,
    CosmologyPowerSpectrumPort,
)
from radioskysims_sources.domain.cosmology import (
    AngularPowerSpectrum,
    CosmologyParameters,
    MatterPowerSpectrum,
    RedshiftWindow,
)


class PyCCLCosmologyAdapter(
    CosmologyPowerSpectrumPort,
    AngularPowerSpectrumPort,
):
    """
    Cosmology backend implemented with PyCCL.

    The adapter translates RadioSkySim domain models into PyCCL objects and
    converts PyCCL outputs back into domain representations.
    """

    @staticmethod
    def _build_cosmology(
        parameters: CosmologyParameters,
    ) -> ccl.Cosmology:
        """
        Build a PyCCL Cosmology from domain parameters.
        """
        return ccl.Cosmology(
            Omega_c=parameters.omega_c,
            Omega_b=parameters.omega_b,
            h=parameters.h,
            sigma8=parameters.sigma8,
            n_s=parameters.n_s,
        )

    def matter_power_spectrum(
        self,
        *,
        cosmology: CosmologyParameters,
        k_mpc: NDArray[np.float64],
        redshift: float,
    ) -> MatterPowerSpectrum:
        """
        Compute the linear matter power spectrum P(k, z).

        Parameters
        ----------
        cosmology:
            Domain cosmological parameters.
        k_mpc:
            Comoving wavenumbers in 1/Mpc.
        redshift:
            Redshift where the spectrum is evaluated.
        """
        k = np.asarray(k_mpc, dtype=float)

        if k.ndim != 1:
            msg = "k_mpc must be a one-dimensional array."
            raise ValueError(msg)

        if k.size == 0:
            msg = "k_mpc cannot be empty."
            raise ValueError(msg)

        if np.any(k <= 0):
            msg = "All k_mpc values must be positive."
            raise ValueError(msg)

        if redshift < 0:
            msg = "redshift cannot be negative."
            raise ValueError(msg)

        ccl_cosmology = self._build_cosmology(cosmology)

        scale_factor = 1.0 / (1.0 + redshift)

        power = ccl.linear_matter_power(
            ccl_cosmology,
            k,
            scale_factor,
        )

        return MatterPowerSpectrum(
            k_mpc=k,
            power_mpc3=np.asarray(power, dtype=float),
            redshift=redshift,
        )

    def angular_cl(
        self,
        *,
        cosmology: CosmologyParameters,
        ell: NDArray[np.integer],
        window: RedshiftWindow,
        bias: float = 1.0,
    ) -> AngularPowerSpectrum:
        """
        Compute angular power spectrum C_ell for a redshift selection window.

        The initial implementation uses a NumberCountsTracer without
        redshift-space distortions and without magnification bias.

        Parameters
        ----------
        cosmology:
            Domain cosmological parameters.
        ell:
            Angular multipoles.
        window:
            Redshift selection function.
        bias:
            Linear tracer bias.
        """
        ell_array = np.asarray(ell, dtype=int)

        if ell_array.ndim != 1:
            msg = "ell must be a one-dimensional array."
            raise ValueError(msg)

        if ell_array.size == 0:
            msg = "ell cannot be empty."
            raise ValueError(msg)

        if np.any(ell_array < 0):
            msg = "ell cannot contain negative values."
            raise ValueError(msg)

        if bias <= 0:
            msg = "bias must be positive."
            raise ValueError(msg)

        ccl_cosmology = self._build_cosmology(cosmology)

        redshift = np.asarray(window.redshift, dtype=float)
        weight = np.asarray(window.weight, dtype=float)

        normalization = np.trapezoid(weight, redshift)

        if normalization <= 0:
            msg = "RedshiftWindow weight integral must be positive."
            raise ValueError(msg)

        normalized_weight = weight / normalization

        bias_values = np.full_like(
            redshift,
            fill_value=float(bias),
            dtype=float,
        )

        tracer = ccl.NumberCountsTracer(
            ccl_cosmology,
            has_rsd=False,
            dndz=(redshift, normalized_weight),
            bias=(redshift, bias_values),
        )

        cls = ccl.angular_cl(
            ccl_cosmology,
            tracer,
            tracer,
            ell_array,
        )

        return AngularPowerSpectrum(
            ell=ell_array,
            cls=np.asarray(cls, dtype=float),
            redshift_min=window.min_z,
            redshift_max=window.max_z,
        )
