"""
Application ports for cosmological calculations.

Concrete implementations belong to Infrastructure. The first implementation
is expected to use PyCCL, but the Application layer does not depend on PyCCL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from radioskysims_sources.domain.cosmology import (
    AngularPowerSpectrum,
    CosmologyParameters,
    MatterPowerSpectrum,
    RedshiftWindow,
)


class CosmologyPowerSpectrumPort(ABC):
    """
    Contract for matter power-spectrum calculations.

    Implementations may use PyCCL or another Boltzmann/LSS backend.
    """

    @abstractmethod
    def matter_power_spectrum(
        self,
        *,
        cosmology: CosmologyParameters,
        k_mpc: NDArray[np.float64],
        redshift: float,
    ) -> MatterPowerSpectrum:
        """
        Compute the three-dimensional matter power spectrum P(k, z).

        Parameters
        ----------
        cosmology:
            Cosmological parameters.
        k_mpc:
            Comoving wavenumbers in 1/Mpc.
        redshift:
            Cosmological redshift.

        Returns
        -------
        MatterPowerSpectrum
            Domain representation of P(k, z).
        """
        raise NotImplementedError


class AngularPowerSpectrumPort(ABC):
    """
    Contract for projected angular matter spectra C_ell.
    """

    @abstractmethod
    def angular_cl(
        self,
        *,
        cosmology: CosmologyParameters,
        ell: NDArray[np.integer],
        window: RedshiftWindow,
        bias: float = 1.0,
    ) -> AngularPowerSpectrum:
        """
        Compute the angular matter power spectrum for a redshift window.

        Parameters
        ----------
        cosmology:
            Cosmological parameters.
        ell:
            Multipoles where C_ell is evaluated.
        window:
            Radial/redshift selection function.
        bias:
            Linear tracer bias. Defaults to unity for the matter field.

        Returns
        -------
        AngularPowerSpectrum
            Domain representation of C_ell.
        """
        raise NotImplementedError


class MatterFieldPort(ABC):
    """
    Contract for realizations of angular matter-density fields.

    Concrete implementations may generate Gaussian, lognormal,
    or other physically motivated matter fields.
    """

    @abstractmethod
    def generate(
        self,
        *,
        angular_spectrum: AngularPowerSpectrum,
        nside: int,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """
        Generate a dimensionless matter overdensity map delta_m.
        """
        raise NotImplementedError


# Backward-compatible alias.
GaussianMatterFieldPort = MatterFieldPort
