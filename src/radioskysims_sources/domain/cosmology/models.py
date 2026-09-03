"""
Domain models for cosmological simulations.

This module contains pure domain objects and intentionally does not depend on
external Boltzmann/cosmology libraries such as PyCCL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CosmologyParameters:
    """
    Cosmological parameters required by neutral-hydrogen simulations.

    Parameters
    ----------
    omega_c:
        Cold-dark-matter density parameter.
    omega_b:
        Baryonic matter density parameter.
    h:
        Reduced Hubble parameter, H0 / (100 km s^-1 Mpc^-1).
    sigma8:
        RMS matter fluctuation amplitude on 8 Mpc/h scales.
    n_s:
        Scalar primordial spectral index.
    """

    omega_c: float
    omega_b: float
    h: float
    sigma8: float
    n_s: float

    def __post_init__(self) -> None:
        if self.omega_c <= 0:
            msg = "omega_c must be positive."
            raise ValueError(msg)

        if self.omega_b <= 0:
            msg = "omega_b must be positive."
            raise ValueError(msg)

        if self.h <= 0:
            msg = "h must be positive."
            raise ValueError(msg)

        if self.sigma8 <= 0:
            msg = "sigma8 must be positive."
            raise ValueError(msg)

        if self.n_s <= 0:
            msg = "n_s must be positive."
            raise ValueError(msg)

        if self.omega_c + self.omega_b >= 1:
            msg = "omega_c + omega_b must be smaller than 1 for the current cosmology model."
            raise ValueError(msg)

    @property
    def omega_m(self) -> float:
        """Total non-relativistic matter density used by the initial model."""
        return self.omega_c + self.omega_b


@dataclass(frozen=True, slots=True)
class RedshiftRange:
    """
    Redshift interval sampled at a constant nominal resolution.

    The range follows the same idea as ``FrequencyRange`` but remains local to
    the cosmology domain because redshift is dimensionless.
    """

    min_z: float
    max_z: float
    resolution_z: float

    def __post_init__(self) -> None:
        if self.min_z < 0:
            msg = "min_z cannot be negative."
            raise ValueError(msg)

        if self.max_z < self.min_z:
            msg = "max_z must be greater than or equal to min_z."
            raise ValueError(msg)

        if self.resolution_z < 0:
            msg = "resolution_z cannot be negative."
            raise ValueError(msg)

        if self.min_z != self.max_z and self.resolution_z == 0:
            msg = "resolution_z must be positive when min_z and max_z differ."
            raise ValueError(msg)

    @property
    def value(self) -> np.ndarray:
        """Return redshift samples as a NumPy array."""
        if self.resolution_z == 0:
            return np.array([self.min_z], dtype=float)

        return np.arange(
            self.min_z,
            self.max_z + self.resolution_z / 2,
            self.resolution_z,
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class MatterPowerSpectrum:
    """
    Espectro tridimensional de potência da matéria P(k, z).

    ``k_mpc`` contém os números de onda em 1/Mpc e ``power_mpc3`` contém
    P(k) em Mpc^3 para um redshift específico.
    """

    k_mpc: np.ndarray
    power_mpc3: np.ndarray
    redshift: float

    def __post_init__(self) -> None:
        k = np.asarray(self.k_mpc, dtype=float)
        power = np.asarray(self.power_mpc3, dtype=float)

        if k.ndim != 1:
            msg = "k_mpc must be a one-dimensional array."
            raise ValueError(msg)

        if power.ndim != 1:
            msg = "power_mpc3 must be a one-dimensional array."
            raise ValueError(msg)

        if k.size == 0:
            msg = "MatterPowerSpectrum cannot be empty."
            raise ValueError(msg)

        if k.shape != power.shape:
            msg = "k_mpc and power_mpc3 must have the same shape."
            raise ValueError(msg)

        if np.any(k <= 0):
            msg = "All k_mpc values must be positive."
            raise ValueError(msg)

        if np.any(power < 0):
            msg = "Matter power spectrum values cannot be negative."
            raise ValueError(msg)

        if self.redshift < 0:
            msg = "redshift cannot be negative."
            raise ValueError(msg)

        object.__setattr__(self, "k_mpc", k.copy())
        object.__setattr__(self, "power_mpc3", power.copy())


@dataclass(frozen=True, slots=True)
class AngularPowerSpectrum:
    """
    Espectro angular de potência C_ell.

    Representa a projeção angular das flutuações de matéria associada a uma
    janela em redshift.
    """

    ell: np.ndarray
    cls: np.ndarray
    redshift_min: float
    redshift_max: float

    def __post_init__(self) -> None:
        ell = np.asarray(self.ell, dtype=int)
        cls = np.asarray(self.cls, dtype=float)

        if ell.ndim != 1:
            msg = "ell must be a one-dimensional array."
            raise ValueError(msg)

        if cls.ndim != 1:
            msg = "cls must be a one-dimensional array."
            raise ValueError(msg)

        if ell.size == 0:
            msg = "AngularPowerSpectrum cannot be empty."
            raise ValueError(msg)

        if ell.shape != cls.shape:
            msg = "ell and cls must have the same shape."
            raise ValueError(msg)

        if np.any(ell < 0):
            msg = "ell cannot contain negative values."
            raise ValueError(msg)

        if np.any(cls < 0):
            msg = "C_ell values cannot be negative."
            raise ValueError(msg)

        if self.redshift_min < 0:
            msg = "redshift_min cannot be negative."
            raise ValueError(msg)

        if self.redshift_max < self.redshift_min:
            msg = "redshift_max must be greater than or equal to redshift_min."
            raise ValueError(msg)

        object.__setattr__(self, "ell", ell.copy())
        object.__setattr__(self, "cls", cls.copy())


@dataclass(frozen=True, slots=True)
class RedshiftWindow:
    """
    Janela de seleção cosmológica associada a um canal de frequência.

    Parameters
    ----------
    redshift:
        Coordenadas de redshift.
    weight:
        Função de seleção normalizada ou pesos relativos.
    """

    redshift: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        redshift = np.asarray(self.redshift, dtype=float)
        weight = np.asarray(self.weight, dtype=float)

        if redshift.ndim != 1 or weight.ndim != 1:
            msg = "redshift and weight must be one-dimensional arrays."
            raise ValueError(msg)

        if redshift.size == 0:
            msg = "RedshiftWindow cannot be empty."
            raise ValueError(msg)

        if redshift.shape != weight.shape:
            msg = "redshift and weight must have the same shape."
            raise ValueError(msg)

        if np.any(redshift < 0):
            msg = "redshift cannot contain negative values."
            raise ValueError(msg)

        if np.any(weight < 0):
            msg = "weight cannot contain negative values."
            raise ValueError(msg)

        if not np.any(weight > 0):
            msg = "RedshiftWindow must contain at least one positive weight."
            raise ValueError(msg)

        object.__setattr__(self, "redshift", redshift.copy())
        object.__setattr__(self, "weight", weight.copy())

    @property
    def min_z(self) -> float:
        return float(np.min(self.redshift))

    @property
    def max_z(self) -> float:
        return float(np.max(self.redshift))
