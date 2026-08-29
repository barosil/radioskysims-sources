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
            raise ValueError("omega_c must be positive.")

        if self.omega_b <= 0:
            raise ValueError("omega_b must be positive.")

        if self.h <= 0:
            raise ValueError("h must be positive.")

        if self.sigma8 <= 0:
            raise ValueError("sigma8 must be positive.")

        if self.n_s <= 0:
            raise ValueError("n_s must be positive.")

        if self.omega_c + self.omega_b >= 1:
            raise ValueError(
                "omega_c + omega_b must be smaller than 1 for the current cosmology model."
            )

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
            raise ValueError("min_z cannot be negative.")

        if self.max_z < self.min_z:
            raise ValueError("max_z must be greater than or equal to min_z.")

        if self.resolution_z < 0:
            raise ValueError("resolution_z cannot be negative.")

        if self.min_z != self.max_z and self.resolution_z == 0:
            raise ValueError("resolution_z must be positive when min_z and max_z differ.")

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
            raise ValueError("k_mpc must be a one-dimensional array.")

        if power.ndim != 1:
            raise ValueError("power_mpc3 must be a one-dimensional array.")

        if k.size == 0:
            raise ValueError("MatterPowerSpectrum cannot be empty.")

        if k.shape != power.shape:
            raise ValueError("k_mpc and power_mpc3 must have the same shape.")

        if np.any(k <= 0):
            raise ValueError("All k_mpc values must be positive.")

        if np.any(power < 0):
            raise ValueError("Matter power spectrum values cannot be negative.")

        if self.redshift < 0:
            raise ValueError("redshift cannot be negative.")

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
            raise ValueError("ell must be a one-dimensional array.")

        if cls.ndim != 1:
            raise ValueError("cls must be a one-dimensional array.")

        if ell.size == 0:
            raise ValueError("AngularPowerSpectrum cannot be empty.")

        if ell.shape != cls.shape:
            raise ValueError("ell and cls must have the same shape.")

        if np.any(ell < 0):
            raise ValueError("ell cannot contain negative values.")

        if np.any(cls < 0):
            raise ValueError("C_ell values cannot be negative.")

        if self.redshift_min < 0:
            raise ValueError("redshift_min cannot be negative.")

        if self.redshift_max < self.redshift_min:
            raise ValueError("redshift_max must be greater than or equal to redshift_min.")

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
            raise ValueError("redshift and weight must be one-dimensional arrays.")

        if redshift.size == 0:
            raise ValueError("RedshiftWindow cannot be empty.")

        if redshift.shape != weight.shape:
            raise ValueError("redshift and weight must have the same shape.")

        if np.any(redshift < 0):
            raise ValueError("redshift cannot contain negative values.")

        if np.any(weight < 0):
            raise ValueError("weight cannot contain negative values.")

        if not np.any(weight > 0):
            raise ValueError("RedshiftWindow must contain at least one positive weight.")

        object.__setattr__(self, "redshift", redshift.copy())
        object.__setattr__(self, "weight", weight.copy())

    @property
    def min_z(self) -> float:
        return float(np.min(self.redshift))

    @property
    def max_z(self) -> float:
        return float(np.max(self.redshift))
