from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from radioskysims_sources.application.ports.cosmology import (
    MatterFieldPort,
)
from radioskysims_sources.domain.cosmology import (
    AngularPowerSpectrum,
)
from radioskysims_sources.infrastructure.cosmology.healpy_gaussian import (
    HealpyGaussianMatterFieldAdapter,
)


class HealpyLognormalMatterFieldAdapter(MatterFieldPort):
    """
    Generate lognormal HEALPix matter overdensity fields.

    The implementation starts from a Gaussian realization g and applies

        delta_LN = exp(g - sigma_g^2 / 2) - 1

    where sigma_g^2 is the variance of the Gaussian realization.

    This guarantees delta_LN > -1 and approximately preserves
    a zero ensemble mean.
    """

    def __init__(self) -> None:
        self._gaussian = HealpyGaussianMatterFieldAdapter()

    def generate(
        self,
        *,
        angular_spectrum: AngularPowerSpectrum,
        nside: int,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        gaussian_map = self._gaussian.generate(
            angular_spectrum=angular_spectrum,
            nside=nside,
            seed=seed,
        )

        gaussian_map = np.asarray(
            gaussian_map,
            dtype=np.float64,
        )

        variance = float(
            np.var(
                gaussian_map,
                dtype=np.float64,
            )
        )

        lognormal_map = np.exp(gaussian_map - 0.5 * variance) - 1.0

        result = np.asarray(
            lognormal_map,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(result)):
            raise RuntimeError("Lognormal matter realization contains non-finite values.")

        if np.any(result <= -1.0):
            raise RuntimeError("Lognormal matter realization must satisfy delta_m > -1.")

        return result
