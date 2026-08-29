"""
Gaussian cosmological-field generator implemented with Healpy.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
from numpy.typing import NDArray

from radioskysims_sources.application.ports.cosmology import (
    MatterFieldPort,
)
from radioskysims_sources.domain.cosmology import AngularPowerSpectrum


class HealpyGaussianMatterFieldAdapter(MatterFieldPort):
    """
    Generate Gaussian HEALPix realizations from angular matter spectra.
    """

    def generate(
        self,
        *,
        angular_spectrum: AngularPowerSpectrum,
        nside: int,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        if isinstance(nside, bool) or not isinstance(nside, int):
            raise TypeError("nside must be an integer.")

        if not hp.isnsideok(nside, nest=True):
            raise ValueError(
                f"Invalid HEALPix nside: {nside}. nside must be a positive power of two."
            )

        ell = np.asarray(
            angular_spectrum.ell,
            dtype=int,
        )

        cls = np.asarray(
            angular_spectrum.cls,
            dtype=float,
        )

        if ell.size == 0:
            raise ValueError("Angular power spectrum cannot be empty.")

        lmax = int(np.max(ell))

        full_cls = np.zeros(
            lmax + 1,
            dtype=float,
        )

        full_cls[ell] = cls

        if seed is not None:
            np.random.seed(seed)

        matter_map = hp.synfast(
            full_cls,
            nside=nside,
            lmax=lmax,
            pol=False,
            new=True,
        )

        result = np.asarray(
            matter_map,
            dtype=np.float64,
        )

        expected_npix = hp.nside2npix(nside)

        if result.shape != (expected_npix,):
            raise RuntimeError(
                "Healpy returned an unexpected map shape: "
                f"expected {(expected_npix,)}, received {result.shape}."
            )

        if not np.all(np.isfinite(result)):
            raise RuntimeError("Gaussian matter realization contains non-finite values.")

        return result
