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
            msg = "nside must be an integer."
            raise TypeError(msg)

        if not hp.isnsideok(nside, nest=True):
            msg = f"Invalid HEALPix nside: {nside}. nside must be a positive power of two."
            raise ValueError(
                msg
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
            msg = "Angular power spectrum cannot be empty."
            raise ValueError(msg)

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
            msg = (
                "Healpy returned an unexpected map shape: "
                f"expected {(expected_npix,)}, received {result.shape}."
            )
            raise RuntimeError(
                msg
            )

        if not np.all(np.isfinite(result)):
            msg = "Gaussian matter realization contains non-finite values."
            raise RuntimeError(msg)

        return result
