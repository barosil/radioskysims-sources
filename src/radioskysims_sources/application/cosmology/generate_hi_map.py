"""
Use case for generating a neutral-hydrogen 21-cm SkyMap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radioskysims_sources.application.ports.cosmology import (
    AngularPowerSpectrumPort,
    MatterFieldPort,
)
from radioskysims_sources.domain.cosmology import (
    CosmologyParameters,
    RedshiftWindow,
    frequency_to_redshift,
    hi_temperature_fluctuation_uk,
)
from radioskysims_sources.domain.radiosources.continuum import (
    NeutralHydrogen,
)
from radioskysims_skymap.domain.skymap import SkyMap


@dataclass(frozen=True, slots=True)
class NeutralHydrogenMapResult:
    """Result of a single-frequency HI map generation."""

    frequency_mhz: float
    redshift: float
    matter_overdensity: np.ndarray
    brightness_temperature_uk: np.ndarray
    skymap: SkyMap


class GenerateNeutralHydrogenMapUseCase:
    """
    Generate a Gaussian HI 21-cm map for one observed frequency.
    """

    def __init__(
        self,
        *,
        angular_power: AngularPowerSpectrumPort,
        gaussian_field: MatterFieldPort,
    ) -> None:
        self._angular_power = angular_power
        self._matter_field = gaussian_field

    def execute(
        self,
        *,
        source: NeutralHydrogen,
        cosmology: CosmologyParameters,
        frequency_mhz: float,
        nside: int,
        ell_max: int | None = None,
        window_width_z: float = 0.01,
        seed: int | None = None,
    ) -> NeutralHydrogenMapResult:
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive.")

        if window_width_z <= 0:
            raise ValueError("window_width_z must be positive.")

        redshift = float(frequency_to_redshift(frequency_mhz))

        half_width = window_width_z / 2.0

        min_z = max(
            0.0,
            redshift - half_width,
        )
        max_z = redshift + half_width

        z_grid = np.linspace(
            min_z,
            max_z,
            64,
        )

        window = RedshiftWindow(
            redshift=z_grid,
            weight=np.ones_like(z_grid),
        )

        if ell_max is None:
            ell_max = 3 * nside - 1

        if ell_max < 2:
            raise ValueError("ell_max must be greater than or equal to 2.")

        ell = np.arange(
            2,
            ell_max + 1,
            dtype=int,
        )

        angular_spectrum = self._angular_power.angular_cl(
            cosmology=cosmology,
            ell=ell,
            window=window,
            bias=1.0,
        )

        matter_map = self._matter_field.generate(
            angular_spectrum=angular_spectrum,
            nside=nside,
            seed=seed,
        )

        temperature_map = np.asarray(
            hi_temperature_fluctuation_uk(
                matter_map,
                redshift=redshift,
                cosmology=cosmology,
                omega_hi=source.omega_hi_0,
                bias=source.hi_bias_0,
            ),
            dtype=np.float64,
        )

        skymap = SkyMap(
            dataset=temperature_map,
            nside=nside,
            unit="uK",
            frame="galactic",
            freq_ghz=frequency_mhz / 1000.0,
            field="I",
        )

        return NeutralHydrogenMapResult(
            frequency_mhz=float(frequency_mhz),
            redshift=redshift,
            matter_overdensity=matter_map.copy(),
            brightness_temperature_uk=temperature_map.copy(),
            skymap=skymap,
        )
