from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radioskysims_sources.application.cosmology.generate_hi_map import (
    GenerateNeutralHydrogenMapUseCase,
    NeutralHydrogenMapResult,
)
from radioskysims_skymap.application.ports.skymap_repository import (
    SkyMapRepositoryPort,
)
from radioskysims_core.domain.common.models import FrequencyRange
from radioskysims_sources.domain.cosmology import CosmologyParameters
from radioskysims_sources.domain.radiosources.continuum import NeutralHydrogen
from radioskysims_skymap.domain.skymap import SkyMap


@dataclass(frozen=True, slots=True)
class NeutralHydrogenMapSet:
    """HI maps generated for all channels of a frequency range."""

    source: NeutralHydrogen
    frequency_range: FrequencyRange
    maps: tuple[SkyMap, ...]
    redshifts: np.ndarray


class GenerateNeutralHydrogenMapsUseCase:
    """Generate one HI SkyMap for each frequency channel."""

    def __init__(
        self,
        *,
        channel_generator: GenerateNeutralHydrogenMapUseCase,
        repository: SkyMapRepositoryPort | None = None,
    ) -> None:
        self._channel_generator = channel_generator
        self._repository = repository

    @staticmethod
    def repository_key(
        *,
        frequency_mhz: float,
        nside: int,
    ) -> str:
        """
        Build the logical repository key for one HI map.
        """
        frequency_value = float(frequency_mhz)

        if frequency_value.is_integer():
            frequency_label = str(int(frequency_value))
        else:
            frequency_label = f"{frequency_value:g}"

        return f"neutral_hydrogen_{frequency_label}mhz_nside{nside}"

    def execute(
        self,
        *,
        source: NeutralHydrogen,
        cosmology: CosmologyParameters,
        frequency_range: FrequencyRange,
        nside: int,
        ell_max: int | None = None,
        window_width_z: float = 0.01,
        seed: int | None = None,
        persist: bool = False,
        overwrite: bool = False,
    ) -> NeutralHydrogenMapSet:
        frequencies = np.asarray(
            frequency_range.value,
            dtype=float,
        )

        if frequencies.size == 0:
            msg = "frequency_range must contain at least one channel."
            raise ValueError(msg)

        if persist and self._repository is None:
            msg = "persist=True requires a SkyMap repository."
            raise ValueError(msg)

        results: list[NeutralHydrogenMapResult] = []

        for index, frequency_mhz in enumerate(frequencies):
            channel_seed = None if seed is None else seed + index

            result = self._channel_generator.execute(
                source=source,
                cosmology=cosmology,
                frequency_mhz=float(frequency_mhz),
                nside=nside,
                ell_max=ell_max,
                window_width_z=window_width_z,
                seed=channel_seed,
            )

            if persist:
                assert self._repository is not None

                key = self.repository_key(
                    frequency_mhz=float(frequency_mhz),
                    nside=nside,
                )

                self._repository.save(
                    key,
                    result.skymap,
                    overwrite=overwrite,
                )

            results.append(result)

        maps = tuple(result.skymap for result in results)

        redshifts = np.asarray(
            [result.redshift for result in results],
            dtype=float,
        )

        return NeutralHydrogenMapSet(
            source=source,
            frequency_range=frequency_range,
            maps=maps,
            redshifts=redshifts,
        )
