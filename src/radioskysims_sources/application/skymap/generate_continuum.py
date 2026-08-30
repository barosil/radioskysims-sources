from __future__ import annotations

from dataclasses import dataclass
from typing import final

from radioskysims_skymap.application.ports.skymap_repository import (
    SkyMapRepositoryPort,
)
from radioskysims_skymap.application.skymap.ports import SkyMapGeneratorPort
from radioskysims_core.domain.common import FrequencyRange
from radioskysims_sources.domain.radiosources.continuum import Galaxy
from radioskysims_skymap.domain.skymap import NSide, SkyMap


@dataclass(frozen=True, slots=True)
class GalaxyMapSet:
    """
    Conjunto de mapas galácticos associados à grade de frequências
    da simulação.
    """

    source: Galaxy
    frequency_range: FrequencyRange
    maps: tuple[SkyMap, ...]

    def __post_init__(self) -> None:
        expected = len(self.frequency_range.value)

        if len(self.maps) != expected:
            msg = (
                "GalaxyMapSet map count does not match FrequencyRange: "
                f"expected {expected}, received {len(self.maps)}."
            )
            raise ValueError(
                msg
            )

    def __len__(self) -> int:
        return len(self.maps)


@final
class GenerateGalaxyMapsUseCase:
    """
    Caso de uso para geração dos mapas difusos da Galáxia.

    Gera um SkyMap para cada bin de frequência e, opcionalmente,
    persiste cada mapa através de um SkyMapRepositoryPort.
    """

    def __init__(
        self,
        generator: SkyMapGeneratorPort,
        repository: SkyMapRepositoryPort | None = None,
    ) -> None:
        self._generator = generator
        self._repository = repository

    def execute(
        self,
        *,
        source: Galaxy,
        frequency_range: FrequencyRange,
        nside: int | NSide,
        persist: bool = False,
        overwrite: bool = False,
    ) -> GalaxyMapSet:
        """
        Gera um SkyMap HEALPix para cada frequência de FrequencyRange.

        FrequencyRange utiliza MHz internamente, enquanto o
        SkyMapGeneratorPort/PySM3 recebe frequência em GHz.

        Se ``persist=True``, cada mapa gerado também é salvo
        através do SkyMapRepositoryPort configurado no use case.
        """

        if not isinstance(source, Galaxy):
            msg = "source must be a Galaxy."
            raise TypeError(msg)

        if not isinstance(
            frequency_range,
            FrequencyRange,
        ):
            msg = "frequency_range must be a FrequencyRange."
            raise TypeError(msg)

        if persist and self._repository is None:
            msg = "persist=True requires a SkyMap repository."
            raise ValueError(msg)

        resolved_nside = nside if isinstance(nside, NSide) else NSide(nside)

        generated_maps: list[SkyMap] = []

        for freq_mhz in frequency_range.value:
            freq_ghz = float(freq_mhz) / 1000.0

            sky_map = self._generator.generate(
                nside=resolved_nside,
                freq_ghz=freq_ghz,
                components=source.components,
            )

            if not isinstance(
                sky_map,
                SkyMap,
            ):
                msg = "SkyMapGeneratorPort must return SkyMap."
                raise TypeError(msg)

            if persist:
                assert self._repository is not None

                frequency_mhz = float(freq_mhz)

                if frequency_mhz.is_integer():
                    frequency_label = str(int(frequency_mhz))
                else:
                    frequency_label = f"{frequency_mhz:g}"

                key = f"galaxy_{frequency_label}mhz_nside{resolved_nside.value}"

                self._repository.save(
                    key,
                    sky_map,
                    overwrite=overwrite,
                )

            generated_maps.append(sky_map)

        return GalaxyMapSet(
            source=source,
            frequency_range=frequency_range,
            maps=tuple(generated_maps),
        )
