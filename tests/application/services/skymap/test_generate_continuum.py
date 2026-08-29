from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radioskysims_sources.application.skymap.generate_continuum import (
    GalaxyMapSet,
    GenerateGalaxyMapsUseCase,
)
from radioskysims_skymap.application.skymap.ports import SkyMapGeneratorPort
from radioskysims_core.domain.common import FrequencyRange
from radioskysims_sources.domain.radiosources.continuum import Galaxy
from radioskysims_skymap.domain.skymap import NSide, SkyMap


class DummyGenerator(SkyMapGeneratorPort):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        nside: NSide,
        freq_ghz: float,
        components=None,
    ) -> SkyMap:
        self.calls.append(
            {
                "nside": nside,
                "freq_ghz": freq_ghz,
                "components": components,
            }
        )

        return SkyMap(
            dataset=np.zeros(
                hp.nside2npix(nside.value),
                dtype=np.float64,
            ),
            nside=nside,
            unit="uK_CMB",
            freq_ghz=freq_ghz,
        )


def test_generate_galaxy_map_for_each_frequency_bin():
    generator = DummyGenerator()

    use_case = GenerateGalaxyMapsUseCase(generator)

    frequency_range = FrequencyRange.from_floats(
        900.0,
        1100.0,
        100.0,
    )

    result = use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=16,
    )

    assert isinstance(result, GalaxyMapSet)
    assert len(result) == 3

    frequencies = [sky_map.freq_ghz for sky_map in result.maps]

    assert frequencies == [
        0.9,
        1.0,
        1.1,
    ]


def test_generator_receives_galactic_components():
    generator = DummyGenerator()

    use_case = GenerateGalaxyMapsUseCase(generator)

    frequency_range = FrequencyRange.from_floats(
        1000.0,
        1000.0,
    )

    use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=16,
    )

    assert len(generator.calls) == 1

    call = generator.calls[0]

    assert call["components"] == (
        "synchrotron",
        "dust",
        "freefree",
    )


def test_generator_receives_frequency_in_ghz():
    generator = DummyGenerator()

    use_case = GenerateGalaxyMapsUseCase(generator)

    frequency_range = FrequencyRange.from_floats(
        900.0,
        1300.0,
        200.0,
    )

    use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=32,
    )

    frequencies = [call["freq_ghz"] for call in generator.calls]

    assert frequencies == [
        0.9,
        1.1,
        1.3,
    ]


def test_generated_maps_preserve_nside():
    generator = DummyGenerator()

    use_case = GenerateGalaxyMapsUseCase(generator)

    frequency_range = FrequencyRange.from_floats(
        900.0,
        1000.0,
        100.0,
    )

    result = use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=32,
    )

    assert all(sky_map.nside == NSide(32) for sky_map in result.maps)


class RecordingSkyMapRepository:
    def __init__(self):
        self.saved = {}

    def save(
        self,
        key,
        skymap,
        *,
        overwrite=False,
    ):
        self.saved[key] = {
            "skymap": skymap,
            "overwrite": overwrite,
        }

        return f"/fake/{key}.h5"

    def load(
        self,
        key,
    ):
        item = self.saved.get(key)

        if item is None:
            return None

        return item["skymap"]

    def exists(
        self,
        key,
    ):
        return key in self.saved

    def delete(
        self,
        key,
    ):
        if key not in self.saved:
            return False

        del self.saved[key]

        return True


def test_generate_galaxy_maps_can_persist():
    generator = DummyGenerator()
    repository = RecordingSkyMapRepository()

    use_case = GenerateGalaxyMapsUseCase(
        generator,
        repository=repository,
    )

    frequency_range = FrequencyRange(
        min_mhz=900.0,
        max_mhz=1100.0,
        resolution_mhz=100.0,
    )

    result = use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=8,
        persist=True,
    )

    assert len(result.maps) == 3

    assert set(repository.saved) == {
        "galaxy_900mhz_nside8",
        "galaxy_1000mhz_nside8",
        "galaxy_1100mhz_nside8",
    }


def test_generate_galaxy_maps_does_not_persist_by_default():
    generator = DummyGenerator()
    repository = RecordingSkyMapRepository()

    use_case = GenerateGalaxyMapsUseCase(
        generator,
        repository=repository,
    )

    frequency_range = FrequencyRange(
        min_mhz=900.0,
        max_mhz=1000.0,
        resolution_mhz=100.0,
    )

    use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=8,
    )

    assert repository.saved == {}


def test_persist_requires_repository():
    generator = DummyGenerator()

    use_case = GenerateGalaxyMapsUseCase(generator)

    frequency_range = FrequencyRange(
        min_mhz=900.0,
        max_mhz=900.0,
        resolution_mhz=0.0,
    )

    with pytest.raises(ValueError):
        use_case.execute(
            source=Galaxy(),
            frequency_range=frequency_range,
            nside=8,
            persist=True,
        )


def test_generate_galaxy_maps_forwards_overwrite():
    generator = DummyGenerator()
    repository = RecordingSkyMapRepository()

    use_case = GenerateGalaxyMapsUseCase(
        generator,
        repository=repository,
    )

    frequency_range = FrequencyRange(
        min_mhz=900.0,
        max_mhz=900.0,
        resolution_mhz=0.0,
    )

    use_case.execute(
        source=Galaxy(),
        frequency_range=frequency_range,
        nside=8,
        persist=True,
        overwrite=True,
    )

    item = repository.saved["galaxy_900mhz_nside8"]

    assert item["overwrite"] is True
