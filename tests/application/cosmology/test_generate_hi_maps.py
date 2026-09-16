from __future__ import annotations

import numpy as np
import pytest

from radioskysims_sources.application.cosmology import (
    GenerateNeutralHydrogenMapsUseCase,
    GenerateNeutralHydrogenMapUseCase,
    NeutralHydrogenMapSet,
)
from radioskysims_core.domain.common.models import FrequencyRange
from radioskysims_sources.domain.cosmology import CosmologyParameters
from radioskysims_sources.domain.radiosources import NeutralHydrogen
from radioskysims_sources.infrastructure.cosmology import (
    HealpyGaussianMatterFieldAdapter,
    PyCCLCosmologyAdapter,
)


@pytest.fixture
def cosmology() -> CosmologyParameters:
    return CosmologyParameters(
        omega_c=0.25,
        omega_b=0.05,
        h=0.67,
        sigma8=0.81,
        n_s=0.965,
    )


@pytest.fixture
def use_case() -> GenerateNeutralHydrogenMapsUseCase:
    channel_generator = GenerateNeutralHydrogenMapUseCase(
        angular_power=PyCCLCosmologyAdapter(),
        gaussian_field=HealpyGaussianMatterFieldAdapter(),
    )

    return GenerateNeutralHydrogenMapsUseCase(
        channel_generator=channel_generator,
    )


def test_generate_hi_maps_returns_map_set(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_range=frequency_range,
        nside=8,
        ell_max=16,
        seed=42,
    )

    assert isinstance(
        result,
        NeutralHydrogenMapSet,
    )


def test_generate_hi_maps_creates_one_map_per_channel(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_range=frequency_range,
        nside=8,
        ell_max=16,
        seed=42,
    )

    assert len(result.maps) == len(frequency_range.value)

    assert len(result.redshifts) == len(frequency_range.value)


def test_generate_hi_maps_preserves_channel_frequencies(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_range=frequency_range,
        nside=8,
        ell_max=16,
        seed=42,
    )

    frequencies_ghz = np.array([skymap.freq_ghz for skymap in result.maps])

    np.testing.assert_allclose(
        frequencies_ghz,
        frequency_range.value / 1000.0,
    )


def test_generate_hi_maps_redshift_decreases_with_frequency(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1040.0,
        resolution_mhz=10.0,
    )

    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_range=frequency_range,
        nside=8,
        ell_max=16,
        seed=42,
    )

    assert np.all(np.diff(result.redshifts) < 0)


def test_generate_hi_maps_channels_are_not_identical(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_range=frequency_range,
        nside=8,
        ell_max=16,
        seed=42,
    )

    first = np.asarray(result.maps[0].dataset)

    second = np.asarray(result.maps[1].dataset)

    assert not np.array_equal(
        first,
        second,
    )


def test_generate_hi_maps_is_reproducible(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    kwargs = {
        "source": NeutralHydrogen(),
        "cosmology": cosmology,
        "frequency_range": frequency_range,
        "nside": 8,
        "ell_max": 16,
        "seed": 123,
    }

    first = use_case.execute(**kwargs)
    second = use_case.execute(**kwargs)

    for first_map, second_map in zip(
        first.maps,
        second.maps,
        strict=False,
    ):
        np.testing.assert_allclose(
            first_map.dataset.to_numpy(),
            second_map.dataset.to_numpy(),
        )


class _FakeSkyMapRepository:
    def __init__(self) -> None:
        self.saved: dict[str, object] = {}

    def save(
        self,
        key: str,
        skymap,
        *,
        overwrite: bool = False,
    ) -> str:
        if key in self.saved and not overwrite:
            raise FileExistsError(key)

        self.saved[key] = skymap

        return key

    def load(
        self,
        key: str,
    ):
        return self.saved.get(key)

    def exists(
        self,
        key: str,
    ) -> bool:
        return key in self.saved

    def delete(
        self,
        key: str,
    ) -> bool:
        if key not in self.saved:
            return False

        del self.saved[key]
        return True


def test_generate_hi_maps_persists_each_channel(
    cosmology: CosmologyParameters,
) -> None:
    repository = _FakeSkyMapRepository()

    channel_generator = GenerateNeutralHydrogenMapUseCase(
        angular_power=PyCCLCosmologyAdapter(),
        gaussian_field=(HealpyGaussianMatterFieldAdapter()),
    )

    use_case = GenerateNeutralHydrogenMapsUseCase(
        channel_generator=channel_generator,
        repository=repository,
    )

    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1020.0,
        resolution_mhz=10.0,
    )

    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_range=frequency_range,
        nside=8,
        ell_max=16,
        seed=42,
        persist=True,
    )

    assert len(repository.saved) == 3

    expected_keys = {
        "neutral_hydrogen_1000mhz_nside8",
        "neutral_hydrogen_1010mhz_nside8",
        "neutral_hydrogen_1020mhz_nside8",
    }

    assert set(repository.saved) == expected_keys

    for skymap in result.maps:
        assert skymap in repository.saved.values()


def test_generate_hi_maps_persist_requires_repository(
    use_case: GenerateNeutralHydrogenMapsUseCase,
    cosmology: CosmologyParameters,
) -> None:
    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1000.0,
        resolution_mhz=1.0,
    )

    with pytest.raises(
        ValueError,
        match="persist=True requires",
    ):
        use_case.execute(
            source=NeutralHydrogen(),
            cosmology=cosmology,
            frequency_range=frequency_range,
            nside=8,
            ell_max=16,
            seed=42,
            persist=True,
        )


def test_generate_hi_maps_propagates_overwrite(
    cosmology: CosmologyParameters,
) -> None:
    repository = _FakeSkyMapRepository()

    channel_generator = GenerateNeutralHydrogenMapUseCase(
        angular_power=PyCCLCosmologyAdapter(),
        gaussian_field=(HealpyGaussianMatterFieldAdapter()),
    )

    use_case = GenerateNeutralHydrogenMapsUseCase(
        channel_generator=channel_generator,
        repository=repository,
    )

    frequency_range = FrequencyRange(
        min_mhz=1000.0,
        max_mhz=1000.0,
        resolution_mhz=1.0,
    )

    kwargs = {
        "source": NeutralHydrogen(),
        "cosmology": cosmology,
        "frequency_range": frequency_range,
        "nside": 8,
        "ell_max": 16,
        "seed": 42,
        "persist": True,
    }

    use_case.execute(**kwargs)

    with pytest.raises(FileExistsError):
        use_case.execute(**kwargs)

    use_case.execute(
        **kwargs,
        overwrite=True,
    )
