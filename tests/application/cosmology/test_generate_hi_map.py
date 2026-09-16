import healpy as hp
import numpy as np
import pytest

from radioskysims_sources.application.cosmology import (
    GenerateNeutralHydrogenMapUseCase,
    NeutralHydrogenMapResult,
)
from radioskysims_sources.domain.cosmology import (
    CosmologyParameters,
)
from radioskysims_sources.domain.radiosources.continuum import (
    NeutralHydrogen,
)
from radioskysims_skymap.domain.skymap import SkyMap
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
def use_case() -> GenerateNeutralHydrogenMapUseCase:
    pyccl = PyCCLCosmologyAdapter()
    gaussian = HealpyGaussianMatterFieldAdapter()

    return GenerateNeutralHydrogenMapUseCase(
        angular_power=pyccl,
        gaussian_field=gaussian,
    )


def test_generate_hi_map_returns_result(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_mhz=1100.0,
        nside=16,
        ell_max=32,
        seed=42,
    )

    assert isinstance(
        result,
        NeutralHydrogenMapResult,
    )

    assert isinstance(
        result.skymap,
        SkyMap,
    )


def test_generate_hi_map_has_correct_frequency_and_redshift(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_mhz=1100.0,
        nside=16,
        ell_max=32,
        seed=42,
    )

    assert result.frequency_mhz == pytest.approx(
        1100.0
    )

    assert result.redshift == pytest.approx(
        1420.405751 / 1100.0 - 1.0
    )


def test_generate_hi_map_has_correct_healpix_size(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_mhz=1100.0,
        nside=16,
        ell_max=32,
        seed=42,
    )

    expected_npix = hp.nside2npix(16)

    assert result.matter_overdensity.shape == (
        expected_npix,
    )

    assert result.brightness_temperature_uk.shape == (
        expected_npix,
    )

    assert result.skymap.npix == expected_npix


def test_generate_hi_map_contains_finite_values(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_mhz=1100.0,
        nside=16,
        ell_max=32,
        seed=42,
    )

    assert np.all(
        np.isfinite(
            result.matter_overdensity
        )
    )

    assert np.all(
        np.isfinite(
            result.brightness_temperature_uk
        )
    )


def test_generate_hi_map_has_positive_and_negative_fluctuations(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_mhz=1100.0,
        nside=16,
        ell_max=32,
        seed=42,
    )

    temperature = (
        result.brightness_temperature_uk
    )

    assert np.any(temperature < 0)
    assert np.any(temperature > 0)


def test_generate_hi_map_skymap_metadata(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    result = use_case.execute(
        source=NeutralHydrogen(),
        cosmology=cosmology,
        frequency_mhz=1100.0,
        nside=16,
        ell_max=32,
        seed=42,
    )

    assert result.skymap.unit == "uK"
    assert result.skymap.frame == "galactic"

    assert result.skymap.freq_ghz == pytest.approx(
        1.1
    )

    assert result.skymap.field == "I"


def test_generate_hi_map_is_reproducible(
    use_case: GenerateNeutralHydrogenMapUseCase,
    cosmology: CosmologyParameters,
) -> None:
    kwargs = {
        "source": NeutralHydrogen(),
        "cosmology": cosmology,
        "frequency_mhz": 1100.0,
        "nside": 16,
        "ell_max": 32,
        "seed": 123,
    }

    first = use_case.execute(**kwargs)
    second = use_case.execute(**kwargs)

    np.testing.assert_allclose(
        first.brightness_temperature_uk,
        second.brightness_temperature_uk,
    )
