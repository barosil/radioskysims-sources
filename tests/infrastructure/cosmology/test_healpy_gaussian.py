import healpy as hp
import numpy as np
import pytest

from radioskysims_sources.application.ports.cosmology import (
    GaussianMatterFieldPort,
)
from radioskysims_sources.domain.cosmology import (
    AngularPowerSpectrum,
)
from radioskysims_sources.infrastructure.cosmology import (
    HealpyGaussianMatterFieldAdapter,
)


@pytest.fixture
def angular_spectrum() -> AngularPowerSpectrum:
    ell = np.arange(2, 32)

    cls = 1.0e-5 / (
        ell * (ell + 1.0)
    )

    return AngularPowerSpectrum(
        ell=ell,
        cls=cls,
        redshift_min=0.2,
        redshift_max=0.3,
    )


def test_healpy_gaussian_adapter_implements_port() -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    assert isinstance(
        adapter,
        GaussianMatterFieldPort,
    )


def test_gaussian_field_has_correct_healpix_size(
    angular_spectrum: AngularPowerSpectrum,
) -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    nside = 16

    result = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=nside,
        seed=42,
    )

    assert result.shape == (
        hp.nside2npix(nside),
    )


def test_gaussian_field_contains_finite_values(
    angular_spectrum: AngularPowerSpectrum,
) -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    result = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=16,
        seed=42,
    )

    assert np.all(np.isfinite(result))


def test_gaussian_field_is_reproducible_with_seed(
    angular_spectrum: AngularPowerSpectrum,
) -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    first = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=16,
        seed=123,
    )

    second = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=16,
        seed=123,
    )

    np.testing.assert_allclose(
        first,
        second,
    )


def test_different_seeds_generate_different_maps(
    angular_spectrum: AngularPowerSpectrum,
) -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    first = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=16,
        seed=1,
    )

    second = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=16,
        seed=2,
    )

    assert not np.array_equal(
        first,
        second,
    )


def test_gaussian_field_has_both_signs(
    angular_spectrum: AngularPowerSpectrum,
) -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    result = adapter.generate(
        angular_spectrum=angular_spectrum,
        nside=16,
        seed=42,
    )

    assert np.any(result < 0)
    assert np.any(result > 0)


@pytest.mark.parametrize(
    "nside",
    [0, 3, -1],
)
def test_gaussian_field_rejects_invalid_nside(
    angular_spectrum: AngularPowerSpectrum,
    nside: int,
) -> None:
    adapter = HealpyGaussianMatterFieldAdapter()

    with pytest.raises(ValueError):
        adapter.generate(
            angular_spectrum=angular_spectrum,
            nside=nside,
            seed=42,
        )
