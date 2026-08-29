from __future__ import annotations

import numpy as np

from radioskysims_sources.application.ports.cosmology import (
    MatterFieldPort,
)
from radioskysims_sources.domain.cosmology import (
    AngularPowerSpectrum,
)
from radioskysims_sources.infrastructure.cosmology import (
    HealpyLognormalMatterFieldAdapter,
)


def _angular_spectrum() -> AngularPowerSpectrum:
    ell = np.arange(
        2,
        17,
        dtype=int,
    )

    cls = np.full(
        ell.shape,
        1.0e-5,
        dtype=float,
    )

    return AngularPowerSpectrum(
        ell=ell,
        cls=cls,
        redshift_min=0.1,
        redshift_max=0.2,
    )


def test_lognormal_adapter_implements_matter_field_port() -> None:
    adapter = HealpyLognormalMatterFieldAdapter()

    assert isinstance(
        adapter,
        MatterFieldPort,
    )


def test_lognormal_field_has_expected_healpix_shape() -> None:
    adapter = HealpyLognormalMatterFieldAdapter()

    result = adapter.generate(
        angular_spectrum=_angular_spectrum(),
        nside=8,
        seed=42,
    )

    assert result.shape == (12 * 8**2,)


def test_lognormal_field_is_finite() -> None:
    adapter = HealpyLognormalMatterFieldAdapter()

    result = adapter.generate(
        angular_spectrum=_angular_spectrum(),
        nside=8,
        seed=42,
    )

    assert np.all(np.isfinite(result))


def test_lognormal_field_is_bounded_below_minus_one() -> None:
    adapter = HealpyLognormalMatterFieldAdapter()

    result = adapter.generate(
        angular_spectrum=_angular_spectrum(),
        nside=8,
        seed=42,
    )

    assert np.all(result > -1.0)


def test_lognormal_field_is_reproducible() -> None:
    adapter = HealpyLognormalMatterFieldAdapter()

    first = adapter.generate(
        angular_spectrum=_angular_spectrum(),
        nside=8,
        seed=42,
    )

    second = adapter.generate(
        angular_spectrum=_angular_spectrum(),
        nside=8,
        seed=42,
    )

    np.testing.assert_allclose(
        first,
        second,
    )


def test_lognormal_field_differs_from_gaussian() -> None:
    from radioskysims_sources.infrastructure.cosmology import (
        HealpyGaussianMatterFieldAdapter,
    )

    spectrum = _angular_spectrum()

    gaussian = HealpyGaussianMatterFieldAdapter().generate(
        angular_spectrum=spectrum,
        nside=8,
        seed=42,
    )

    lognormal = HealpyLognormalMatterFieldAdapter().generate(
        angular_spectrum=spectrum,
        nside=8,
        seed=42,
    )

    assert not np.array_equal(
        gaussian,
        lognormal,
    )


def test_lognormal_field_has_approximately_zero_mean_over_ensemble() -> None:
    adapter = HealpyLognormalMatterFieldAdapter()
    spectrum = _angular_spectrum()

    means = []

    for seed in range(16):
        field = adapter.generate(
            angular_spectrum=spectrum,
            nside=8,
            seed=seed,
        )
        means.append(float(np.mean(field)))

    ensemble_mean = float(np.mean(means))

    assert abs(ensemble_mean) < 5.0e-2


def test_lognormal_field_has_finite_nonnegative_angular_power() -> None:
    import healpy as hp

    adapter = HealpyLognormalMatterFieldAdapter()

    field = adapter.generate(
        angular_spectrum=_angular_spectrum(),
        nside=8,
        seed=42,
    )

    observed_cl = hp.anafast(
        field,
        lmax=16,
    )

    assert np.all(np.isfinite(observed_cl))
    assert np.all(observed_cl >= -1.0e-12)
    assert np.any(observed_cl[2:] > 0.0)
