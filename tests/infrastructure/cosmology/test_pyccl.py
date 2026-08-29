import numpy as np
import pytest

from radioskysims_sources.application.ports.cosmology import (
    AngularPowerSpectrumPort,
    CosmologyPowerSpectrumPort,
)
from radioskysims_sources.domain.cosmology import (
    AngularPowerSpectrum,
    CosmologyParameters,
    MatterPowerSpectrum,
    RedshiftWindow,
)
from radioskysims_sources.infrastructure.cosmology import (
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


def test_pyccl_adapter_implements_ports() -> None:
    adapter = PyCCLCosmologyAdapter()

    assert isinstance(adapter, CosmologyPowerSpectrumPort)
    assert isinstance(adapter, AngularPowerSpectrumPort)


def test_build_cosmology_preserves_parameters(
    cosmology: CosmologyParameters,
) -> None:
    adapter = PyCCLCosmologyAdapter()

    ccl_cosmology = adapter._build_cosmology(cosmology)

    assert ccl_cosmology["Omega_c"] == pytest.approx(0.25)
    assert ccl_cosmology["Omega_b"] == pytest.approx(0.05)
    assert ccl_cosmology["Omega_m"] == pytest.approx(0.30)
    assert ccl_cosmology["h"] == pytest.approx(0.67)


def test_matter_power_spectrum_returns_domain_model(
    cosmology: CosmologyParameters,
) -> None:
    adapter = PyCCLCosmologyAdapter()

    k = np.logspace(-3, 0, 10)

    result = adapter.matter_power_spectrum(
        cosmology=cosmology,
        k_mpc=k,
        redshift=0.3,
    )

    assert isinstance(result, MatterPowerSpectrum)

    np.testing.assert_allclose(
        result.k_mpc,
        k,
    )

    assert result.redshift == pytest.approx(0.3)
    assert np.all(np.isfinite(result.power_mpc3))
    assert np.all(result.power_mpc3 > 0)


def test_matter_power_decreases_with_redshift_on_linear_scales(
    cosmology: CosmologyParameters,
) -> None:
    adapter = PyCCLCosmologyAdapter()

    k = np.array([0.1])

    today = adapter.matter_power_spectrum(
        cosmology=cosmology,
        k_mpc=k,
        redshift=0.0,
    )

    earlier = adapter.matter_power_spectrum(
        cosmology=cosmology,
        k_mpc=k,
        redshift=1.0,
    )

    assert today.power_mpc3[0] > earlier.power_mpc3[0]


def test_matter_power_rejects_negative_redshift(
    cosmology: CosmologyParameters,
) -> None:
    adapter = PyCCLCosmologyAdapter()

    with pytest.raises(ValueError, match="cannot be negative"):
        adapter.matter_power_spectrum(
            cosmology=cosmology,
            k_mpc=np.array([0.1]),
            redshift=-0.1,
        )


def test_angular_cl_returns_domain_model(
    cosmology: CosmologyParameters,
) -> None:
    adapter = PyCCLCosmologyAdapter()

    window = RedshiftWindow(
        redshift=np.linspace(0.2, 0.3, 32),
        weight=np.ones(32),
    )

    ell = np.arange(2, 32)

    result = adapter.angular_cl(
        cosmology=cosmology,
        ell=ell,
        window=window,
        bias=1.0,
    )

    assert isinstance(result, AngularPowerSpectrum)

    np.testing.assert_array_equal(
        result.ell,
        ell,
    )

    assert result.redshift_min == pytest.approx(0.2)
    assert result.redshift_max == pytest.approx(0.3)

    assert np.all(np.isfinite(result.cls))
    assert np.all(result.cls >= 0)


def test_angular_cl_bias_changes_amplitude(
    cosmology: CosmologyParameters,
) -> None:
    adapter = PyCCLCosmologyAdapter()

    window = RedshiftWindow(
        redshift=np.linspace(0.2, 0.3, 32),
        weight=np.ones(32),
    )

    ell = np.arange(2, 32)

    unbiased = adapter.angular_cl(
        cosmology=cosmology,
        ell=ell,
        window=window,
        bias=1.0,
    )

    biased = adapter.angular_cl(
        cosmology=cosmology,
        ell=ell,
        window=window,
        bias=2.0,
    )

    assert np.mean(biased.cls) > np.mean(unbiased.cls)
