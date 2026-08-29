import pytest

from radioskysims_sources.application.ports.cosmology import (
    AngularPowerSpectrumPort,
    CosmologyPowerSpectrumPort,
)


def test_cosmology_power_spectrum_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        CosmologyPowerSpectrumPort()


def test_angular_power_spectrum_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        AngularPowerSpectrumPort()
