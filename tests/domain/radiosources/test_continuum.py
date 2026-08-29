from __future__ import annotations

import pytest

from radioskysims_sources.domain.radiosources.continuum import (
    ContinuumSource,
    Galaxy,
)


def test_continuum_source_creation():
    source = ContinuumSource(
        name="Diffuse source",
        components=("synchrotron",),
    )

    assert source.name == "Diffuse source"
    assert source.components == ("synchrotron",)


def test_continuum_source_requires_name():
    with pytest.raises(ValueError):
        ContinuumSource(
            name="",
            components=("synchrotron",),
        )


def test_continuum_source_requires_components():
    with pytest.raises(ValueError):
        ContinuumSource(
            name="Diffuse source",
            components=(),
        )


def test_galaxy_defaults():
    galaxy = Galaxy()

    assert galaxy.name == "Milky Way"
    assert galaxy.components == (
        "synchrotron",
        "dust",
        "freefree",
    )


def test_galaxy_rejects_non_galactic_component():
    with pytest.raises(ValueError):
        Galaxy(
            components=(
                "synchrotron",
                "cmb",
            )
        )


def test_neutral_hydrogen_defaults():
    from radioskysims_sources.domain.radiosources.continuum import NeutralHydrogen

    source = NeutralHydrogen()

    assert source.name == "Neutral Hydrogen"
    assert source.components == ("21cm",)
    assert source.omega_hi_0 == 4.0e-4
    assert source.hi_bias_0 == 1.0


def test_neutral_hydrogen_is_continuum_source():
    from radioskysims_sources.domain.radiosources.continuum import (
        ContinuumSource,
        NeutralHydrogen,
    )

    source = NeutralHydrogen()

    assert isinstance(source, ContinuumSource)


def test_neutral_hydrogen_rejects_unsupported_component():
    import pytest

    from radioskysims_sources.domain.radiosources.continuum import NeutralHydrogen

    with pytest.raises(
        ValueError,
        match="currently supports only",
    ):
        NeutralHydrogen(
            components=("21cm", "other"),
        )


def test_neutral_hydrogen_rejects_non_positive_omega_hi():
    import pytest

    from radioskysims_sources.domain.radiosources.continuum import NeutralHydrogen

    with pytest.raises(
        ValueError,
        match="omega_hi_0 must be positive",
    ):
        NeutralHydrogen(
            omega_hi_0=0.0,
        )


def test_neutral_hydrogen_rejects_non_positive_bias():
    import pytest

    from radioskysims_sources.domain.radiosources.continuum import NeutralHydrogen

    with pytest.raises(
        ValueError,
        match="hi_bias_0 must be positive",
    ):
        NeutralHydrogen(
            hi_bias_0=0.0,
        )


def test_neutral_hydrogen_is_exported_from_radiosources():
    from radioskysims_sources.domain.radiosources import NeutralHydrogen

    source = NeutralHydrogen()

    assert source.name == "Neutral Hydrogen"
