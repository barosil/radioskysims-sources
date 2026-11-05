from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from radioskysims_sources.application.services.flux_view import FluxDensity


def test_flux_density_converts_inputs_to_float_arrays() -> None:
    flux = FluxDensity(
        freqs_mhz=[900, 1000, 1100],
        flux_by_source={
            "SRC-1": [1, 2, 3],
        },
    )

    assert flux.freqs_mhz.dtype.kind == "f"
    assert flux.flux_by_source["SRC-1"].dtype.kind == "f"

    np.testing.assert_allclose(
        flux.freqs_mhz,
        [900.0, 1000.0, 1100.0],
    )
    np.testing.assert_allclose(
        flux.flux_by_source["SRC-1"],
        [1.0, 2.0, 3.0],
    )


def test_total_sums_flux_from_all_sources() -> None:
    flux = FluxDensity(
        freqs_mhz=[900.0, 1000.0, 1100.0],
        flux_by_source={
            "SRC-1": [1.0, 2.0, 3.0],
            "SRC-2": [0.5, 1.0, 1.5],
        },
    )

    np.testing.assert_allclose(
        flux.total,
        [1.5, 3.0, 4.5],
    )


def test_total_returns_zeros_when_there_are_no_sources() -> None:
    flux = FluxDensity(
        freqs_mhz=[900.0, 1000.0, 1100.0],
        flux_by_source={},
    )

    np.testing.assert_allclose(
        flux.total,
        [0.0, 0.0, 0.0],
    )


def test_view_delegates_to_visualization_package(
    monkeypatch,
) -> None:
    calls: list[
        tuple[
            np.ndarray,
            dict[str, np.ndarray],
            str | None,
            dict[str, object],
        ]
    ] = []

    expected_figure = object()

    def fake_radio_sources_flux(
        freqs_mhz,
        flux_by_source,
        *,
        title=None,
        **kwargs,
    ):
        calls.append(
            (
                freqs_mhz,
                flux_by_source,
                title,
                kwargs,
            )
        )
        return expected_figure

    viz_module = ModuleType("radioskysims_viz")
    sources_module = ModuleType("radioskysims_viz.sources")
    sources_module.radio_sources_flux = fake_radio_sources_flux

    monkeypatch.setitem(
        sys.modules,
        "radioskysims_viz",
        viz_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "radioskysims_viz.sources",
        sources_module,
    )

    flux = FluxDensity(
        freqs_mhz=[1000.0, 1100.0],
        flux_by_source={
            "SRC-1": [2.0, 3.0],
        },
    )

    result = flux.view(
        title="Radio sources",
        linewidth=2,
    )

    assert result is expected_figure
    assert len(calls) == 1

    freqs_mhz, flux_by_source, title, kwargs = calls[0]

    np.testing.assert_allclose(
        freqs_mhz,
        [1000.0, 1100.0],
    )
    np.testing.assert_allclose(
        flux_by_source["SRC-1"],
        [2.0, 3.0],
    )
    assert title == "Radio sources"
    assert kwargs == {"linewidth": 2}


def test_len_returns_number_of_sources() -> None:
    flux = FluxDensity(
        freqs_mhz=[1000.0],
        flux_by_source={
            "SRC-1": [1.0],
            "SRC-2": [2.0],
        },
    )

    assert len(flux) == 2


def test_repr_describes_number_of_sources_and_frequencies() -> None:
    flux = FluxDensity(
        freqs_mhz=[900.0, 1000.0, 1100.0],
        flux_by_source={
            "SRC-1": [1.0, 2.0, 3.0],
            "SRC-2": [2.0, 3.0, 4.0],
        },
    )

    assert repr(flux) == ("<FluxDensity: 2 fonte(s), 3 frequências>")
