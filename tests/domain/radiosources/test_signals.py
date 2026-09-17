from __future__ import annotations

import numpy as np

from radioskysims_sources.domain.radiosources.models import (
    RadioSource,
    SpectralModel,
)
from radioskysims_sources.domain.radiosources.signals import (
    RadioSourceSignalEvaluator,
)


def _source(
    source_id: str,
    *,
    s0_jy: float,
    nu0_mhz: float = 1000.0,
    alpha: float = 1.0,
) -> RadioSource:
    return RadioSource(
        id=source_id,
        ra_deg=10.0,
        dec_deg=-5.0,
        spectral_model=SpectralModel(
            nu0_mhz=nu0_mhz,
            s0_jy=s0_jy,
            alpha=alpha,
        ),
    )


def test_evaluate_flux_batch_returns_sample_per_source() -> None:
    sources = [
        _source(
            "SRC-1",
            s0_jy=2.0,
        ),
        _source(
            "SRC-2",
            s0_jy=4.0,
        ),
    ]
    freqs_mhz = np.array([1000.0, 2000.0])

    result = RadioSourceSignalEvaluator.evaluate_flux_batch(
        sources,
        freqs_mhz,
    )

    assert len(result) == 2

    first, second = result

    assert first.source_id == "SRC-1"
    assert first.freq_mhz == 1500.0
    assert first.flux_jy == 1.5

    assert second.source_id == "SRC-2"
    assert second.freq_mhz == 1500.0
    assert second.flux_jy == 3.0


def test_evaluate_flux_batch_accepts_frequency_list() -> None:
    source = _source(
        "SRC",
        s0_jy=3.0,
        alpha=0.0,
    )

    result = RadioSourceSignalEvaluator.evaluate_flux_batch(
        [source],
        [900.0, 1000.0, 1100.0],
    )

    assert len(result) == 1

    sample = result[0]

    assert sample.source_id == "SRC"
    assert sample.freq_mhz == 1000.0
    assert sample.flux_jy == 3.0


def test_evaluate_flux_batch_with_no_sources_returns_empty_list() -> None:
    result = RadioSourceSignalEvaluator.evaluate_flux_batch(
        [],
        np.array([1000.0]),
    )

    assert result == []


def test_evaluate_flux_grid_returns_curve_for_each_source() -> None:
    sources = [
        _source(
            "SRC-1",
            s0_jy=2.0,
        ),
        _source(
            "SRC-2",
            s0_jy=4.0,
        ),
    ]
    freqs_mhz = np.array([1000.0, 2000.0])

    result = RadioSourceSignalEvaluator.evaluate_flux_grid(
        sources,
        freqs_mhz,
    )

    assert set(result) == {"SRC-1", "SRC-2"}

    np.testing.assert_allclose(
        result["SRC-1"],
        np.array([2.0, 1.0]),
    )
    np.testing.assert_allclose(
        result["SRC-2"],
        np.array([4.0, 2.0]),
    )
