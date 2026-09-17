from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from radioskysims_sources.domain.radiosources.tracks import RadioSourceTrack


def _track_dataset() -> xr.Dataset:
    times = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:10:00",
        ],
        dtype="datetime64[ns]",
    )

    sources = ["SRC-1", "SRC-2"]

    return xr.Dataset(
        {
            "theta_rad": (
                ("source", "time"),
                np.array(
                    [
                        [0.1, 0.3],
                        [0.5, 0.7],
                    ]
                ),
            ),
            "az_rad": (
                ("source", "time"),
                np.array(
                    [
                        [1.0, 1.2],
                        [2.0, 2.2],
                    ]
                ),
            ),
            "el_rad": (
                ("source", "time"),
                np.array(
                    [
                        [0.4, 0.6],
                        [0.8, 1.0],
                    ]
                ),
            ),
            "range_m": (
                ("source", "time"),
                np.array(
                    [
                        [100.0, 120.0],
                        [200.0, 220.0],
                    ]
                ),
            ),
        },
        coords={
            "source": sources,
            "time": times,
        },
    )


def _track() -> RadioSourceTrack:
    return RadioSourceTrack(
        dataset=_track_dataset(),
        source_ids=["SRC-1", "SRC-2"],
        time_range=(
            "2026-07-26T00:00:00",
            "2026-07-26T00:10:00",
        ),
    )


def test_accepts_valid_dataset() -> None:
    track = _track()

    assert track.source_ids == ["SRC-1", "SRC-2"]
    assert set(track.dataset.data_vars) == {
        "theta_rad",
        "az_rad",
        "el_rad",
        "range_m",
    }


def test_accepts_empty_dataset() -> None:
    dataset = xr.Dataset()

    track = RadioSourceTrack(
        dataset=dataset,
        source_ids=[],
        time_range=(
            "2026-07-26T00:00:00",
            "2026-07-26T00:10:00",
        ),
    )

    assert track.dataset is dataset
    assert len(track) == 0


def test_rejects_non_xarray_dataset() -> None:
    with pytest.raises(TypeError, match="dataset deve ser um xr.Dataset"):
        RadioSourceTrack(
            dataset={"theta_rad": []},
            source_ids=[],
            time_range=(
                "2026-07-26T00:00:00",
                "2026-07-26T00:10:00",
            ),
        )


def test_rejects_dataset_missing_required_variable() -> None:
    dataset = _track_dataset().drop_vars("range_m")

    with pytest.raises(
        ValidationError,
        match="Dataset sem variáveis obrigatórias",
    ):
        RadioSourceTrack(
            dataset=dataset,
            source_ids=["SRC-1", "SRC-2"],
            time_range=(
                "2026-07-26T00:00:00",
                "2026-07-26T00:10:00",
            ),
        )


def test_rejects_dataset_missing_required_dimension() -> None:
    dataset = xr.Dataset(
        {
            "theta_rad": ("time", [0.1, 0.2]),
            "az_rad": ("time", [1.0, 1.1]),
            "el_rad": ("time", [0.4, 0.5]),
            "range_m": ("time", [100.0, 110.0]),
        },
        coords={
            "time": np.array(
                [
                    "2026-07-26T00:00:00",
                    "2026-07-26T00:10:00",
                ],
                dtype="datetime64[ns]",
            ),
        },
    )

    with pytest.raises(
        ValidationError,
        match="Dimensões ausentes no dataset",
    ):
        RadioSourceTrack(
            dataset=dataset,
            source_ids=[],
            time_range=(
                "2026-07-26T00:00:00",
                "2026-07-26T00:10:00",
            ),
        )


def test_evaluate_interpolates_and_returns_numpy_arrays(
    monkeypatch,
) -> None:
    track = _track()

    interpolated = xr.Dataset(
        {
            "theta_rad": ("source", [0.2, 0.6]),
            "az_rad": ("source", [1.1, 2.1]),
            "el_rad": ("source", [0.5, 0.9]),
            "range_m": ("source", [110.0, 210.0]),
        },
        coords={
            "source": ["SRC-1", "SRC-2"],
        },
    )

    calls: list[tuple[np.datetime64, dict[str, str]]] = []

    def fake_interp(
        _self,
        *,
        time,
        kwargs,
    ):
        calls.append((time, kwargs))
        return interpolated

    monkeypatch.setattr(
        xr.Dataset,
        "interp",
        fake_interp,
    )

    result = track.evaluate(
        np.datetime64("2026-07-26T00:05:00"),
    )

    assert len(calls) == 1

    t_ns, kwargs = calls[0]

    assert t_ns == np.datetime64(
        "2026-07-26T00:05:00",
        "ns",
    )
    assert kwargs == {"fill_value": "extrapolate"}

    np.testing.assert_allclose(
        result["theta_rad"],
        [0.2, 0.6],
    )
    np.testing.assert_allclose(
        result["az_rad"],
        [1.1, 2.1],
    )
    np.testing.assert_allclose(
        result["el_rad"],
        [0.5, 0.9],
    )
    np.testing.assert_allclose(
        result["range_m"],
        [110.0, 210.0],
    )


def test_filter_by_flux_keeps_sources_at_or_above_threshold() -> None:
    track = _track()

    filtered = track.filter_by_flux(
        {
            "SRC-1": 1.0,
            "SRC-2": 2.0,
        },
        min_flux_jy=2.0,
    )

    assert filtered.source_ids == ["SRC-2"]
    assert list(filtered.dataset["source"].values) == ["SRC-2"]
    assert len(filtered) == 1


def test_filter_by_flux_returns_empty_track_when_no_source_passes() -> None:
    track = _track()

    filtered = track.filter_by_flux(
        {
            "SRC-1": 0.5,
            "SRC-2": 0.9,
        },
        min_flux_jy=1.0,
    )

    assert filtered.source_ids == []
    assert len(filtered) == 0
    assert not filtered.dataset.data_vars


def test_len_returns_number_of_source_ids() -> None:
    track = _track()

    assert len(track) == 2
