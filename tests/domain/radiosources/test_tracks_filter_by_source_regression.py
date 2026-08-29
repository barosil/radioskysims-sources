import numpy as np
import xarray as xr

from radioskysims_sources.domain.radiosources.tracks import RadioSourceTrack


def _make_track() -> RadioSourceTrack:
    time = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:00:05",
            "2026-07-26T00:00:10",
        ],
        dtype="datetime64[ns]",
    )

    source = np.array(
        [
            "source_a",
            "source_b",
        ],
        dtype=str,
    )

    shape = (
        source.size,
        time.size,
    )

    ds = xr.Dataset(
        data_vars={
            "theta_rad": (
                ("source", "time"),
                np.array(
                    [
                        [0.1, 0.2, 0.3],
                        [1.1, 1.2, 1.3],
                    ],
                    dtype=float,
                ),
            ),
            "az_rad": (
                ("source", "time"),
                np.array(
                    [
                        [0.4, 0.5, 0.6],
                        [1.4, 1.5, 1.6],
                    ],
                    dtype=float,
                ),
            ),
            "el_rad": (
                ("source", "time"),
                np.array(
                    [
                        [0.7, 0.8, 0.9],
                        [1.7, 1.8, 1.9],
                    ],
                    dtype=float,
                ),
            ),
            "range_m": (
                ("source", "time"),
                np.full(
                    shape,
                    1.0e20,
                    dtype=float,
                ),
            ),
        },
        coords={
            "source": source,
            "time": time,
        },
    )

    return RadioSourceTrack(
        dataset=ds,
        source_ids=[
            "source_a",
            "source_b",
        ],
        time_range=(
            "2026-07-26T00:00:00",
            "2026-07-26T00:00:10",
        ),
    )


def test_filter_by_source_preserves_source_dimension():
    track = _make_track()

    filtered = track.filter_by_source(
        "source_b"
    )

    assert filtered.source_ids == [
        "source_b"
    ]

    assert filtered.dataset.sizes[
        "source"
    ] == 1

    assert filtered.dataset.sizes[
        "time"
    ] == 3

    assert list(
        filtered.dataset[
            "source"
        ].values
    ) == [
        "source_b"
    ]

    np.testing.assert_allclose(
        filtered.dataset[
            "theta_rad"
        ].values,
        np.array(
            [
                [
                    1.1,
                    1.2,
                    1.3,
                ]
            ]
        ),
    )


def test_filter_by_source_result_remains_valid_track():
    track = _make_track()

    filtered = track.filter_by_source(
        "source_a"
    )

    assert isinstance(
        filtered,
        RadioSourceTrack,
    )

    assert set(
        filtered.dataset.dims
    ) == {
        "source",
        "time",
    }

    assert filtered.dataset[
        "theta_rad"
    ].dims == (
        "source",
        "time",
    )

    assert filtered.dataset[
        "az_rad"
    ].dims == (
        "source",
        "time",
    )

    assert filtered.dataset[
        "el_rad"
    ].dims == (
        "source",
        "time",
    )

    assert filtered.dataset[
        "range_m"
    ].dims == (
        "source",
        "time",
    )


def test_filter_by_source_does_not_modify_original_track():
    track = _make_track()

    filtered = track.filter_by_source(
        "source_a"
    )

    assert track.dataset.sizes[
        "source"
    ] == 2

    assert track.source_ids == [
        "source_a",
        "source_b",
    ]

    assert filtered.dataset.sizes[
        "source"
    ] == 1
