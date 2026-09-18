from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_sources.domain.satellite.chunks import set_chunks


def _dataset(
    *,
    satellites: int = 4,
    times: int = 5,
    constellations: list[str] | None = None,
) -> xr.Dataset:
    coords = {
        "satellite": np.arange(1, satellites + 1),
        "time": np.arange(times),
        "xyz": ["x", "y", "z"],
    }
    if constellations is not None:
        coords["constellation"] = ("satellite", constellations)
    return xr.Dataset(
        {
            "positions": (
                ("satellite", "time", "xyz"),
                np.zeros((satellites, times, 3)),
            )
        },
        coords=coords,
    )


def test_set_chunks_rejects_missing_variable():
    dataset = _dataset().drop_vars("positions")

    with pytest.raises(KeyError, match="Variable 'positions' not present"):
        set_chunks(dataset)


def test_set_chunks_default_uses_one_satellite_and_full_time_and_xyz():
    result = set_chunks(_dataset(satellites=3, times=4))

    assert result.chunks == {
        "satellite": (1, 1, 1),
        "time": (4,),
        "xyz": (3,),
    }


def test_set_chunks_reduces_time_when_single_satellite_exceeds_limit():
    result = set_chunks(_dataset(satellites=2, times=10), max_chunk_mb=0.0001)

    assert result.chunks["satellite"] == (1, 1)
    assert result.chunks["time"] == (4, 4, 2)
    assert result.chunks["xyz"] == (3,)


def test_set_chunks_uses_at_least_one_time_step_for_tiny_limit():
    result = set_chunks(_dataset(times=2), max_chunk_mb=0)

    assert result.chunks["satellite"] == (1, 1, 1, 1)
    assert result.chunks["time"] == (1, 1)


def test_set_chunks_merges_largest_constellation_that_fits():
    result = set_chunks(
        _dataset(
            satellites=5,
            times=2,
            constellations=["GPS", "GPS", "GALILEO", "GALILEO", "GALILEO"],
        ),
        max_chunk_mb=0.00015,
    )

    assert result.chunks == {
        "satellite": (3, 2),
        "time": (2,),
        "xyz": (3,),
    }


def test_set_chunks_without_constellation_treats_all_satellites_as_one_group():
    result = set_chunks(_dataset(satellites=4, times=2), max_chunk_mb=0.0002)

    assert result.chunks["satellite"] == (4,)
    assert result.chunks["time"] == (2,)


def test_set_chunks_can_disable_constellation_merge():
    result = set_chunks(
        _dataset(
            satellites=3,
            times=2,
            constellations=["GPS", "GPS", "GALILEO"],
        ),
        max_chunk_mb=0.0002,
        allow_merge_constellations=False,
    )

    assert result.chunks == {
        "satellite": (1, 1, 1),
        "time": (2,),
        "xyz": (3,),
    }
