# domain/satellite/chunks.py
from __future__ import annotations

import numpy as np
import xarray as xr


def _bytes_per_element(dtype: np.dtype) -> int:
    """Return itemsize in bytes for dtype. For float64 -> 8."""
    return np.dtype(dtype).itemsize


def _estimate_chunk_bytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    """Estimate bytes for a chunk with given shape and dtype."""
    return int(np.prod(shape)) * int(_bytes_per_element(dtype))


def _group_satellites_by_constellation(ds: xr.Dataset):
    """
    Return dictionary constellation -> array of satellite indices (or values).
    If 'constellation' coord missing, return single group named 'ALL'.
    """
    if "constellation" not in ds.coords and "constellation" not in ds:
        return {"ALL": ds["satellite"].values}

    const_vals = ds["constellation"].values
    sats = ds["satellite"].values
    groups = {}
    for c in np.unique(const_vals):
        groups[str(c)] = sats[const_vals == c]
    return groups


def set_chunks(
    ds: xr.Dataset,
    var: str = "positions",
    *,
    max_chunk_mb: float | None = None,
    allow_merge_constellations: bool = True,
) -> xr.Dataset:
    """
    Smart chunk setter.

    Behavior (per your request):
    - Default (max_chunk_mb is None): start with chunk per satellite,
      keep full time, keep full xyz (i.e. chunk = {"satellite":1, "time": -1, "xyz": ds.sizes["xyz"]})
      and return ds.chunk({...}).
    - If max_chunk_mb is provided (optional, not default), function will:
        - start chunk_sat = 1, chunk_time = full time
        - if estimated chunk bytes > max_chunk_mb -> reduce chunk_time to fit
        - if estimated chunk bytes < max_chunk_mb and allow_merge_constellations:
            try to aggregate satellites by constellation to increase sat chunk size
        - never chunk the xyz axis (xyz is always full)
    - `var` must exist in ds (typically 'positions').
    - Chunk dict uses keys exactly: "satellite", "time", "xyz".

    Returns the chunked xr.Dataset.
    """
    if var not in ds:
        msg = f"Variable '{var}' not present in dataset"
        raise KeyError(msg)

    # dtype for positions (float64 expected)
    dtype = ds[var].dtype
    
    n_time = int(ds.sizes["time"])
    n_xyz = int(ds.sizes["xyz"])

    # DEFAULT: no size limit -> 1 sat per chunk, time full, xyz full
    if max_chunk_mb is None:
        chunks = {"satellite": 1, "time": -1, "xyz": n_xyz}
        return ds.chunk(chunks)

    # Else: try to respect provided limit (max_chunk_mb in MB)
    max_bytes = int(max_chunk_mb * 1024 * 1024)

    # 1) initial chunk candidate: (1, full time, full xyz)
    chunk_sat = 1
    chunk_time = n_time
    chunk_xyz = n_xyz
    estimated = _estimate_chunk_bytes((chunk_sat, chunk_time, chunk_xyz), dtype)

    # 2) If it exceeds max_bytes -> reduce time
    if estimated > max_bytes:
        # bytes per single time-step for 1 satellite
        bytes_per_time = _estimate_chunk_bytes((1, 1, chunk_xyz), dtype)
        if bytes_per_time <= 0:
            chunk_time = 1
        else:
            chunk_time = max(int(max_bytes // bytes_per_time), 1)
        chunks = {"satellite": 1, "time": chunk_time, "xyz": chunk_xyz}
        return ds.chunk(chunks)

    # 3) If it's under the limit, consider aggregating satellites by constellation
    if allow_merge_constellations:
        groups = _group_satellites_by_constellation(ds)
        # compute maximum satellites we can pack per chunk without exceeding limit
        best_sat_count = 1
        for _, sats in groups.items():
            k = len(sats)
            est = _estimate_chunk_bytes((k, n_time, chunk_xyz), dtype)
            if est <= max_bytes:
                # prefer larger groups if they fit
                best_sat_count = max(best_sat_count, k)
        chunks = {"satellite": best_sat_count, "time": -1, "xyz": chunk_xyz}
        return ds.chunk(chunks)

    # fallback
    return ds.chunk({"satellite": 1, "time": -1, "xyz": chunk_xyz})
