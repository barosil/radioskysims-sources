from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_sources.domain.signal_source_group import (
    RadioSourceGroupSignalSource,
)


class _FakeRadioSource:
    def __init__(
        self,
        name: str,
        *,
        ra_deg: float | None,
        dec_deg: float | None,
        psd_values: tuple[float, ...] = (1.0, 2.0),
    ) -> None:
        self.name = name
        self.source_type = "radiosource"
        self._psd_values = np.asarray(psd_values, dtype=float)
        self.psd_calls = 0

        data_vars: dict[str, tuple[str, list[float]]] = {
            "theta": ("time", [0.1, 0.1]),
            "az": ("time", [0.2, 0.2]),
            "el": ("time", [0.3, 0.3]),
            "range": ("time", [1.0, 1.0]),
        }

        if ra_deg is not None:
            data_vars["ra"] = ("time", [ra_deg, ra_deg])

        if dec_deg is not None:
            data_vars["dec"] = ("time", [dec_deg, dec_deg])

        self._tracks = xr.Dataset(data_vars)

    @property
    def tracks(self) -> xr.Dataset:
        return self._tracks

    def psd(self) -> xr.DataArray:
        self.psd_calls += 1
        return xr.DataArray(
            self._psd_values,
            dims="frequency",
            coords={"frequency": np.arange(len(self._psd_values), dtype=float)},
            attrs={"units": "W/m²/Hz"},
        )


def test_group_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="requer ao menos uma fonte"):
        RadioSourceGroupSignalSource("empty", [])


def test_group_exposes_name_type_and_first_track() -> None:
    first = _FakeRadioSource("A", ra_deg=10.0, dec_deg=-20.0)
    second = _FakeRadioSource("B", ra_deg=10.0, dec_deg=-20.0)

    group = RadioSourceGroupSignalSource("group", [first, second])

    assert group.name == "group"
    assert group.source_type == "radiosource_group"
    assert group.tracks is first.tracks


def test_group_psd_sums_member_fluxes() -> None:
    first = _FakeRadioSource(
        "A",
        ra_deg=10.0,
        dec_deg=-20.0,
        psd_values=(1.0, 2.0),
    )
    second = _FakeRadioSource(
        "B",
        ra_deg=10.0,
        dec_deg=-20.0,
        psd_values=(3.0, 4.0),
    )

    group = RadioSourceGroupSignalSource("group", [first, second])

    result = group.psd()

    np.testing.assert_allclose(result.values, [4.0, 6.0])
    assert result.name == "psd"
    assert result.attrs["units"] == "W/m²/Hz"
    assert result.attrs["n_sources"] == 2
    assert result.attrs["group_name"] == "group"


def test_group_psd_is_cached() -> None:
    first = _FakeRadioSource("A", ra_deg=10.0, dec_deg=-20.0)
    second = _FakeRadioSource("B", ra_deg=10.0, dec_deg=-20.0)

    group = RadioSourceGroupSignalSource("group", [first, second])

    first_result = group.psd()
    second_result = group.psd()

    assert second_result is first_result
    assert first.psd_calls == 1
    assert second.psd_calls == 1


def test_group_by_position_empty_input_returns_empty_list() -> None:
    assert RadioSourceGroupSignalSource.group_by_position([]) == []


def test_group_by_position_groups_colocated_sources() -> None:
    first = _FakeRadioSource("A", ra_deg=10.0, dec_deg=-20.0)
    second = _FakeRadioSource(
        "B",
        ra_deg=10.0 + 5.0e-7,
        dec_deg=-20.0 + 5.0e-7,
    )

    result = RadioSourceGroupSignalSource.group_by_position(
        [first, second],
        ra_dec_tol_deg=1.0e-6,
    )

    assert len(result) == 1
    assert isinstance(result[0], RadioSourceGroupSignalSource)
    assert result[0].source_type == "radiosource_group"


def test_group_by_position_preserves_separated_sources() -> None:
    first = _FakeRadioSource("A", ra_deg=10.0, dec_deg=-20.0)
    second = _FakeRadioSource("B", ra_deg=11.0, dec_deg=-20.0)

    result = RadioSourceGroupSignalSource.group_by_position([first, second])

    assert result == [first, second]


def test_group_by_position_preserves_source_without_radec() -> None:
    valid = _FakeRadioSource("A", ra_deg=10.0, dec_deg=-20.0)
    missing = _FakeRadioSource("B", ra_deg=None, dec_deg=None)

    result = RadioSourceGroupSignalSource.group_by_position([valid, missing])

    assert len(result) == 2
    assert valid in result
    assert missing in result


def test_collapse_by_fwhm_empty_input_returns_empty_list() -> None:
    assert RadioSourceGroupSignalSource.collapse_by_fwhm([]) == []


def test_collapse_by_fwhm_groups_sources_inside_threshold() -> None:
    first = _FakeRadioSource("A", ra_deg=10.0, dec_deg=0.0)
    second = _FakeRadioSource("B", ra_deg=10.2, dec_deg=0.0)

    result = RadioSourceGroupSignalSource.collapse_by_fwhm(
        [first, second],
        fwhm_deg=1.0,
        collapse_factor=0.5,
    )

    assert len(result) == 1
    assert isinstance(result[0], RadioSourceGroupSignalSource)


def test_collapse_by_fwhm_preserves_sources_outside_threshold() -> None:
    first = _FakeRadioSource("A", ra_deg=10.0, dec_deg=0.0)
    second = _FakeRadioSource("B", ra_deg=11.0, dec_deg=0.0)

    result = RadioSourceGroupSignalSource.collapse_by_fwhm(
        [first, second],
        fwhm_deg=1.0,
        collapse_factor=0.5,
    )

    assert result == [first, second]


def test_collapse_by_fwhm_preserves_source_without_radec() -> None:
    valid = _FakeRadioSource("A", ra_deg=10.0, dec_deg=0.0)
    missing = _FakeRadioSource("B", ra_deg=None, dec_deg=None)

    result = RadioSourceGroupSignalSource.collapse_by_fwhm(
        [valid, missing],
        fwhm_deg=1.0,
    )

    assert len(result) == 2
    assert valid in result
    assert missing in result
