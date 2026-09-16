from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from radioskysims_sources.infrastructure.external.solar_flux_downloader import (
    SFU_TO_JY,
    SolarFluxDownloader,
)


def test_build_url_uses_expected_srd_filename() -> None:
    downloader = SolarFluxDownloader()

    result = downloader._build_url(date(2026, 7, 26))

    assert result.endswith("/2026/L260726.SRD")


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1234", 1.23e4 * SFU_TO_JY),
        ("10", 10.0 * SFU_TO_JY),
        ("1.5", 1.5 * SFU_TO_JY),
        ("0", 0.0),
    ],
)
def test_decode_flux_token_to_jy_valid_values(
    token: str,
    expected: float,
) -> None:
    downloader = SolarFluxDownloader()

    result = downloader._decode_flux_token_to_jy(token)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "token",
    [
        "12/34",
        "invalid",
        "-1",
    ],
)
def test_decode_flux_token_to_jy_rejects_invalid_values(
    token: str,
) -> None:
    downloader = SolarFluxDownloader()

    assert downloader._decode_flux_token_to_jy(token) is None


def test_parse_srs_parses_new_format() -> None:
    downloader = SolarFluxDownloader()

    raw_text = """
: HEADER
::
120000 1000 2000 3000 4000
"""

    result = downloader._parse_srs(raw_text, date(2026, 7, 26))

    assert list(result.columns) == [
        "time",
        "frequency_mhz",
        "flux_jy",
    ]
    assert len(result) == 4

    assert result.iloc[0]["time"] == pd.Timestamp("2026-07-26 12:00:00")
    assert result.iloc[0]["frequency_mhz"] == 245.0


def test_parse_srs_parses_legacy_format() -> None:
    downloader = SolarFluxDownloader()

    raw_text = """
1200 30 10 20
"""

    result = downloader._parse_srs(raw_text, date(2026, 7, 26))

    assert len(result) == 2
    assert result.iloc[0]["time"] == pd.Timestamp(
        "2026-07-26 12:00:30"
    )
    assert result.iloc[0]["frequency_mhz"] == 245.0
    assert result.iloc[1]["frequency_mhz"] == 410.0


def test_parse_srs_ignores_headers_invalid_and_short_lines() -> None:
    downloader = SolarFluxDownloader()

    raw_text = """
: HEADER
garbage
120000 invalid 10
::

130000 10 20
"""

    result = downloader._parse_srs(raw_text, date(2026, 7, 26))

    assert not result.empty
    assert set(result["frequency_mhz"]) <= {
        245.0,
        410.0,
    }


def test_download_days_empty_input_returns_empty_dict() -> None:
    downloader = SolarFluxDownloader()

    result = asyncio.run(downloader._download_days([]))

    assert result == {}



def test_download_days_collects_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = SolarFluxDownloader()

    first = date(2026, 7, 25)
    second = date(2026, 7, 26)

    async def fake_download(_client, day):
        if day == second:
            msg = "network failure"
            raise RuntimeError(msg)
        return "120000 10 20"

    monkeypatch.setattr(
        downloader,
        "_download_text_async",
        fake_download,
    )

    result = asyncio.run(
        downloader._download_days([first, second])

    )

    assert result[first] == "120000 10 20"
    assert result[second] is None



def test_fetch_async_builds_solar_flux_from_downloaded_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = SolarFluxDownloader()

    center = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)

    async def fake_download_days(days):
        return dict.fromkeys(
            days,
            "110000 10 20\n"
            "120000 30 40\n"
            "130000 50 60\n",
        )

    monkeypatch.setattr(
        downloader,
        "_download_days",
        fake_download_days,
    )

    result = asyncio.run(
        downloader.fetch_async(
            center_utc=center,
            window_hours=2,
        )
    )

    assert result.start_utc == datetime(2026, 7, 26, 11, 0)
    assert result.end_utc == datetime(2026, 7, 26, 13, 0)

    assert result.source_url is not None
    assert "flux_jy" in result.dataset

    assert result.dataset.sizes["time"] >= 3
    assert result.dataset.sizes["frequency_mhz"] == 2

    assert result.dataset.attrs["units"] == "Jy"
    assert result.dataset.attrs["source"] == "SRD"



def test_fetch_async_reuses_cached_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = SolarFluxDownloader()

    center = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)

    cached = pd.DataFrame(
        {
            "time": [
                datetime(2026, 7, 26, 12, 0),
            ],
            "frequency_mhz": [245.0],
            "flux_jy": [1.0e6],
        }
    )

    downloader._day_cache["2026-07-26"] = cached

    async def fake_download_days(days):
        return dict.fromkeys(days)

    monkeypatch.setattr(
        downloader,
        "_download_days",
        fake_download_days,
    )

    result = asyncio.run(
        downloader.fetch_async(
            center_utc=center,
            window_hours=2,
        )
    )

    assert result.dataset.sizes["time"] == 1
    assert result.dataset.sizes["frequency_mhz"] == 1



def test_fetch_async_raises_when_no_data_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = SolarFluxDownloader()

    async def fake_download_days(days):
        return dict.fromkeys(days)

    monkeypatch.setattr(
        downloader,
        "_download_days",
        fake_download_days,
    )

    with pytest.raises(
        RuntimeError,
        match="Nenhum dado SRD/SRS disponível",
    ):
        asyncio.run(
            downloader.fetch_async(
                center_utc=datetime(
                    2026,
                    7,
                    26,
                    12,
                    0,
                    tzinfo=UTC,
                ),
                window_hours=2,
            )
        )



def test_fetch_async_raises_when_data_is_outside_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = SolarFluxDownloader()

    async def fake_download_days(days):
        return dict.fromkeys(days, "000000 10 20")

    monkeypatch.setattr(
        downloader,
        "_download_days",
        fake_download_days,
    )

    with pytest.raises(
        RuntimeError,
        match="sem amostras no intervalo",
    ):
        asyncio.run(
            downloader.fetch_async(
                center_utc=datetime(
                    2026,
                    7,
                    26,
                    12,
                    0,
                    tzinfo=UTC,
                ),
                window_hours=1,
            )
        )


def test_fetch_runs_async_interface_synchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = SolarFluxDownloader()

    expected = object()

    async def fake_fetch_async(*, center_utc, window_hours):
        assert center_utc == datetime(
            2026,
            7,
            26,
            12,
            0,
            tzinfo=UTC,
        )
        assert window_hours == 4
        return expected

    monkeypatch.setattr(
        downloader,
        "fetch_async",
        fake_fetch_async,
    )

    result = downloader.fetch(
        center_utc=datetime(
            2026,
            7,
            26,
            12,
            0,
            tzinfo=UTC,
        ),
        window_hours=4,
    )

    assert result is expected
