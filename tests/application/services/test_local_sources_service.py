from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from radioskysims_sources.application.services.local_sources_service import (
    LocalSourcesService,
)


class _FakeRepository:
    def __init__(self, cached: Any = None) -> None:
        self.cached = cached
        self.load_calls: list[dict[str, Any]] = []
        self.saved: list[Any] = []

    def load_interval(self, **kwargs: Any) -> Any:
        self.load_calls.append(kwargs)
        return self.cached

    def save(self, value: Any) -> Any:
        self.saved.append(value)
        return value


class _FakeDownloader:
    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def fetch(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


class _FakeTrackProvider:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def compute_track(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


def _service(
    *,
    cached: Any = None,
    downloaded: Any = None,
    track_provider: Any = None,
    default_window_hours: int = 2,
) -> tuple[LocalSourcesService, _FakeRepository, _FakeDownloader]:
    repository = _FakeRepository(cached=cached)
    downloader = _FakeDownloader(result=downloaded)

    service = LocalSourcesService(
        solar_flux_downloader=downloader,
        solar_flux_repository=repository,
        local_source_track_provider=track_provider,
        default_window_hours=default_window_hours,
    )
    return service, repository, downloader


def test_build_local_sources_returns_sun_mvp() -> None:
    service, _, _ = _service()

    result = service.build_local_sources()

    assert len(result.sources) == 1
    sun = result.sources[0]
    assert sun.name == "Sun"
    assert sun.kind == "sun"
    assert sun.has_track is True
    assert sun.has_signal is True


def test_get_solar_flux_returns_cached_value_without_download() -> None:
    cached = SimpleNamespace(name="cached")
    service, repository, downloader = _service(cached=cached)

    result = service.get_solar_flux(
        datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
    )

    assert result is cached
    assert len(repository.load_calls) == 1
    assert downloader.calls == []


def test_get_solar_flux_downloads_and_saves_on_cache_miss() -> None:
    downloaded = SimpleNamespace(name="downloaded")
    service, repository, downloader = _service(downloaded=downloaded)

    result = service.get_solar_flux(
        datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
    )

    assert result is downloaded
    assert len(downloader.calls) == 1
    assert repository.saved == [downloaded]

    call = downloader.calls[0]
    assert call["center_utc"] == datetime(2026, 7, 26, 12, 0)
    assert call["window_hours"] == 2


def test_get_solar_flux_uses_custom_window() -> None:
    cached = SimpleNamespace(name="cached")
    service, repository, _ = _service(
        cached=cached,
        default_window_hours=8,
    )

    service.get_solar_flux(
        datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        window_hours=4,
    )

    call = repository.load_calls[0]

    assert call["start_utc"] == datetime(2026, 7, 26, 10, 0)
    assert call["end_utc"] == datetime(2026, 7, 26, 14, 0)


def test_get_local_source_track_requires_provider() -> None:
    service, _, _ = _service()

    with pytest.raises(RuntimeError, match="sem provider de track"):
        service.get_local_source_track(
            source_name="Sun",
            observer=object(),
            time_range=object(),
        )


def test_get_local_source_track_delegates_to_provider() -> None:
    expected = SimpleNamespace(name="track")
    provider = _FakeTrackProvider(expected)
    service, _, _ = _service(track_provider=provider)

    observer = object()
    time_range = object()

    result = service.get_local_source_track(
        source_name="Sun",
        observer=observer,
        time_range=time_range,
    )

    assert result is expected
    assert provider.calls == [
        {
            "source_name": "Sun",
            "observer": observer,
            "time_range": time_range,
        }
    ]
