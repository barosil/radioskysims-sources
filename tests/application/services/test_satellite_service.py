from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from radioskysims_sources.application.services.satellite_service import (
    SatelliteService,
)


class _FakeRepository:
    def __init__(self) -> None:
        self.satellites: list[Any] = []
        self.query_sats_result: list[Any] = []
        self.constellation_results: list[list[Any]] = []
        self.verify_result = True

        self.query_sats_calls: list[dict[str, Any]] = []
        self.query_constellation_calls: list[dict[str, Any]] = []
        self.update_orbital_calls: list[dict[str, Any]] = []
        self.celestrak_calls = 0
        self.verify_calls: list[dict[str, Any]] = []
        self.reset_calls: list[dict[str, Any]] = []

        self.raise_query_sats: Exception | None = None
        self.raise_query_constellation: Exception | None = None
        self.raise_update_orbital: Exception | None = None
        self.raise_celestrak: Exception | None = None
        self.raise_verify: Exception | None = None
        self.raise_reset: Exception | None = None

    def query_sats(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
    ) -> list[Any]:
        if self.raise_query_sats is not None:
            raise self.raise_query_sats

        self.query_sats_calls.append(
            {"norad_ids": norad_ids, "epoch": epoch}
        )
        return self.query_sats_result

    def query_constellation(
        self,
        constellations: list[str],
        epoch: datetime | None = None,
    ) -> list[Any]:
        if self.raise_query_constellation is not None:
            raise self.raise_query_constellation

        self.query_constellation_calls.append(
            {"constellations": constellations, "epoch": epoch}
        )

        if self.constellation_results:
            result = self.constellation_results.pop(0)
            self.satellites = result
            return result

        return []

    def update_orbital_elements_sync(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
    ) -> None:
        if self.raise_update_orbital is not None:
            raise self.raise_update_orbital

        self.update_orbital_calls.append(
            {"norad_ids": norad_ids, "epoch": epoch}
        )

    def update_satellites_from_celestrak(self) -> None:
        if self.raise_celestrak is not None:
            raise self.raise_celestrak

        self.celestrak_calls += 1

    def verify(self, epoch: datetime | None = None) -> bool:
        if self.raise_verify is not None:
            raise self.raise_verify

        self.verify_calls.append({"epoch": epoch})
        return self.verify_result

    def reset(self, kind: str = "all") -> None:
        if self.raise_reset is not None:
            raise self.raise_reset

        self.reset_calls.append({"kind": kind})


class _FakeLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, Any]]] = []
        self.warning_calls: list[tuple[str, dict[str, Any]]] = []
        self.exception_calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.info_calls.append((event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self.warning_calls.append((event, fields))

    def exception(self, event: str, **fields: Any) -> None:
        self.exception_calls.append((event, fields))


def _sat(norad_id: int) -> Any:
    return SimpleNamespace(norad_id=norad_id)


def test_get_all_returns_repository_satellites() -> None:
    repo = _FakeRepository()
    repo.query_sats_result = [_sat(1), _sat(2)]

    service = SatelliteService(repo)

    result = service.get_all()

    assert result == repo.query_sats_result
    assert repo.query_sats_calls == [
        {"norad_ids": None, "epoch": None}
    ]


def test_get_by_ids_forwards_ids_to_repository() -> None:
    repo = _FakeRepository()
    repo.query_sats_result = [_sat(11)]

    service = SatelliteService(repo)

    result = service.get_by_ids([11, 22])

    assert result == repo.query_sats_result
    assert repo.query_sats_calls == [
        {"norad_ids": [11, 22], "epoch": None}
    ]


def test_get_by_constellations_uses_existing_catalog() -> None:
    repo = _FakeRepository()
    first = [_sat(10), _sat(20)]
    refreshed = [_sat(10), _sat(20)]
    repo.constellation_results = [first, refreshed]
    repo.verify_result = True

    service = SatelliteService(repo)

    result = service.get_by_constellations(["GPS"])

    assert result == refreshed
    assert repo.celestrak_calls == 0
    assert len(repo.query_constellation_calls) == 2
    assert repo.verify_calls == [{"epoch": None}]
    assert repo.update_orbital_calls == []


def test_get_by_constellations_refreshes_empty_catalog() -> None:
    repo = _FakeRepository()
    populated = [_sat(30)]
    refreshed = [_sat(30)]
    repo.constellation_results = [[], populated, refreshed]
    repo.verify_result = True

    service = SatelliteService(repo)

    result = service.get_by_constellations(["GALILEO"])

    assert result == refreshed
    assert repo.celestrak_calls == 1
    assert len(repo.query_constellation_calls) == 3


def test_get_by_constellations_returns_empty_after_failed_refresh() -> None:
    repo = _FakeRepository()
    repo.constellation_results = [[], []]

    logger = _FakeLogger()
    service = SatelliteService(repo, logger=logger)

    result = service.get_by_constellations(["GLONASS"])

    assert result == []
    assert repo.celestrak_calls == 1
    assert repo.update_orbital_calls == []
    assert logger.warning_calls
    assert logger.warning_calls[-1][0] == (
        "satellite_get_by_constellations_catalog_empty"
    )


def test_update_orbital_elements_skips_when_data_is_recent() -> None:
    repo = _FakeRepository()
    repo.satellites = [_sat(1), _sat(2)]
    repo.verify_result = True

    service = SatelliteService(repo)

    service.update_orbital_elements()

    assert repo.verify_calls == [{"epoch": None}]
    assert repo.update_orbital_calls == []


def test_update_orbital_elements_updates_when_data_is_stale() -> None:
    repo = _FakeRepository()
    repo.satellites = [_sat(1), _sat(2)]
    repo.verify_result = False

    service = SatelliteService(repo)

    service.update_orbital_elements()

    assert repo.update_orbital_calls == [
        {"norad_ids": [1, 2], "epoch": None}
    ]


def test_update_orbital_elements_force_bypasses_verify() -> None:
    repo = _FakeRepository()
    repo.satellites = [_sat(7)]

    service = SatelliteService(repo)

    service.update_orbital_elements(force=True)

    assert repo.verify_calls == []
    assert repo.update_orbital_calls == [
        {"norad_ids": [7], "epoch": None}
    ]


def test_update_orbital_elements_passes_epoch() -> None:
    repo = _FakeRepository()
    repo.satellites = [_sat(99)]
    repo.verify_result = False

    service = SatelliteService(repo)
    epoch = datetime(2026, 7, 26, 12, 0)

    service.update_orbital_elements(epoch=epoch)

    assert repo.verify_calls == [{"epoch": epoch}]
    assert repo.update_orbital_calls == [
        {"norad_ids": [99], "epoch": epoch}
    ]


def test_update_orbital_elements_uses_none_when_no_satellites_loaded() -> None:
    repo = _FakeRepository()
    repo.verify_result = False

    service = SatelliteService(repo)

    service.update_orbital_elements()

    assert repo.update_orbital_calls == [
        {"norad_ids": None, "epoch": None}
    ]


def test_update_from_celestrak_delegates_to_repository() -> None:
    repo = _FakeRepository()
    service = SatelliteService(repo)

    service.update_from_celestrak()

    assert repo.celestrak_calls == 1


def test_reset_delegates_to_repository() -> None:
    repo = _FakeRepository()
    service = SatelliteService(repo)

    service.reset()

    assert repo.reset_calls == [{"kind": "all"}]


def test_get_all_logs_and_reraises_repository_error() -> None:
    repo = _FakeRepository()
    repo.raise_query_sats = RuntimeError("boom")
    logger = _FakeLogger()

    service = SatelliteService(repo, logger=logger)

    with pytest.raises(RuntimeError, match="boom"):
        service.get_all()

    assert logger.exception_calls[-1][0] == "satellite_get_all_failed"


def test_update_from_celestrak_logs_and_reraises_error() -> None:
    repo = _FakeRepository()
    repo.raise_celestrak = RuntimeError("offline")
    logger = _FakeLogger()

    service = SatelliteService(repo, logger=logger)

    with pytest.raises(RuntimeError, match="offline"):
        service.update_from_celestrak()

    assert logger.exception_calls[-1][0] == (
        "satellite_update_from_celestrak_failed"
    )


def test_reset_logs_and_reraises_error() -> None:
    repo = _FakeRepository()
    repo.raise_reset = RuntimeError("reset failed")
    logger = _FakeLogger()

    service = SatelliteService(repo, logger=logger)

    with pytest.raises(RuntimeError, match="reset failed"):
        service.reset()

    assert logger.exception_calls[-1][0] == "satellite_reset_failed"
