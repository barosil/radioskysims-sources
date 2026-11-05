from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from astropy.time import Time
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from radioskysims_sources.infrastructure.db.satellite._query import QueryMixin
from radioskysims_sources.infrastructure.db.satellite.migration import (
    backfill_prn,
    ensure_prn_column,
)
from radioskysims_sources.infrastructure.db.satellite.models import (
    OrbitalElement,
    SatID,
    SatSignalLink,
    Signal,
)


EPOCH = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


class _Database:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(
            self.engine,
            tables=[
                SatID.__table__,
                OrbitalElement.__table__,
                Signal.__table__,
                SatSignalLink.__table__,
            ],
        )

    def session(self):
        return Session(self.engine)

    def close(self):
        self.engine.dispose()


class _Repository(QueryMixin):
    def __init__(self, database):
        self.engine = database
        self.tr = None
        self.epoch = None
        self.norad_ids = []
        self.sats_df = None
        self.satellites = []

    def update_orbital_elements_sync(self, **_kwargs):
        self.updated = True


def _satellite(norad_id, constellation="GPS", elements=None):
    sat = SatID(norad_id=norad_id, name=f"SAT-{norad_id}", constellation=constellation)
    sat.elements = elements or []
    return sat


def _element(norad_id, epoch):
    return OrbitalElement(
        norad_id=norad_id,
        epoch=epoch,
        tle_line1="line1",
        tle_line2="line2",
    )


@pytest.fixture
def repository():
    database = _Database()
    with database.session() as session:
        session.add_all(
            [
                _satellite(
                    1,
                    elements=[
                        _element(1, EPOCH - timedelta(days=2)),
                        _element(1, EPOCH + timedelta(days=1)),
                    ],
                ),
                _satellite(2, "GALILEO", [_element(2, EPOCH)]),
            ]
        )
        session.commit()
    value = _Repository(database)
    yield value
    database.close()


def test_load_summary_handles_empty_and_populated_database(repository):
    empty_database = _Database()
    empty_repository = _Repository(empty_database)
    empty_repository._load_sats_summary()
    assert list(empty_repository.sats_df.columns) == [
        "norad_id",
        "name",
        "constellation",
        "type",
        "num_elements",
        "num_signals",
    ]
    empty_database.close()

    repository.sats_df = None
    with repository.engine.session() as session:
        session.add(_satellite(3, "GPS", [_element(3, EPOCH)]))
        session.commit()
    repository._load_sats_summary()

    assert set(repository.sats_df["norad_id"]) == {1, 2, 3}
    assert repository.norad_ids == [1, 2, 3]


def test_epoch_from_time_range_uses_midpoint(repository):
    repository.tr = SimpleNamespace(
        to_tuple=lambda: (Time(2460000.0, format="jd"), Time(2460002.0, format="jd"))
    )

    result = repository._epoch_from_tr()

    assert result is not None
    assert isinstance(result, datetime)


def test_query_satids_selects_closest_element_and_rejects_bad_epoch(repository):
    with pytest.raises(ValueError, match="datetime"):
        repository._query_satids([1], epoch="bad")

    result = repository._query_satids([1], epoch=EPOCH)

    assert len(result) == 1
    assert result[0].elements[0].epoch == (EPOCH + timedelta(days=1)).replace(tzinfo=None)


def test_query_satids_uses_state_and_empty_fallback(repository):
    repository.norad_ids = [2]
    assert [sat.norad_id for sat in repository._query_satids(None)] == [2]
    repository.norad_ids = []
    repository.sats_df = pd.DataFrame({"norad_id": [1]})
    assert [sat.norad_id for sat in repository._query_satids(None)] == [1]
    repository.sats_df = None
    assert repository._query_satids(None) == []


def test_query_and_load_methods_cover_domain_and_append_paths(repository, monkeypatch):
    assert [sat.norad_id for sat in repository.query_sats([1])] == [1]
    assert repository.query_sats([2, 999])
    assert repository.query_constellation([]) == []

    monkeypatch.setattr(repository, "_query_satids", lambda *_args, **_kwargs: ["loaded"])
    assert repository.load_elements([1], append=False) == ["loaded"]
    assert repository.satellites == ["loaded"]
    assert repository.load_elements([1], append=True) == ["loaded"]
    assert repository.satellites == ["loaded", "loaded"]
    assert repository.load_elements([]) is None
    assert repository.load_elements_from_constellation([]) is None


def test_ensure_elements_refreshes_missing_and_keeps_existing(repository, monkeypatch):
    existing = _satellite(1, elements=[_element(1, EPOCH)])
    missing = _satellite(2, elements=[])
    refreshed = _satellite(2, elements=[_element(2, EPOCH)])
    repository.update_orbital_elements_sync = lambda **_kwargs: None
    monkeypatch.setattr(repository, "_query_satids", lambda *_args, **_kwargs: [refreshed])

    result = repository._ensure_elements([existing, missing], epoch=EPOCH)

    assert result == [existing, refreshed]


def test_ensure_elements_returns_original_on_update_failure(repository):
    missing = _satellite(2, elements=[])

    def fail(**_kwargs):
        message = "offline"
        raise RuntimeError(message)

    repository.update_orbital_elements_sync = fail

    assert repository._ensure_elements([missing]) == [missing]


def test_migration_backfills_prn_and_requires_database_session(repository):
    with repository.engine.session() as session:
        session.add_all(
            [
                SatID(norad_id=10, name="GPS BIIR-5 (PRN 22)", constellation="GPS"),
                SatID(norad_id=11, name="UNKNOWN", constellation="GPS", prn=7),
            ]
        )
        session.commit()

    assert backfill_prn(repository.engine) == 1
    assert backfill_prn(repository.engine) == 0
    with pytest.raises(TypeError, match=r"\.session"):
        backfill_prn(object())


def test_ensure_prn_column_reports_sqlite_ddl_failure(repository):
    assert ensure_prn_column(repository.engine) is False


def test_to_domain_rejects_missing_elements(repository):
    with pytest.raises(ValueError, match="no orbital elements"):
        repository._to_domain_satellite(_satellite(99))
