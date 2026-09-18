from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from radioskysims_sources.infrastructure.db.satellite.models import (
    OrbitalElement,
    SatID,
)
from radioskysims_sources.infrastructure.db.satellite.repository_adapter import (
    SqlAlchemySatelliteRepository,
    _select_orbital_element,
)


EPOCH = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[SatID.__table__, OrbitalElement.__table__],
    )
    with Session(engine) as value:
        yield value
    engine.dispose()


def _element(norad_id: int, epoch: datetime) -> OrbitalElement:
    return OrbitalElement(
        norad_id=norad_id,
        epoch=epoch,
        tle_line1=f"1 {norad_id}",
        tle_line2=f"2 {norad_id}",
    )


def _satellite(
    norad_id: int,
    *,
    constellation: str = "GPS",
    elements: list[OrbitalElement] | None = None,
) -> SatID:
    satellite = SatID(
        norad_id=norad_id,
        name=f"SAT-{norad_id}",
        constellation=constellation,
        prn=22 if constellation == "GPS" else None,
    )
    satellite.elements = elements or []
    return satellite


def _populate(session: Session) -> None:
    session.add(
        _satellite(
            1,
            elements=[
                _element(1, EPOCH - timedelta(days=2)),
                _element(1, EPOCH + timedelta(days=1)),
            ],
        )
    )
    session.add(
        _satellite(
            2,
            constellation="GALILEO",
            elements=[_element(2, EPOCH)],
        )
    )
    session.commit()


def test_repository_requires_sqlmodel_session():
    with pytest.raises(TypeError, match="Expected Session"):
        SqlAlchemySatelliteRepository(object())


@pytest.mark.parametrize("method", ["query_sats", "query_constellation"])
def test_query_methods_reject_empty_filters(session, method):
    repository = SqlAlchemySatelliteRepository(session)

    with pytest.raises(ValueError, match="cannot be empty"):
        getattr(repository, method)([])


def test_query_sats_maps_rows_and_selects_element_near_epoch(session):
    _populate(session)
    repository = SqlAlchemySatelliteRepository(session)

    result = repository.query_sats([1, 999], epoch=EPOCH)

    assert len(result) == 1
    assert result[0].norad_id == 1
    assert result[0].name == "SAT-1"
    assert result[0].prn == 22
    assert result[0].element.epoch == EPOCH + timedelta(days=1)


def test_query_constellation_uses_latest_element_when_epoch_is_omitted(session):
    _populate(session)
    repository = SqlAlchemySatelliteRepository(session)

    result = repository.query_constellation(["GPS"])

    assert [sat.norad_id for sat in result] == [1]
    assert result[0].element.epoch == EPOCH + timedelta(days=1)


def test_legacy_wrappers_delegate_to_canonical_queries(session, monkeypatch):
    repository = SqlAlchemySatelliteRepository(session)
    expected = [object()]
    query_sats = monkeypatch.setattr(
        repository,
        "query_sats",
        lambda **kwargs: (kwargs, expected)[1],
    )
    query_constellation = monkeypatch.setattr(
        repository,
        "query_constellation",
        lambda **kwargs: (kwargs, expected)[1],
    )

    assert repository.get_by_norad_ids([1], EPOCH) is expected
    assert repository.get_by_constellations(["GPS"], EPOCH) is expected
    assert query_sats is None
    assert query_constellation is None


def test_sync_and_celestrak_methods_are_explicitly_unsupported(session):
    repository = SqlAlchemySatelliteRepository(session)

    with pytest.raises(NotImplementedError, match="não implementa sincronização"):
        repository.update_orbital_elements_sync([1], EPOCH)
    with pytest.raises(NotImplementedError, match="não implementa ingestão"):
        repository.update_satellites_from_celestrak()


def test_verify_reports_empty_missing_elements_and_available_elements(session):
    repository = SqlAlchemySatelliteRepository(session)

    assert repository.verify() is False
    session.add(_satellite(10, elements=[]))
    session.commit()
    assert repository.verify(EPOCH) is False
    session.add(_satellite(11, elements=[_element(11, EPOCH)]))
    session.commit()
    assert repository.verify() is True


def test_verify_returns_false_when_query_fails(session, monkeypatch):
    repository = SqlAlchemySatelliteRepository(session)

    def fail(_statement):
        message = "database unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(repository.session, "exec", fail)

    assert repository.verify() is False


def test_reset_is_stateless(session):
    repository = SqlAlchemySatelliteRepository(session)

    assert repository.reset("satellites") is None


def test_select_orbital_element_handles_empty_and_latest():
    satellite = _satellite(1, elements=[])
    assert _select_orbital_element(satellite) is None

    older = _element(1, EPOCH - timedelta(days=3))
    latest = _element(1, EPOCH + timedelta(days=2))
    satellite.elements = [older, latest]

    assert _select_orbital_element(satellite) is latest
    assert _select_orbital_element(satellite, EPOCH - timedelta(days=1)) is older


def test_to_domain_rejects_satellite_without_orbital_elements():
    satellite = _satellite(99, elements=[])

    with pytest.raises(ValueError, match="has no orbital elements"):
        SqlAlchemySatelliteRepository._to_domain(satellite)
