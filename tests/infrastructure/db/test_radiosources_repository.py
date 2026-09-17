from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from radioskysims_sources.domain.radiosources.models import (
    RadioSource,
    SpectralModel,
)
from radioskysims_sources.infrastructure.db.radiosources_db_model import (
    RadioSourceDB,
)
from radioskysims_sources.infrastructure.db.radiosources_repository import (
    InMemoryRadioSourcesRepository,
    PostgresRadioSourcesRepository,
)


def _source(
    source_id: str,
    *,
    ra_deg: float = 10.0,
    dec_deg: float = -5.0,
    s0_jy: float = 2.0,
    nu0_mhz: float = 1400.0,
    alpha: float = 0.7,
) -> RadioSource:
    return RadioSource(
        id=source_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        spectral_model=SpectralModel(
            nu0_mhz=nu0_mhz,
            s0_jy=s0_jy,
            alpha=alpha,
        ),
    )


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[RadioSourceDB.__table__],
    )
    yield engine
    engine.dispose()


def test_db_model_roundtrip_preserves_domain_values() -> None:
    source = _source(
        "SRC-1",
        ra_deg=123.4,
        dec_deg=-42.5,
        s0_jy=7.5,
        nu0_mhz=843.0,
        alpha=0.82,
    )

    db_source = RadioSourceDB.from_domain(source)
    restored = db_source.to_domain()

    assert db_source.id == source.id
    assert db_source.ra_deg == source.ra_deg
    assert db_source.dec_deg == source.dec_deg
    assert db_source.nu0_mhz == source.spectral_model.nu0_mhz
    assert db_source.s0_jy == source.spectral_model.s0_jy
    assert db_source.alpha == source.spectral_model.alpha

    assert restored == source


def test_in_memory_save_sources_adds_sources() -> None:
    repository = InMemoryRadioSourcesRepository()

    first = _source("SRC-1")
    second = _source("SRC-2", ra_deg=20.0)

    repository.save_sources([first, second])

    assert {source.id for source in repository._sources} == {
        "SRC-1",
        "SRC-2",
    }


def test_in_memory_save_sources_updates_existing_source_by_id() -> None:
    repository = InMemoryRadioSourcesRepository()

    repository.save_sources(
        [
            _source(
                "SRC-1",
                ra_deg=10.0,
                s0_jy=1.0,
            )
        ]
    )

    updated = _source(
        "SRC-1",
        ra_deg=30.0,
        s0_jy=9.0,
    )

    repository.save_sources([updated])

    assert len(repository._sources) == 1
    assert repository._sources[0] == updated


def test_in_memory_region_query_filters_by_angular_distance() -> None:
    repository = InMemoryRadioSourcesRepository()

    near = _source(
        "NEAR",
        ra_deg=10.2,
        dec_deg=-5.0,
    )
    far = _source(
        "FAR",
        ra_deg=40.0,
        dec_deg=-5.0,
    )

    repository.save_sources([near, far])

    result = repository.list_sources_in_region(
        ra_deg=10.0,
        dec_deg=-5.0,
        radius_deg=1.0,
    )

    assert result == [near]


@pytest.mark.parametrize(
    ("flux_jy", "expected"),
    [
        (2.1, True),
        (2.0, False),
        (1.9, False),
    ],
)
def test_in_memory_region_query_uses_strict_minimum_flux_cut(
    flux_jy: float,
    expected: bool,
) -> None:
    repository = InMemoryRadioSourcesRepository()

    source = _source(
        "SRC",
        ra_deg=10.0,
        dec_deg=-5.0,
        s0_jy=flux_jy,
    )
    repository.save_sources([source])

    result = repository.list_sources_in_region(
        ra_deg=10.0,
        dec_deg=-5.0,
        radius_deg=1.0,
        min_flux_jy=2.0,
    )

    assert (source in result) is expected


def test_postgres_save_empty_list_does_not_open_session() -> None:
    def session_factory():
        message = "session_factory não deveria ser chamado"
        raise AssertionError(message)

    repository = PostgresRadioSourcesRepository(
        session_factory=session_factory,
    )

    repository.save_sources([])


def test_postgres_save_sources_inserts_new_rows(
    sqlite_engine,
) -> None:
    repository = PostgresRadioSourcesRepository(
        session_factory=lambda: Session(sqlite_engine),
    )

    first = _source("SRC-1")
    second = _source(
        "SRC-2",
        ra_deg=20.0,
        s0_jy=4.0,
    )

    repository.save_sources([first, second])

    with Session(sqlite_engine) as session:
        rows = session.exec(select(RadioSourceDB)).all()

    assert len(rows) == 2
    assert {row.id for row in rows} == {
        "SRC-1",
        "SRC-2",
    }


def test_postgres_save_sources_updates_existing_row(
    sqlite_engine,
) -> None:
    repository = PostgresRadioSourcesRepository(
        session_factory=lambda: Session(sqlite_engine),
    )

    repository.save_sources(
        [
            _source(
                "SRC-1",
                ra_deg=10.0,
                dec_deg=-5.0,
                s0_jy=1.0,
                nu0_mhz=1400.0,
                alpha=0.7,
            )
        ]
    )

    updated = _source(
        "SRC-1",
        ra_deg=120.0,
        dec_deg=30.0,
        s0_jy=8.0,
        nu0_mhz=843.0,
        alpha=0.95,
    )

    repository.save_sources([updated])

    with Session(sqlite_engine) as session:
        rows = session.exec(select(RadioSourceDB)).all()

    assert len(rows) == 1

    row = rows[0]

    assert row.id == "SRC-1"
    assert row.ra_deg == 120.0
    assert row.dec_deg == 30.0
    assert row.s0_jy == 8.0
    assert row.nu0_mhz == 843.0
    assert row.alpha == 0.95


def test_postgres_region_query_returns_empty_list_for_empty_database(
    sqlite_engine,
) -> None:
    repository = PostgresRadioSourcesRepository(
        session_factory=lambda: Session(sqlite_engine),
    )

    result = repository.list_sources_in_region(
        ra_deg=10.0,
        dec_deg=-5.0,
        radius_deg=1.0,
    )

    assert result == []


def test_postgres_region_query_filters_by_angular_distance(
    sqlite_engine,
) -> None:
    repository = PostgresRadioSourcesRepository(
        session_factory=lambda: Session(sqlite_engine),
    )

    near = _source(
        "NEAR",
        ra_deg=10.2,
        dec_deg=-5.0,
    )
    far = _source(
        "FAR",
        ra_deg=50.0,
        dec_deg=-5.0,
    )

    repository.save_sources([near, far])

    result = repository.list_sources_in_region(
        ra_deg=10.0,
        dec_deg=-5.0,
        radius_deg=1.0,
    )

    assert result == [near]


@pytest.mark.parametrize(
    ("flux_jy", "expected"),
    [
        (2.1, True),
        (2.0, False),
        (1.9, False),
    ],
)
def test_postgres_region_query_uses_strict_minimum_flux_cut(
    sqlite_engine,
    flux_jy: float,
    expected: bool,
) -> None:
    repository = PostgresRadioSourcesRepository(
        session_factory=lambda: Session(sqlite_engine),
    )

    source = _source(
        "SRC",
        ra_deg=10.0,
        dec_deg=-5.0,
        s0_jy=flux_jy,
    )
    repository.save_sources([source])

    result = repository.list_sources_in_region(
        ra_deg=10.0,
        dec_deg=-5.0,
        radius_deg=1.0,
        min_flux_jy=2.0,
    )

    assert (source in result) is expected
