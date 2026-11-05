from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from radioskysims_sources.domain.localsources import SolarFlux
from radioskysims_sources.infrastructure.db.solar_flux_db_model import SolarFluxDB
from radioskysims_sources.infrastructure.db.solar_flux_repository import (
    SolarFluxRepository,
)


class _TestDatabase:
    """Adaptador SQLite mínimo compatível com SolarFluxRepository."""

    def __init__(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

    def session(self) -> Session:
        return Session(self.engine)

    def close(self) -> None:
        self.engine.dispose()


@pytest.fixture
def database():
    db = _TestDatabase()
    yield db
    db.close()


@pytest.fixture
def scipy_netcdf(monkeypatch) -> None:
    """Força backend SciPy nos testes NetCDF."""
    original_to_netcdf = xr.Dataset.to_netcdf
    original_load_dataset = xr.load_dataset

    def to_netcdf_with_scipy(self, path, *args, **kwargs):
        kwargs.setdefault("engine", "scipy")
        return original_to_netcdf(self, path, *args, **kwargs)

    def load_dataset_with_scipy(*args, **kwargs):
        kwargs.setdefault("engine", "scipy")
        return original_load_dataset(*args, **kwargs)

    monkeypatch.setattr(
        xr.Dataset,
        "to_netcdf",
        to_netcdf_with_scipy,
    )
    monkeypatch.setattr(
        xr,
        "load_dataset",
        load_dataset_with_scipy,
    )


def _solar_flux(
    *,
    source_url: str | None = "https://example.test/solar.SRD",
) -> SolarFlux:
    dataset = xr.Dataset(
        {
            "flux_jy": (
                ("time", "frequency_mhz"),
                np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                ),
            )
        },
        coords={
            "time": np.array(
                [
                    "2026-07-26T11:00:00",
                    "2026-07-26T13:00:00",
                ],
                dtype="datetime64[ns]",
            ),
            "frequency_mhz": [245.0, 410.0],
        },
    )

    return SolarFlux(
        start_utc=datetime(2026, 7, 26, 11, 0),
        end_utc=datetime(2026, 7, 26, 13, 0),
        dataset=dataset,
        source_url=source_url,
    )


def test_repository_creates_cache_directory(tmp_path) -> None:
    cache_dir = tmp_path / "solar"

    repository = SolarFluxRepository(
        engine=None,
        cache_dir=cache_dir,
    )

    assert repository.cache_dir == cache_dir.resolve()
    assert repository.cache_dir.is_dir()


def test_load_interval_without_engine_returns_none(tmp_path) -> None:
    repository = SolarFluxRepository(
        engine=None,
        cache_dir=tmp_path,
    )

    result = repository.load_interval(
        start_utc=datetime(2026, 7, 26, 11, 0),
        end_utc=datetime(2026, 7, 26, 13, 0),
    )

    assert result is None


def test_save_without_engine_writes_expected_netcdf_path(
    tmp_path,
    monkeypatch,
) -> None:
    repository = SolarFluxRepository(
        engine=None,
        cache_dir=tmp_path,
    )
    solar_flux = _solar_flux()

    written_paths: list[Path] = []

    def fake_to_netcdf(_dataset, path) -> None:
        file_path = Path(path)
        written_paths.append(file_path)
        file_path.touch()

    monkeypatch.setattr(
        xr.Dataset,
        "to_netcdf",
        fake_to_netcdf,
    )

    result = repository.save(solar_flux)

    expected = tmp_path / "solar_flux_20260726T110000_20260726T130000.nc"

    assert result is solar_flux
    assert written_paths == [expected]
    assert expected.is_file()


def test_save_without_engine_does_not_require_database(
    tmp_path,
    monkeypatch,
) -> None:
    repository = SolarFluxRepository(
        engine=None,
        cache_dir=tmp_path,
    )
    solar_flux = _solar_flux()

    calls = 0

    def fake_to_netcdf(_dataset, _path) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        xr.Dataset,
        "to_netcdf",
        fake_to_netcdf,
    )

    result = repository.save(solar_flux)

    assert result is solar_flux
    assert calls == 1


def test_repository_with_engine_creates_solar_flux_table(
    tmp_path,
    database,
) -> None:
    SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )

    with database.session() as session:
        result = session.exec(select(SolarFluxDB)).all()

    assert result == []


def test_load_interval_with_database_without_record_returns_none(
    tmp_path,
    database,
) -> None:
    repository = SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )

    result = repository.load_interval(
        start_utc=datetime(2026, 7, 26, 11, 0),
        end_utc=datetime(2026, 7, 26, 13, 0),
    )

    assert result is None


@pytest.mark.usefixtures("scipy_netcdf")
def test_save_with_database_registers_metadata(
    tmp_path,
    database,
) -> None:
    repository = SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )
    solar_flux = _solar_flux()

    result = repository.save(solar_flux)

    expected_filename = "solar_flux_20260726T110000_20260726T130000.nc"
    expected_file = tmp_path / expected_filename

    assert result is solar_flux
    assert expected_file.is_file()

    with database.session() as session:
        rows = session.exec(select(SolarFluxDB)).all()

    assert len(rows) == 1
    assert rows[0].filename == expected_filename
    assert rows[0].start_utc == solar_flux.start_utc
    assert rows[0].end_utc == solar_flux.end_utc
    assert rows[0].source_url == solar_flux.source_url


@pytest.mark.usefixtures("scipy_netcdf")
def test_save_and_load_interval_roundtrip(
    tmp_path,
    database,
) -> None:
    repository = SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )
    solar_flux = _solar_flux()

    repository.save(solar_flux)

    loaded = repository.load_interval(
        start_utc=datetime(2026, 7, 26, 11, 30),
        end_utc=datetime(2026, 7, 26, 12, 30),
    )

    assert loaded is not None
    assert loaded.start_utc == solar_flux.start_utc
    assert loaded.end_utc == solar_flux.end_utc
    assert loaded.source_url == solar_flux.source_url

    xr.testing.assert_equal(
        loaded.dataset,
        solar_flux.dataset,
    )


def test_load_interval_returns_none_when_registered_file_is_missing(
    tmp_path,
    database,
    caplog,
) -> None:
    repository = SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )

    row = SolarFluxDB(
        filename="missing.nc",
        start_utc=datetime(2026, 7, 26, 10, 0),
        end_utc=datetime(2026, 7, 26, 14, 0),
        source_url="https://example.test/missing.SRD",
    )

    with database.session() as session:
        session.add(row)
        session.commit()

    result = repository.load_interval(
        start_utc=datetime(2026, 7, 26, 11, 0),
        end_utc=datetime(2026, 7, 26, 13, 0),
    )

    assert result is None
    assert "Arquivo SolarFlux registrado não encontrado" in caplog.text


@pytest.mark.usefixtures("scipy_netcdf")
def test_save_existing_filename_updates_database_record(
    tmp_path,
    database,
) -> None:
    repository = SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )

    first = _solar_flux(
        source_url="https://example.test/original.SRD",
    )
    second = _solar_flux(
        source_url="https://example.test/updated.SRD",
    )

    repository.save(first)
    repository.save(second)

    with database.session() as session:
        rows = session.exec(select(SolarFluxDB)).all()

    assert len(rows) == 1
    assert rows[0].source_url == "https://example.test/updated.SRD"


def test_repository_resolves_relative_cache_directory(
    monkeypatch,
    tmp_path,
) -> None:
    from radioskysims_sources.infrastructure.db import solar_flux_repository

    monkeypatch.setattr(
        solar_flux_repository,
        "ROOT",
        tmp_path,
    )

    repository = SolarFluxRepository(
        engine=None,
        cache_dir="relative/solar",
    )

    assert repository.cache_dir == (tmp_path / "relative" / "solar").resolve()

    assert repository.cache_dir.is_dir()


def test_load_interval_reconstructs_solar_flux_from_registered_cache(
    tmp_path,
    database,
    monkeypatch,
) -> None:
    repository = SolarFluxRepository(
        engine=database,
        cache_dir=tmp_path,
    )

    expected_dataset = _solar_flux().dataset

    filename = "cached.nc"
    file_path = tmp_path / filename
    file_path.touch()

    expected_start = datetime(2026, 7, 26, 10, 0)
    expected_end = datetime(2026, 7, 26, 14, 0)
    expected_source_url = "https://example.test/cached.SRD"

    row = SolarFluxDB(
        filename=filename,
        start_utc=expected_start,
        end_utc=expected_end,
        source_url=expected_source_url,
    )

    with database.session() as session:
        session.add(row)
        session.commit()

    monkeypatch.setattr(
        xr,
        "load_dataset",
        lambda _path: expected_dataset,
    )

    result = repository.load_interval(
        start_utc=datetime(2026, 7, 26, 11, 0),
        end_utc=datetime(2026, 7, 26, 13, 0),
    )

    assert result is not None
    assert result.start_utc == expected_start
    assert result.end_utc == expected_end
    assert result.source_url == expected_source_url
    assert result.dataset is expected_dataset
