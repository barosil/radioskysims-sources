from __future__ import annotations

from types import SimpleNamespace

import pytest
from radioskysims_sources.infrastructure.db.satellite.models import (
    SatID,
    SatSignalLink,
    Signal,
)
from radioskysims_sources.infrastructure.db.signal.repository import SignalRepository
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


class _Database:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(
            self.engine,
            tables=[SatID.__table__, Signal.__table__, SatSignalLink.__table__],
        )

    def session(self):
        return Session(self.engine)

    def close(self):
        self.engine.dispose()


@pytest.fixture
def database():
    value = _Database()
    with value.session() as session:
        session.add_all(
            [
                SatID(norad_id=1, name="GPS-1", constellation="GPS"),
                SatID(norad_id=2, name="GAL-1", constellation="GALILEO"),
            ]
        )
        session.commit()
    yield value
    value.close()


def _signals_dict():
    return {
        "constellations": [
            {
                "constellation": "GPS",
                "constellation_std": "GPS",
                "eirp_dbw": 24.6,
                "signals": [
                    {"frequency": "1575.42 MHz", "signal_name": "L1", "modulation": "BPSK"},
                    {"frequency": "bad MHz", "signal_name": "BAD", "modulation": "QPSK"},
                ],
            },
            {
                "constellation": "GSAT",
                "eirp_dbw": 25.8,
                "signals": [
                    {"frequency": "1207.14 MHz", "signal_name": "E5", "modulation": "QPSK"},
                ],
            },
        ]
    }


def test_repository_uses_injected_dict_and_builds_eirp_map():
    repository = SignalRepository(signals_dict=_signals_dict())

    assert repository.get_eirp_map() == {
        "GPS": 24.6,
        "GSAT": 25.8,
    }


def test_repository_builds_per_satellite_eirp_map():
    """Seção opcional 'satellites' permite EIRP real por NORAD ID (item 2)."""
    signals = _signals_dict()
    signals["satellites"] = [
        {"norad_id": 1, "constellation": "GPS", "eirp_dbw": 25.3},
        {"norad_id": "2", "eirp_dbw": 22.0},
        {"constellation": "GPS"},  # sem eirp → ignorado
        {"norad_id": "bad", "eirp_dbw": 1.0},  # norad inválido → ignorado
        {"norad_id": 9},  # sem eirp → ignorado
    ]
    repository = SignalRepository(signals_dict=signals)

    assert repository.get_eirp_by_satellite() == {1: 25.3, 2: 22.0}


def test_per_satellite_eirp_empty_for_legacy_format():
    """Formato legado (só constelações) → sem EIRP individual."""
    repository = SignalRepository(signals_dict=_signals_dict())

    assert repository.get_eirp_by_satellite() == {}


def test_json_loader_handles_missing_and_supported_formats(tmp_path):
    missing = SignalRepository(signals_json=str(tmp_path / "missing.json"))
    assert missing.signals_dict == {}

    path = tmp_path / "signals.json"
    path.write_text('[{"constellation": "GPS", "signals": []}]', encoding="utf-8")
    repository = SignalRepository(signals_json=str(path))
    assert repository.signals_dict["constellations"][0]["constellation"] == "GPS"


def test_initialize_loads_associations_and_is_idempotent(database):
    repository = SignalRepository(engine=database, signals_dict=_signals_dict())

    repository.initialize_signals()
    repository.initialize_signals()

    assert len(repository.load_signals()) == 2
    assert len(repository.get_signals_by_constellation("GPS")) == 1
    assert [
        signal.signal_name
        for signal in repository.get_satellites_signals([SimpleNamespace(norad_id=1)])
    ] == ["L1"]


def test_signal_queries_return_empty_for_unknown_ids(database):
    repository = SignalRepository(engine=database, signals_dict={"constellations": []})

    assert repository.get_signals_by_norad_ids([999]) == {999: []}
    assert repository.get_satellites_signals([]) == []


def test_initialize_skips_unknown_constellation_and_invalid_frequency(database):
    repository = SignalRepository(
        engine=database,
        signals_dict={
            "constellations": [
                {
                    "constellation": "UNKNOWN",
                    "signals": [{"frequency": "100 MHz", "signal_name": "X", "modulation": "M"}],
                },
                {
                    "constellation": "GPS",
                    "signals": [{"frequency": "invalid", "signal_name": "Y", "modulation": "M"}],
                },
            ]
        },
    )

    repository.initialize_signals()

    assert repository.load_signals() == []


def test_set_satellite_signals_updates_existing_satellite(database):
    repository = SignalRepository(engine=database, signals_dict={"constellations": []})
    signal = Signal(signal_name="L1", modulation="BPSK", frequency=1575.42)
    with database.session() as session:
        session.add(signal)
        session.commit()
        persisted_signal = session.get(Signal, signal.id)

    repository.set_satellite_signals(
        SimpleNamespace(id=1),
        [persisted_signal],
    )

    assert repository.get_signals_by_norad_ids([1])[1]
