from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from radioskysims_sources.infrastructure.db.satellite._convert import (
    _safe_float,
    _safe_int,
)
from radioskysims_sources.infrastructure.db.satellite._process import ProcessMixin
from radioskysims_sources.infrastructure.db.satellite._signals import SignalsMixin
from radioskysims_sources.infrastructure.db.satellite.models import (
    OrbitalElement,
    SatID,
    SatSignalLink,
    Signal,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("2.5", 2.5), (np.nan, None), ("bad", None)],
)
def test_safe_float_handles_valid_nan_and_invalid_values(value, expected):
    assert _safe_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("2.9", 2), (np.nan, None), ("bad", None)],
)
def test_safe_int_handles_valid_nan_and_invalid_values(value, expected):
    assert _safe_int(value) == expected


class _SignalRepository(SignalsMixin):
    def __init__(self, engine):
        self.engine = engine
        self.sats_df = None
        self.satellites = []


@pytest.fixture
def signal_repository():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(
        engine,
        tables=[SatID.__table__, OrbitalElement.__table__, Signal.__table__, SatSignalLink.__table__],
    )
    with Session(engine) as session:
        session.add_all(
            [
                SatID(norad_id=1, name="GPS-1", constellation="GPS"),
                SatID(norad_id=2, name="GAL-1", constellation="GALILEO"),
            ]
        )
        session.commit()
    engine.session = lambda: Session(engine)
    repository = _SignalRepository(engine)
    yield repository
    engine.dispose()


def _signals_file(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(
        json.dumps(
            [
                {
                    "constellation": "GPS",
                    "signals": [
                        {"frequency": "1575.42 MHz", "modulation": "BPSK", "signal_name": "L1"},
                        {"frequency": "bad", "modulation": "QPSK", "signal_name": "bad"},
                    ],
                },
                {
                    "constellation": "GSAT",
                    "signals": [
                        {"frequency": 1207.14, "modulation": None, "signal_name": "E5"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_signals_json_reader_returns_empty_for_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not json", encoding="utf-8")

    repository = _SignalRepository(None)

    assert repository._read_signals_json(path) == []


def test_make_signals_normalizes_aliases_and_discards_bad_frequency(
    signal_repository,
    tmp_path,
):
    path = _signals_file(tmp_path)
    satellites = [
        SatID(norad_id=1, name="GPS-1", constellation="GPS"),
        SatID(norad_id=2, name="GAL-1", constellation="GALILEO"),
    ]

    norad_ids, signals = signal_repository._make_signals(satellites, path)

    assert norad_ids == [1, 2]
    assert [(signal.signal_name, signal.modulation, signal.frequency) for signal in signals] == [
        ("L1", "BPSK", 1575.42),
        ("E5", "", 1207.14),
    ]


def test_signals_from_json_accepts_ids_and_is_idempotent(signal_repository, tmp_path):
    path = _signals_file(tmp_path)

    signal_repository._signals_from_json([1, 2], path)
    signal_repository._signals_from_json([1, 2], path)

    with Session(signal_repository.engine) as session:
        signals = session.exec(select(Signal)).all()
        links = session.exec(select(SatSignalLink)).all()

    assert len(signals) == 2
    assert len(links) == 2


def test_signals_from_json_returns_without_satellites(signal_repository, tmp_path):
    signal_repository._signals_from_json([], _signals_file(tmp_path))

    with Session(signal_repository.engine) as session:
        assert session.exec(select(Signal)).all() == []


class _ProcessRepository(ProcessMixin):
    def __init__(self):
        self.norad_ids = [1]
        self.constellations = ["GPS"]
        self.satellites = []
        self.epoch = None
        self.failed_sats_df = pd.DataFrame({"id": [1]})
        self.sats_df = pd.DataFrame({"id": [1]})
        self.processed_sats = ["processed"]
        self.loaded = []

    def _query_satids(self, *_args, **_kwargs):
        return []

    def _epoch_from_tr(self):
        return None

    def load_elements(self, norad_ids, **kwargs):
        self.loaded.append(("ids", norad_ids, kwargs))
        return ["by-id"]

    def load_elements_from_constellation(self, constellations, **kwargs):
        self.loaded.append(("constellations", constellations, kwargs))
        return ["by-constellation"]

    def load_signals(self, sats):
        self.loaded.append(("signals", sats, {}))
        return sats


def test_process_mixin_processes_ids_and_constellations():
    repository = _ProcessRepository()

    result = repository._process(epoch="epoch")

    assert result == ["by-id", "by-constellation"]
    assert repository.loaded == [
        ("ids", [1], {"append": True, "epoch": "epoch"}),
        ("constellations", ["GPS"], {"append": True, "epoch": "epoch"}),
        ("signals", ["by-id", "by-constellation"], {}),
    ]


@pytest.mark.parametrize("kind", ["all", "satellites", "norad_ids", "constellations", "selection", "processed_sats"])
def test_reset_clears_requested_state(kind):
    repository = _ProcessRepository()

    result = repository.reset(kind)

    assert result is repository
    if kind == "all":
        assert repository.satellites == []
        assert repository.norad_ids == []
        assert repository.constellations == []
        assert repository.sats_df.empty
    elif kind == "satellites":
        assert repository.satellites == []
    elif kind == "norad_ids":
        assert repository.norad_ids == []
    elif kind == "constellations":
        assert repository.constellations == []
    elif kind == "selection":
        assert repository.failed_sats_df.empty
    else:
        assert repository.processed_sats == []


def _satellite(norad_id, *, elements=True, signals=True, epoch=None):
    return SimpleNamespace(
        norad_id=norad_id,
        elements=[SimpleNamespace(epoch=epoch or datetime.now(timezone.utc))] if elements else [],
        signals=[object()] if signals else [],
    )


def test_verify_with_explicit_satellites_classifies_elements_and_signals():
    repository = _ProcessRepository()
    satellites = [
        _satellite(1),
        _satellite(2, elements=False),
        _satellite(3, signals=False),
    ]

    assert repository.verify(satellites) == ([1], [2], [3])


@pytest.mark.parametrize(
    "satellites",
    [[], [_satellite(1, elements=False)], [_satellite(1, epoch=datetime.now(timezone.utc) - timedelta(days=8))]],
)
def test_verify_without_explicit_satellites_rejects_invalid_selection(monkeypatch, satellites):
    repository = _ProcessRepository()
    monkeypatch.setattr(repository, "_query_satids", lambda *_args, **_kwargs: satellites)

    assert repository.verify(epoch=datetime.now(timezone.utc)) is False


def test_verify_without_explicit_satellites_accepts_recent_elements(monkeypatch):
    repository = _ProcessRepository()
    current = datetime.now(timezone.utc)
    monkeypatch.setattr(
        repository,
        "_query_satids",
        lambda *_args, **_kwargs: [_satellite(1, epoch=current)],
    )

    assert repository.verify(epoch=current) is True
