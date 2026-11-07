from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
import radioskysims_sources.infrastructure.external.radiosources_ingest as ingest_module
from radioskysims_sources.infrastructure.external.radiosources_ingest import (
    VizierTapIngestAdapter,
)


class _FakeTable:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def to_pandas(self) -> pd.DataFrame:
        return self._dataframe


class _FakeJob:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def get_results(self) -> _FakeTable:
        return _FakeTable(self._dataframe)


class _FakeTapPlus:
    instances: list[_FakeTapPlus] = []

    def __init__(self, url: str) -> None:
        self.url = url
        self.queries: list[str] = []
        self.dataframe = pd.DataFrame(
            {
                "source_id": ["SRC-1", "SRC-2"],
                "flux_jy": [1.5, 2.5],
            }
        )
        self.__class__.instances.append(self)

    def launch_job(self, query: str) -> _FakeJob:
        self.queries.append(query)
        return _FakeJob(self.dataframe)


@pytest.fixture(autouse=True)
def reset_fake_tap_instances() -> None:
    _FakeTapPlus.instances.clear()


def test_adapter_initializes_tap_client(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _FakeTapPlus,
    )

    tap_url = "https://example.test/tap"

    adapter = VizierTapIngestAdapter(tap_url)

    assert adapter._tap_url == tap_url
    assert len(_FakeTapPlus.instances) == 1
    assert _FakeTapPlus.instances[0].url == tap_url
    assert adapter._tap is _FakeTapPlus.instances[0]


def test_ingest_from_tap_executes_query_and_returns_dataframe(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _FakeTapPlus,
    )

    adapter = VizierTapIngestAdapter(
        "https://example.test/tap",
    )

    query = "SELECT * FROM nvss"

    result = adapter.ingest_from_tap(
        survey="NVSS",
        adql_query=query,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result["source_id"]) == [
        "SRC-1",
        "SRC-2",
    ]
    assert list(result["flux_jy"]) == [
        1.5,
        2.5,
    ]
    assert _FakeTapPlus.instances[0].queries == [query]


def test_ingest_from_file_reads_csv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _FakeTapPlus,
    )

    path = tmp_path / "catalog.csv"

    expected = pd.DataFrame(
        {
            "source_id": ["SRC-1", "SRC-2"],
            "flux_jy": [1.2, 3.4],
        }
    )
    expected.to_csv(path, index=False)

    adapter = VizierTapIngestAdapter(
        "https://example.test/tap",
    )

    result = adapter.ingest_from_file(
        path,
        format="csv",
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_ingest_from_file_accepts_case_insensitive_csv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _FakeTapPlus,
    )

    path = tmp_path / "catalog.csv"

    expected = pd.DataFrame(
        {
            "source_id": ["SRC-1"],
            "flux_jy": [2.0],
        }
    )
    expected.to_csv(path, index=False)

    adapter = VizierTapIngestAdapter(
        "https://example.test/tap",
    )

    result = adapter.ingest_from_file(
        path,
        format="CSV",
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_ingest_from_file_rejects_unsupported_format(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _FakeTapPlus,
    )

    adapter = VizierTapIngestAdapter(
        "https://example.test/tap",
    )

    path = tmp_path / "catalog.fits"

    with pytest.raises(
        ValueError,
        match="Formato de arquivo não suportado",
    ):
        adapter.ingest_from_file(
            path,
            format="fits",
        )


class _SequencedTapPlus:
    """Fake que falha no primeiro servidor e responde nos demais."""

    instances: ClassVar[list[_SequencedTapPlus]] = []

    def __init__(self, url: str) -> None:
        self.url = url
        self.queries: list[str] = []
        self.__class__.instances.append(self)

    def launch_job(self, query: str) -> _FakeJob:
        self.queries.append(query)
        if self is self.__class__.instances[0]:
            msg = "servidor primário fora do ar"
            raise RuntimeError(msg)
        return _FakeJob(
            pd.DataFrame(
                {
                    "source_id": ["SRC-9"],
                    "flux_jy": [9.0],
                }
            )
        )


class _AlwaysFailTapPlus:
    """Fake que falha em todos os servidores."""

    instances: ClassVar[list[_AlwaysFailTapPlus]] = []

    def __init__(self, url: str) -> None:
        self.url = url
        self.__class__.instances.append(self)

    def launch_job(self, _query: str) -> None:
        msg = "TAP indisponível"
        raise RuntimeError(msg)


def test_adapter_accepts_list_of_servers(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _FakeTapPlus,
    )

    urls = ["https://a.test/tap", "https://b.test/tap"]
    adapter = VizierTapIngestAdapter(urls)

    assert adapter._tap_urls == urls
    assert adapter._tap_url == urls[0]
    assert adapter._tap is _FakeTapPlus.instances[0]


def test_ingest_from_tap_falls_back_when_first_server_fails(monkeypatch) -> None:
    _SequencedTapPlus.instances.clear()
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _SequencedTapPlus,
    )

    adapter = VizierTapIngestAdapter(
        ["https://primary.test/tap", "https://mirror.test/tap"],
    )

    result = adapter.ingest_from_tap(
        survey="NVSS",
        adql_query="SELECT 1",
    )

    assert list(result["source_id"]) == ["SRC-9"]
    assert len(_SequencedTapPlus.instances) == 2
    assert _SequencedTapPlus.instances[0].queries == ["SELECT 1"]
    assert _SequencedTapPlus.instances[1].queries == ["SELECT 1"]


def test_ingest_from_tap_raises_when_all_servers_fail(monkeypatch) -> None:
    _AlwaysFailTapPlus.instances.clear()
    monkeypatch.setattr(
        ingest_module,
        "TapPlus",
        _AlwaysFailTapPlus,
    )

    adapter = VizierTapIngestAdapter(
        ["https://a.test/tap", "https://b.test/tap"],
    )

    with pytest.raises(
        RuntimeError,
        match="Todos os servidores TAP VizieR falharam",
    ):
        adapter.ingest_from_tap(
            survey="NVSS",
            adql_query="SELECT 1",
        )

    assert len(_AlwaysFailTapPlus.instances) == 2
