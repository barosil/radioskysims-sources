from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from radioskysims_sources.application.services.source_assembler import SourceAssembler


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def debug(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def test_source_assembler_logs_solar_flux_fetch_failure() -> None:
    logger = FakeLogger()

    local_sources_service = MagicMock()
    local_sources_service.build_local_sources.return_value = SimpleNamespace(
        sources=[
            SimpleNamespace(
                name="Sun",
                has_track=True,
            )
        ]
    )
    local_sources_service.get_solar_flux.side_effect = RuntimeError(
        "solar flux unavailable"
    )

    # Faz o processamento parar depois do fallback de fluxo,
    # sem precisar montar um track real neste teste.
    local_sources_service.get_local_source_track.side_effect = RuntimeError(
        "track unavailable"
    )

    assembler = SourceAssembler(
        local_sources_service=local_sources_service,
        logger=logger,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    assembler.assemble_local_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
    )

    assert logger.events[0] == (
        "solar_flux_fetch_failed",
        {
            "error_type": "RuntimeError",
            "error": "solar flux unavailable",
        },
    )
def test_source_assembler_logs_solar_flux_default_fallback() -> None:
    logger = FakeLogger()

    local_sources_service = MagicMock()
    local_sources_service.build_local_sources.return_value = SimpleNamespace(
        sources=[
            SimpleNamespace(
                name="Sun",
                has_track=True,
            )
        ]
    )
    local_sources_service.get_solar_flux.side_effect = RuntimeError(
        "solar flux unavailable"
    )
    local_sources_service.get_local_source_track.side_effect = RuntimeError(
        "track unavailable"
    )

    assembler = SourceAssembler(
        local_sources_service=local_sources_service,
        logger=logger,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    assembler.assemble_local_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
    )

    assert (
        "solar_flux_default_fallback",
        {
            "solar_flux_sfu": 150.0,
        },
    ) in logger.events
def test_source_assembler_logs_solar_flux_interpolation_failure() -> None:
    logger = FakeLogger()

    solar_flux = MagicMock()
    solar_flux.evaluate_grid.side_effect = RuntimeError(
        "interpolation failed"
    )

    local_sources_service = MagicMock()
    local_sources_service.build_local_sources.return_value = SimpleNamespace(
        sources=[
            SimpleNamespace(
                name="Sun",
                has_track=True,
            )
        ]
    )
    local_sources_service.get_solar_flux.return_value = solar_flux
    local_sources_service.get_local_source_track.side_effect = RuntimeError(
        "track unavailable"
    )

    assembler = SourceAssembler(
        local_sources_service=local_sources_service,
        logger=logger,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    assembler.assemble_local_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
    )

    assert (
        "solar_flux_interpolation_failed",
        {
            "error_type": "RuntimeError",
            "error": "interpolation failed",
        },
    ) in logger.events
def test_source_assembler_logs_local_source_track_failure() -> None:
    logger = FakeLogger()

    solar_flux = MagicMock()
    solar_flux.evaluate_grid.return_value = np.ones((2, 2), dtype=float)

    local_sources_service = MagicMock()
    local_sources_service.build_local_sources.return_value = SimpleNamespace(
        sources=[
            SimpleNamespace(
                name="Sun",
                has_track=True,
            )
        ]
    )
    local_sources_service.get_solar_flux.return_value = solar_flux
    local_sources_service.get_local_source_track.side_effect = RuntimeError(
        "track unavailable"
    )

    assembler = SourceAssembler(
        local_sources_service=local_sources_service,
        logger=logger,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    assembler.assemble_local_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
    )

    assert (
        "local_source_track_failed",
        {
            "source_name": "Sun",
            "error_type": "RuntimeError",
            "error": "track unavailable",
        },
    ) in logger.events
def test_source_assembler_logs_radio_source_track_failure() -> None:
    logger = FakeLogger()

    source = SimpleNamespace(
        id="SRC-001",
        ra_deg=180.0,
        dec_deg=0.0,
    )

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = [source]
    radio_catalog.evaluate_flux.return_value = {
        "SRC-001": np.array([1.0, 1.0], dtype=float)
    }
    radio_catalog.compute_tracks.side_effect = RuntimeError(
        "radio track unavailable"
    )

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
    )

    assert result == []
    assert (
        "radio_source_track_failed",
        {
            "error_type": "RuntimeError",
            "error": "radio track unavailable",
        },
    ) in logger.events


def test_source_assembler_logs_no_radio_sources_in_region() -> None:
    logger = FakeLogger()

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = []

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
    )

    assert result == []
    assert (
        "radio_sources_not_found_in_region",
        {
            "ra_deg": 180.0,
            "dec_deg": 0.0,
            "radius_deg": 5.0,
            "min_flux_jy": 0.1,
        },
    ) in logger.events
def test_source_assembler_logs_no_radio_sources_in_beam(monkeypatch) -> None:
    logger = FakeLogger()

    source = SimpleNamespace(
        id="SRC-001",
        ra_deg=180.0,
        dec_deg=0.0,
    )

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = [source]

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    def fake_select_sources_in_beam(*_args: object, **_kwargs: object):
        return [], np.array([], dtype=float)

    monkeypatch.setattr(
        "radioskysims_sources.domain.radiosources.selection."
        "select_sources_in_beam",
        fake_select_sources_in_beam,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
        beam_fwhm_deg=10.0,
    )

    assert result == []
    assert (
        "radio_sources_not_found_in_beam",
        {
            "ra_deg": 180.0,
            "dec_deg": 0.0,
            "beam_radius_deg": 30.0,
        },
    ) in logger.events
def test_source_assembler_logs_radio_sources_beam_filter(monkeypatch) -> None:
    logger = FakeLogger()

    source_a = SimpleNamespace(
        id="SRC-001",
        ra_deg=180.0,
        dec_deg=0.0,
    )
    source_b = SimpleNamespace(
        id="SRC-002",
        ra_deg=181.0,
        dec_deg=0.0,
    )

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = [
        source_a,
        source_b,
    ]
    radio_catalog.evaluate_flux.return_value = {
        "SRC-001": np.array([1.0, 1.0], dtype=float),
    }
    radio_catalog.compute_tracks.side_effect = RuntimeError(
        "stop after beam filter"
    )

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    def fake_select_sources_in_beam(
        *_args: object,
        **_kwargs: object,
    ):
        return [source_a], np.array([0.0], dtype=float)

    monkeypatch.setattr(
        "radioskysims_sources.domain.radiosources.selection."
        "select_sources_in_beam",
        fake_select_sources_in_beam,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
        beam_fwhm_deg=10.0,
    )

    assert result == []
    assert (
        "radio_sources_beam_filter",
        {
            "sources_before": 2,
            "sources_after": 1,
            "sources_discarded": 1,
        },
    ) in logger.events
def test_source_assembler_logs_radio_sources_assembly_started(monkeypatch) -> None:
    logger = FakeLogger()

    source = SimpleNamespace(
        id="SRC-001",
        ra_deg=180.0,
        dec_deg=0.0,
    )

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = [source]
    radio_catalog.evaluate_flux.return_value = {
        "SRC-001": np.array([1.0, 1.0], dtype=float),
    }
    radio_catalog.compute_tracks.side_effect = RuntimeError(
        "stop after assembly start"
    )

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    def fake_select_sources_in_beam(
        *_args: object,
        **_kwargs: object,
    ):
        return [source], np.array([0.0], dtype=float)

    monkeypatch.setattr(
        "radioskysims_sources.domain.radiosources.selection."
        "select_sources_in_beam",
        fake_select_sources_in_beam,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
        beam_fwhm_deg=10.0,
    )

    assert result == []
    assert (
        "radio_sources_assembly_started",
        {
            "source_count": 1,
        },
    ) in logger.events
def test_source_assembler_logs_radio_sources_grouped(monkeypatch) -> None:
    logger = FakeLogger()

    source_a = SimpleNamespace(
        id="SRC-001",
        ra_deg=180.0,
        dec_deg=0.0,
    )
    source_b = SimpleNamespace(
        id="SRC-002",
        ra_deg=180.0,
        dec_deg=0.0,
    )

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = [
        source_a,
        source_b,
    ]
    radio_catalog.evaluate_flux.return_value = {
        "SRC-001": np.array([1.0, 1.0], dtype=float),
        "SRC-002": np.array([2.0, 2.0], dtype=float),
    }
    radio_catalog.compute_tracks.return_value = MagicMock()

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    def fake_select_sources_in_beam(
        *_args: object,
        **_kwargs: object,
    ):
        return [source_a, source_b], np.array([0.0, 0.0], dtype=float)

    monkeypatch.setattr(
        "radioskysims_sources.domain.radiosources.selection."
        "select_sources_in_beam",
        fake_select_sources_in_beam,
    )

    monkeypatch.setattr(
        "radioskysims_sources.application.services.source_assembler."
        "_extract_single_source_track",
        lambda *_args, **_kwargs: MagicMock(),
    )

    monkeypatch.setattr(
        "radioskysims_sources.application.services.source_assembler."
        "RadioSourceSignalSource",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    monkeypatch.setattr(
        "radioskysims_sources.domain.signal_source_group."
        "RadioSourceGroupSignalSource.group_by_position",
        lambda signal_sources: [signal_sources[0]],
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
        beam_fwhm_deg=10.0,
    )

    assert len(result) == 1
    assert (
        "radio_sources_grouped",
        {
            "sources_before": 2,
            "sources_after": 1,
            "reduction_percent": 50.0,
        },
    ) in logger.events
def test_source_assembler_logs_radio_source_without_track(monkeypatch) -> None:
    logger = FakeLogger()

    source = SimpleNamespace(
        id="SRC-001",
        ra_deg=180.0,
        dec_deg=0.0,
    )

    radio_catalog = MagicMock()
    radio_catalog.list_sources_in_region.return_value = [source]
    radio_catalog.evaluate_flux.return_value = {
        "SRC-001": np.array([1.0, 1.0], dtype=float),
    }
    radio_catalog.compute_tracks.return_value = MagicMock()

    assembler = SourceAssembler(
        radio_source_catalog_service=radio_catalog,
        logger=logger,
    )

    def fake_select_sources_in_beam(
        *_args: object,
        **_kwargs: object,
    ):
        return [source], np.array([0.0], dtype=float)

    monkeypatch.setattr(
        "radioskysims_sources.domain.radiosources.selection."
        "select_sources_in_beam",
        fake_select_sources_in_beam,
    )

    monkeypatch.setattr(
        "radioskysims_sources.application.services.source_assembler."
        "_extract_single_source_track",
        lambda *_args, **_kwargs: None,
    )

    time_grid = np.array(
        [
            "2026-07-26T00:00:00",
            "2026-07-26T00:01:00",
        ],
        dtype="datetime64[ns]",
    )

    freq_mhz = np.array([980.0, 1260.0], dtype=float)

    result = assembler.assemble_radio_sources(
        observer=MagicMock(),
        time_range=MagicMock(),
        time_grid=time_grid,
        freq_mhz=freq_mhz,
        ra_center_deg=180.0,
        dec_center_deg=0.0,
        search_radius_deg=5.0,
        min_flux_jy=0.1,
        beam_fwhm_deg=10.0,
    )

    assert result == []
    assert (
        "radio_source_track_missing",
        {
            "source_id": "SRC-001",
        },
    ) in logger.events
