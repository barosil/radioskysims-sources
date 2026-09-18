from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from radioskysims_sources.domain.radiosources.models import (
    RadioSource,
    SpectralModel,
)

from radioskysims_sources.application.services.catalog_services import (
    RadioSourceCatalogService,
)


class FakeLogger:
    """Logger estruturado mínimo para testes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def exception(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def test_catalog_service_accepts_injected_structured_logger() -> None:
    logger = FakeLogger()

    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=MagicMock(),
        track_provider=MagicMock(),
        logger=logger,
    )

    service._use_case.execute = MagicMock(return_value=[object()])

    result = service.ingest_nvss_specfind(
        ra_min_deg=10.0,
        ra_max_deg=20.0,
        dec_min_deg=-5.0,
        dec_max_deg=5.0,
        smin_jy=0.1,
    )

    assert result.status == "success"

    assert logger.events == [
        (
            "radio_source_catalog_ingest_success",
            {
                "num_sources": 1,
                "ra_min_deg": 10.0,
                "ra_max_deg": 20.0,
                "dec_min_deg": -5.0,
                "dec_max_deg": 5.0,
                "smin_jy": 0.1,
            },
        )
    ]


def _radio_source(
    source_id: str,
    *,
    s0_jy: float = 1.0,
    alpha: float = 0.0,
) -> RadioSource:
    return RadioSource(
        id=source_id,
        ra_deg=10.0,
        dec_deg=-5.0,
        spectral_model=SpectralModel(
            nu0_mhz=1000.0,
            s0_jy=s0_jy,
            alpha=alpha,
        ),
    )


def test_catalog_service_uses_fallback_logger_and_returns_partial_when_empty() -> None:
    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=MagicMock(),
        track_provider=MagicMock(),
    )
    service._use_case.execute = MagicMock(return_value=[])

    result = service.ingest_nvss_specfind(
        ra_min_deg=10.0,
        ra_max_deg=20.0,
        dec_min_deg=-5.0,
        dec_max_deg=5.0,
        smin_jy=0.1,
    )

    assert result.status == "partial"
    assert result.num_sources == 0
    assert result.catalog_name == "nvss_specfind"


def test_catalog_service_returns_failed_ingestion_when_use_case_raises() -> None:
    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=MagicMock(),
        track_provider=MagicMock(),
    )

    service._use_case.execute = MagicMock(
        side_effect=RuntimeError("catalog unavailable"),
    )

    result = service.ingest_nvss_specfind(
        ra_min_deg=10.0,
        ra_max_deg=20.0,
        dec_min_deg=-5.0,
        dec_max_deg=5.0,
        smin_jy=0.1,
    )

    assert result.status == "failed"
    assert result.num_sources == 0
    assert result.catalog_name == "nvss_specfind"
    assert result.error == "catalog unavailable"


def test_list_sources_in_region_delegates_to_repository() -> None:
    repository = MagicMock()
    expected = [
        _radio_source(
            "SRC-1",
            s0_jy=2.0,
        )
    ]
    repository.list_sources_in_region.return_value = expected

    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=repository,
        track_provider=MagicMock(),
    )

    result = service.list_sources_in_region(
        ra_deg=12.0,
        dec_deg=-3.0,
        radius_deg=4.5,
        min_flux_jy=0.7,
    )

    assert result == expected
    repository.list_sources_in_region.assert_called_once_with(
        ra_deg=12.0,
        dec_deg=-3.0,
        radius_deg=4.5,
        min_flux_jy=0.7,
    )


def test_compute_tracks_delegates_to_track_provider() -> None:
    track_provider = MagicMock()
    expected_track = MagicMock()
    track_provider.compute_track.return_value = expected_track

    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=MagicMock(),
        track_provider=track_provider,
    )

    sources = [_radio_source("SRC-1")]
    observer = MagicMock()
    time_range = MagicMock()

    result = service.compute_tracks(
        sources=sources,
        observer=observer,
        time_range=time_range,
    )

    assert result is expected_track
    track_provider.compute_track.assert_called_once_with(
        sources=sources,
        observer=observer,
        time_range=time_range,
    )


def test_evaluate_flux_returns_curve_for_each_source() -> None:
    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=MagicMock(),
        track_provider=MagicMock(),
    )

    sources = [
        _radio_source(
            "SRC-1",
            s0_jy=2.0,
            alpha=1.0,
        ),
        _radio_source(
            "SRC-2",
            s0_jy=4.0,
            alpha=1.0,
        ),
    ]

    result = service.evaluate_flux(
        sources,
        np.array([1000.0, 2000.0]),
    )

    assert set(result) == {"SRC-1", "SRC-2"}

    np.testing.assert_allclose(
        result["SRC-1"],
        [2.0, 1.0],
    )
    np.testing.assert_allclose(
        result["SRC-2"],
        [4.0, 2.0],
    )


def test_get_brightest_sources_orders_by_mean_flux_and_limits_result() -> None:
    service = RadioSourceCatalogService(
        ingest_port=MagicMock(),
        repository=MagicMock(),
        track_provider=MagicMock(),
    )

    faint = _radio_source(
        "FAINT",
        s0_jy=1.0,
    )
    bright = _radio_source(
        "BRIGHT",
        s0_jy=5.0,
    )
    medium = _radio_source(
        "MEDIUM",
        s0_jy=3.0,
    )

    result = service.get_brightest_sources(
        [faint, bright, medium],
        freqs_mhz=[900.0, 1000.0, 1100.0],
        top_n=2,
    )

    assert [source.id for source in result] == [
        "BRIGHT",
        "MEDIUM",
    ]
