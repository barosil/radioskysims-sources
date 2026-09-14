from __future__ import annotations

from unittest.mock import MagicMock

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
