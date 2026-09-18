from __future__ import annotations

import pandas as pd

from radioskysims_sources.infrastructure.db.satellite.satelliterepository import (
    SatelliteRepository,
)


def test_satellite_repository_repr_handles_empty_catalog():
    repository = SatelliteRepository.__new__(SatelliteRepository)
    repository.sats_df = pd.DataFrame()

    assert "Catalog vazio" in repr(repository)


def test_satellite_repository_repr_summarizes_constellations():
    repository = SatelliteRepository.__new__(SatelliteRepository)
    repository.sats_df = pd.DataFrame(
        [
            {"norad_id": 1, "constellation": "GPS", "num_elements": 2, "num_signals": 3},
            {"norad_id": 2, "constellation": "GPS", "num_elements": 1, "num_signals": 0},
            {"norad_id": 3, "constellation": "GALILEO", "num_elements": 1, "num_signals": 2},
        ]
    )

    text = repr(repository)

    assert "3 satélites em 2 constelações" in text
    assert "GPS" in text
    assert "GALILEO" in text
