# application/services/satellite_manager.py
from __future__ import annotations

from radioskysims_sources.domain.satellite.factory import SatelliteFactory


class SatelliteManager:
    def __init__(self, factory: SatelliteFactory):
        self.factory = factory

    def set_satellite(self, satellite_id: str):
        self.satellite = self.factory.create_from_id(satellite_id)
