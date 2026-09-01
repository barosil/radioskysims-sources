"""\
Módulo: tests/domain/satellite/test_downloader_prn.py

Descrição:
    Testes de extração de PRN no downloader CelesTrak (``fetch_sats``) e da
    propagação do PRN no seletor de TLE (``TLESelector.select``) — sem rede.

Copyright BINGO Collaboration
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from radioskysims_sources.domain.satellite.models import OrbitalElement, Satellite
from radioskysims_sources.domain.satellite.selector import TLESelector
from radioskysims_sources.infrastructure.external.satellite_downloader import (
    CelestrakClient,
)

class _FakeResponse:
    status_code = 200

    def __init__(self, content):
        self.content = content


# ----------------------------------------------------------------------
# CelestrakClient.fetch_sats → coluna prn
# ----------------------------------------------------------------------
class TestFetchSatsPrn:
    @staticmethod
    def _fake_read_html(*_args, **_kwargs):
        table = pd.DataFrame(
            [
                {
                    "International Designator": "2019-XXXX",
                    "NORAD Catalog Number": 26407,
                    "Name": "GPS BIIR-5 (PRN 22)",
                    "Period (minutes)": 700.0,
                    "Inclination (degrees)": 55.0,
                    "Apogee Height (km)": 20200.0,
                    "Perigee Height (km)": 20000.0,
                    "Eccentricity": 0.001,
                    "GP Age (days)": 1.0,
                },
                {
                    "International Designator": "2020-YYYY",
                    "NORAD Catalog Number": 61186,
                    "Name": "BEIDOU-3 M25 (C47)",
                    "Period (minutes)": 770.0,
                    "Inclination (degrees)": 55.0,
                    "Apogee Height (km)": 35786.0,
                    "Perigee Height (km)": 35700.0,
                    "Eccentricity": 0.002,
                    "GP Age (days)": 2.0,
                },
                {
                    "International Designator": "2014-ZZZZ",
                    "NORAD Catalog Number": 40128,
                    "Name": "GSAT0201 (GALILEO 5)",
                    "Period (minutes)": 770.0,
                    "Inclination (degrees)": 56.0,
                    "Apogee Height (km)": 23222.0,
                    "Perigee Height (km)": 23222.0,
                    "Eccentricity": 0.001,
                    "GP Age (days)": 1.0,
                },
            ]
        )
        return [pd.DataFrame(), table]

    def test_fetch_sats_adds_prn(self, monkeypatch):
        monkeypatch.setattr(pd, "read_html", self._fake_read_html)
        monkeypatch.setattr(
            "radioskysims_sources.infrastructure.external.satellite_downloader.httpx.Client",
            lambda *a, **k: _FakeClient(),
        )

        df = CelestrakClient().fetch_sats()
        assert "prn" in df.columns
        by_norad = df.set_index("norad_id")["prn"].to_dict()
        assert by_norad[26407] == 22  # GPS
        assert by_norad[61186] == 47  # BeiDou
        # GALILEO: número do veículo (5) — sem PRN derivável do nome
        assert by_norad[40128] == 5


class _FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url):
        return _FakeResponse(b"<html><table></table><table></table></html>")


# ----------------------------------------------------------------------
# TLESelector.select → propaga prn ao modelo de domínio
# ----------------------------------------------------------------------
def test_selector_propagates_prn():
    from types import SimpleNamespace

    epoch = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    # Satélite "persistido" com prn (como virá do SatID após a migração).
    satid = SimpleNamespace(
        norad_id=26407,
        name="GPS BIIR-5 (PRN 22)",
        constellation="GPS",
        prn=22,
        elements=[
            OrbitalElement(
                epoch=epoch,
                norad_id=26407,
                tle_line1="1 x",
                tle_line2="2 x",
            )
        ],
    )
    selected = TLESelector.select([satid], epoch_jd=epoch.timestamp() / 86400.0 + 2440587.5)
    domain = Satellite.model_validate(selected[0])
    assert domain.prn == 22
    assert domain.sv == "G22"
