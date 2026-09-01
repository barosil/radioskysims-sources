"""\
Módulo: tests/application/services/test_radio_sources_manager_tap.py

Descrição:
    Testes do fluxo TAP real (``RadioSourcesManager.from_tap``) com um
    adapter de ingestão fake (sem rede).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import pandas as pd
import pytest

from radioskysims_sources.application.services import RadioSourcesManager


class FakeIngest:
    """Adapter de ingestão TAP fake (NVSS + SPECFIND) sem rede."""

    def ingest_from_tap(self, survey: str, adql_query: str):  # noqa: ARG002 - interface TAP
        if survey == "NVSS":
            return pd.DataFrame(
                {
                    "NVSS": ["123456+123456", "234567-001122"],
                    "RAJ2000": [193.0, 200.0],
                    "DEJ2000": [12.5, -0.2],
                    "S1_4": [500.0, 800.0],  # mJy
                }
            )
        # SPECFIND: apenas a primeira fonte tem medida espectral
        # (coluna S_nu_ = nome normalizado de "S(nu)" após a query TAP)
        return pd.DataFrame(
            {
                "Name": ["NVSS 123456+123456"],
                "nu": [610.0],
                "S_nu_": [1.5],  # Jy
            }
        )

    def ingest_from_file(self, path, format):
        raise NotImplementedError


def test_from_tap_with_fake_ingest():
    mgr = RadioSourcesManager.from_tap(
        ra_min_deg=0.0,
        ra_max_deg=360.0,
        dec_min_deg=-90.0,
        dec_max_deg=90.0,
        smin_jy=0.1,
        ingest_port=FakeIngest(),
    )
    assert len(mgr) == 2
    df = mgr.list_properties()
    assert len(df) == 2
    assert set(df["id"]) == {"123456+123456", "234567-001122"}


def test_from_tap_converts_flux_and_alpha():
    mgr = RadioSourcesManager.from_tap(
        ra_min_deg=0.0,
        ra_max_deg=360.0,
        dec_min_deg=-90.0,
        dec_max_deg=90.0,
        smin_jy=0.1,
        ingest_port=FakeIngest(),
    )
    s = mgr.select(ids=["123456+123456"]).sources[0]
    # S1.4 = 500 mJy → 0.5 Jy em 1400 MHz
    assert s.spectral_model.s0_jy == pytest.approx(0.5)
    assert s.spectral_model.nu0_mhz == pytest.approx(1400.0)
    # alpha estimado a partir do ponto SPECFIND (610 MHz, 1.5 Jy) → > 0
    assert s.spectral_model.alpha > 0.0
    assert s.spectral_model.alpha != pytest.approx(0.7)  # não caiu no default
