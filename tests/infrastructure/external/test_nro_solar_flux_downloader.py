"""Testes do downloader de fluxo solar NoRP/NRO (XDR/IDL SAVE) — item 4.

Módulo: tests/infrastructure/external/test_nro_solar_flux_downloader.py

Usa um ``readsav`` falso (monkeypatch) para validar o parsing sem rede.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from radioskysims_sources.infrastructure.external.nro_solar_flux_downloader import (
    NroSolarFluxDownloader,
)


def _fake_save():
    tim = np.array(
        [(0, 17379), (1000, 17379), (2000, 17380)],
        dtype=[("TIME", ">i4"), ("DAY", ">i2")],
    )
    return {
        # Stokes I (absolute measurements), em SFU
        "fi": np.array([[10.0, 100.0], [20.0, 200.0], [30.0, np.nan]]),
        # Stokes V, em SFU
        "fv": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]]),
        "freq": np.array([1.0, 2.0]),  # GHz
        "tim": tim,
    }


@pytest.fixture
def patched_readsav(monkeypatch):
    import scipy.io

    monkeypatch.setattr(scipy.io, "readsav", lambda *_a, **_k: _fake_save())


@pytest.mark.usefixtures("patched_readsav")
def test_parse_save_builds_long_dataframe():
    df = NroSolarFluxDownloader.parse_save("/tmp/whatever.xdr")

    assert list(df.columns) == ["time", "frequency_mhz", "flux_jy"]
    # 2 canais × 3 amostras = 6 linhas
    assert len(df) == 6
    assert sorted(df["frequency_mhz"].unique()) == [1000.0, 2000.0]
    # SFU → Jy (×1e4)
    first = df[(df["frequency_mhz"] == 1000.0)].sort_values("time").iloc[0]
    assert first["flux_jy"] == pytest.approx(10.0 * 1.0e4)
    # Época: DAY=17379 → 2026-08-01; TIME em ms.
    assert str(df["time"].min())[:10] == "2026-08-01"


@pytest.mark.usefixtures("patched_readsav")
def test_parse_save_quantity_v_uses_fv():
    df = NroSolarFluxDownloader.parse_save("/tmp/x.xdr", quantity="V")
    first = df[(df["frequency_mhz"] == 1000.0)].sort_values("time").iloc[0]
    assert first["flux_jy"] == pytest.approx(1.0 * 1.0e4)


def test_parse_save_drops_all_nan_channels(monkeypatch):
    import scipy.io

    save = _fake_save()
    save["fi"] = np.array([[10.0, np.nan], [20.0, np.nan], [30.0, np.nan]])
    monkeypatch.setattr(scipy.io, "readsav", lambda *_a, **_k: save)

    df = NroSolarFluxDownloader.parse_save("/tmp/x.xdr", drop_dead_channels=True)

    assert sorted(df["frequency_mhz"].unique()) == [1000.0]


def test_build_url_matches_nro_layout():
    dl = NroSolarFluxDownloader()
    url = dl._build_url(date(2026, 8, 1))
    assert url == "https://solar.nro.nao.ac.jp/norp/xdr/2026/08/norp260801.xdr"


def test_invalid_quantity_raises():
    with pytest.raises(ValueError, match="quantity"):
        NroSolarFluxDownloader(quantity="X")
