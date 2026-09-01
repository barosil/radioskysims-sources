"""\
Módulo: tests/application/services/test_radio_sources_manager.py

Descrição:
    Testes do facade ``RadioSourcesManager`` (construção, seleção, fluxo).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radioskysims_sources.application.services import RadioSourcesManager
from radioskysims_sources.application.services.flux_view import FluxDensity
from radioskysims_sources.domain.radiosources.selection import angular_separation_deg


def test_simulated_creates_n_sources():
    mgr = RadioSourcesManager.simulated(n_sources=25, seed=0)
    assert len(mgr) == 25
    assert len(mgr.sources) == 25


def test_simulated_deterministic():
    a = RadioSourcesManager.simulated(n_sources=10, seed=3)
    b = RadioSourcesManager.simulated(n_sources=10, seed=3)
    assert [s.id for s in a.sources] == [s.id for s in b.sources]
    assert np.allclose(
        [s.ra_deg for s in a.sources], [s.ra_deg for s in b.sources]
    )


def test_list_properties_dataframe():
    mgr = RadioSourcesManager.simulated(n_sources=8, seed=1)
    df = mgr.list_properties()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    for col in ["id", "ra_deg", "dec_deg", "nu0_mhz", "s0_jy", "alpha"]:
        assert col in df.columns


def test_select_by_ids():
    mgr = RadioSourcesManager.simulated(n_sources=10, seed=2)
    sub = mgr.select(ids=["SIM-0000", "SIM-0003"])
    assert [s.id for s in sub.sources] == ["SIM-0000", "SIM-0003"]


def test_select_by_min_flux():
    mgr = RadioSourcesManager.simulated(n_sources=100, seed=5)
    sub = mgr.select(min_flux_jy=1.0)
    assert len(sub) > 0
    assert all(s.spectral_model.s0_jy >= 1.0 for s in sub.sources)


def test_select_by_region():
    mgr = RadioSourcesManager.simulated(n_sources=200, seed=6)
    sub = mgr.select(ra_deg=0.0, dec_deg=0.0, radius_deg=30.0)
    assert len(sub) < len(mgr)
    for s in sub.sources:
        sep = angular_separation_deg(0.0, 0.0, s.ra_deg, s.dec_deg)
        assert float(sep) <= 30.0


def test_select_returns_independent_manager():
    mgr = RadioSourcesManager.simulated(n_sources=10, seed=9)
    sub = mgr.select(min_flux_jy=1e9)
    assert len(sub) == 0
    assert len(mgr) == 10  # o manager original permanece intacto


def test_flux_density_returns_fluxdensity():
    mgr = RadioSourcesManager.simulated(n_sources=5, seed=8)
    fd = mgr.flux_density()
    assert isinstance(fd, FluxDensity)
    assert len(fd) == 5
    assert fd.freqs_mhz.size == 51
    for f in fd.flux_by_source.values():
        assert f.shape == fd.freqs_mhz.shape
    # soma >= qualquer curva individual
    stacked = np.stack(list(fd.flux_by_source.values()))
    assert np.all(fd.total >= stacked.max(axis=0))


def test_flux_density_custom_freqs():
    mgr = RadioSourcesManager.simulated(n_sources=3, seed=10)
    freqs = np.array([245.0, 410.0, 610.0, 1415.0])
    fd = mgr.flux_density(freqs_mhz=freqs)
    assert np.allclose(fd.freqs_mhz, freqs)
