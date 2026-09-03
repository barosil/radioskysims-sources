"""\
Módulo: tests/application/orchestrators/test_radio_sources_orchestrator.py

Descrição:
    Testes da fachada ``RadioSourcesOrchestrator`` (UX Onda 3/R6).

Copyright BINGO Collaboration
"""

from __future__ import annotations

from radioskysims_sources import RadioSourcesOrchestrator


def test_simulated_cria_catalogo():
    rs = RadioSourcesOrchestrator.simulated(n_sources=25, seed=0)
    assert len(rs) == 25
    assert len(rs.sources) == 25


def test_select_encadeia_orquestrador():
    rs = RadioSourcesOrchestrator.simulated(n_sources=100, seed=5)
    sub = rs.select(min_flux_jy=1.0)
    assert isinstance(sub, RadioSourcesOrchestrator)
    assert 0 < len(sub) < len(rs)


def test_flux_density():
    rs = RadioSourcesOrchestrator.simulated(n_sources=3, seed=1)
    fd = rs.flux_density()
    assert len(fd) == 3
    assert fd.freqs_mhz.size == 51


def test_summary_e_list_properties():
    rs = RadioSourcesOrchestrator.simulated(n_sources=5, seed=2)
    assert "RadioSourcesManager" in rs.summary()
    df = rs.list_properties()
    assert len(df) == 5
    assert "ra_deg" in df.columns
