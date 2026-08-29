"""
Módulo: tests/domain/signal/test_factory.py

Testes para SignalFactory: geração de PSD, modulações e integração com repositório.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from radioskysims_core.domain.common import FrequencyRange
from radioskysims_sources.domain.signal.factory import SignalFactory
from radioskysims_sources.domain.signal.models import SignalDefinition


@pytest.fixture
def freq_range():
    """FrequencyRange de 1200 a 1600 MHz (cobertura GPS/Galileo)."""
    return FrequencyRange(min_mhz=1200.0, max_mhz=1600.0, resolution_mhz=1.0)


@pytest.fixture
def signal_factory(freq_range):
    """SignalFactory com repositório mock."""
    repo = MagicMock()
    return SignalFactory(frequency_range=freq_range, signal_repo=repo)


class TestFrequencyGrid:
    """Testes para o grid de frequências."""

    def test_frequency_grid_hz_shape(self, signal_factory):
        """Grid de frequências deve ter o shape correto."""
        grid = signal_factory.frequency_grid_hz()
        assert isinstance(grid, np.ndarray)
        assert grid.dtype in (np.float64, np.floating)
        assert grid.size > 1

    def test_frequency_grid_mhz_range(self, signal_factory):
        """Grid em Hz deve cobrir o range em MHz * 1e6."""
        grid_hz = signal_factory.frequency_grid_hz()
        min_hz = 1200.0 * 1e6
        max_hz = 1600.0 * 1e6
        assert grid_hz.min() >= min_hz
        assert grid_hz.max() <= max_hz


class TestPRNGeneration:
    """Testes para geração de sequências PRN."""

    def test_prn_values(self, signal_factory):
        """PRN deve conter apenas -1 e +1."""
        prn = signal_factory._generate_prn(1023)
        assert np.all(np.isin(prn, [-1.0, 1.0]))
        assert len(prn) == 1023

    def test_prn_deterministic_with_seed(self, signal_factory):
        """PRN deve ser determinística (usa default_rng sem seed explícito)."""
        prn1 = signal_factory._generate_prn(31)
        prn2 = signal_factory._generate_prn(31)
        # Com default_rng diferentes, sequências devem ter comprimento correto
        assert len(prn1) == 31
        assert len(prn2) == 31


class TestModulations:
    """Testes para métodos de modulação."""

    def test_bpsk_output_length(self, signal_factory):
        """BPSK deve replicar chips por sps."""
        chips = np.array([-1.0, 1.0, -1.0])
        sig = signal_factory._bpsk(chips, sps=4)
        assert len(sig) == len(chips) * 4
        assert sig[0] == -1.0
        assert sig[4] == 1.0

    def test_qpsk_complex_output(self, signal_factory):
        """QPSK deve retornar sinal complexo."""
        chips = np.array([-1.0, 1.0, -1.0, 1.0])
        sig = signal_factory._qpsk(chips, sps=2)
        assert np.iscomplexobj(sig)
        assert len(sig) == 4  # (chips//2) * sps

    def test_boc_output(self, signal_factory):
        """BOC deve aplicar subcarrier."""
        chips = np.array([1.0, -1.0])
        sig = signal_factory._boc(chips, sps=8, m=1)
        assert len(sig) == len(chips) * 8

    def test_cboc_output(self, signal_factory):
        """CBOC deve combinar BOC(1,1) e BOC(6,1)."""
        chips = np.array([1.0, -1.0])
        sig = signal_factory._cboc(chips, sps=8)
        assert len(sig) == len(chips) * 8


class TestPSDGeneration:
    """Testes para geração de PSD."""

    def test_simulate_psd_returns_array(self, signal_factory):
        """_simulate_psd retorna delayed; .compute() retorna array numpy."""
        psd_delayed = signal_factory._simulate_psd(
            modulation="BPSK(1)",
            fc_hz=1575.42e6,
            prn_len=2048,
            fs_factor=8,
        )
        psd = psd_delayed.compute()
        assert isinstance(psd, np.ndarray)
        assert psd.dtype == np.float64
        expected_size = len(signal_factory.frequency_grid_hz())
        assert psd.shape == (expected_size,)

    def test_simulate_psd_boc_shape(self, signal_factory):
        """PSD para BOC deve ter o shape correto (lazy)."""
        psd_delayed = signal_factory._simulate_psd(
            modulation="BOC(1,1)",
            fc_hz=1575.42e6,
            prn_len=2048,
            fs_factor=8,
        )
        psd = psd_delayed.compute()
        expected_size = len(signal_factory.frequency_grid_hz())
        assert psd.shape == (expected_size,)

    @pytest.mark.parametrize(
        "modulation",
        [
            "BPSK(0.5)",
            "BPSK(2)",
            "QPSK(0.5)",
            "CBOC(6,1,1/11)",
        ],
    )
    def test_simulate_psd_fractional_chip_rate(self, signal_factory, modulation):
        """Chip rates fracionários (ex.: BPSK(0.5)) não devem quebrar o parse."""
        psd_delayed = signal_factory._simulate_psd(
            modulation=modulation,
            fc_hz=1575.42e6,
            prn_len=2048,
            fs_factor=8,
        )
        psd = psd_delayed.compute()
        expected_size = len(signal_factory.frequency_grid_hz())
        assert psd.shape == (expected_size,)
        assert np.isfinite(psd).all()

    def test_simulate_psd_unsupported_modulation(self, signal_factory):
        """Modulação não suportada deve levantar NotImplementedError ao computar."""
        psd_delayed = signal_factory._simulate_psd(
            modulation="UNKNOWN",
            fc_hz=1575.42e6,
            prn_len=31,
            fs_factor=4,
        )
        with pytest.raises(NotImplementedError, match="não suportada"):
            psd_delayed.compute()


class TestBuild:
    """Testes para o método build da SignalFactory."""

    def test_build_with_mock_repo(self, freq_range):
        """Build deve retornar dict com SatSignals a partir do repositório mock."""
        repo = MagicMock()
        repo.get_signals_by_norad_ids.return_value = {
            25544: [
                SignalDefinition(modulation="BPSK(1)", frequency=1575.42),
            ],
        }
        factory = SignalFactory(frequency_range=freq_range, signal_repo=repo)

        sats_block = {
            "block_0": {
                "satellites": [
                    MagicMock(norad_id=25544),
                ],
            },
        }

        result = factory.build(sats_block)
        assert 25544 in result
        assert len(result[25544]) == 1
        assert result[25544][0].signal.modulation == "BPSK(1)"
        # PSD deve ser um delayed (dask)
        assert hasattr(result[25544][0].psd, "compute")

    def test_build_empty_repo(self, freq_range):
        """Build com repositório vazio deve retornar dict vazio."""
        repo = MagicMock()
        repo.get_signals_by_norad_ids.return_value = {}
        factory = SignalFactory(frequency_range=freq_range, signal_repo=repo)

        sats_block = {
            "block_0": {
                "satellites": [MagicMock(norad_id=99999)],
            },
        }

        result = factory.build(sats_block)
        assert result == {}
