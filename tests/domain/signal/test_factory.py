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
    """SignalFactory com repositório mock e provider canônico injetado."""
    from radioskysims_sources.infrastructure.signal.gnsstools_psd import (
        GnsstoolsAnalyticPsdProvider,
    )

    repo = MagicMock()
    return SignalFactory(
        frequency_range=freq_range,
        signal_repo=repo,
        psd_provider=GnsstoolsAnalyticPsdProvider(),
    )


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


class TestPSDGeneration:
    """Testes para geração de PSD analítica."""

    def test_simulate_psd_returns_array(self, signal_factory):
        """_simulate_psd retorna delayed; .compute() retorna array numpy."""
        psd_delayed = signal_factory._simulate_psd(
            modulation="BPSK(1)",
            fc_hz=1575.42e6,
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
        )
        psd = psd_delayed.compute()
        expected_size = len(signal_factory.frequency_grid_hz())
        assert psd.shape == (expected_size,)
        assert np.isfinite(psd).all()

    def test_simulate_psd_no_high_floor_far_from_carrier(self, signal_factory):
        """Longe da portadora a PSD decai (sem piso de ruído de Welch)."""
        psd_delayed = signal_factory._simulate_psd(
            modulation="BPSK(1)",
            fc_hz=1575.42e6,
        )
        psd = psd_delayed.compute()
        grid = signal_factory.frequency_grid_hz()
        far = np.abs(grid - 1575.42e6) > 100e6
        if far.any():
            assert (psd[far] < -100.0).all()

    def test_simulate_psd_unsupported_modulation(self, signal_factory):
        """Modulação não suportada deve levantar NotImplementedError ao computar."""
        psd_delayed = signal_factory._simulate_psd(
            modulation="UNKNOWN",
            fc_hz=1575.42e6,
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


class StubPsdProvider:
    """Stub da porta ``SignalPsdPort`` (não depende de gnsstools4astro)."""

    def analytic_psd(self, modulation, freq_hz, fc_hz):
        del modulation
        return np.exp(-(((np.asarray(freq_hz, dtype=float)) - fc_hz) / 1e6) ** 2)

    def psd_db(self, psd_w_hz, floor_db=-300.0):
        with np.errstate(divide="ignore"):
            return np.maximum(10.0 * np.log10(psd_w_hz), floor_db).astype(np.float64)


class TestPsdPort:
    """Ports/adapters: SignalFactory depende da interface SignalPsdPort."""

    def test_uses_injected_psd_provider(self, freq_range):
        """Provider injetado (stub) é usado na geração da PSD."""
        repo = MagicMock()
        repo.get_signals_by_norad_ids.return_value = {
            1: [SignalDefinition(modulation="BPSK(1)", frequency=1575.42)],
        }
        factory = SignalFactory(
            frequency_range=freq_range,
            signal_repo=repo,
            psd_provider=StubPsdProvider(),
        )
        sats_block = {"b": {"satellites": [MagicMock(norad_id=1)]}}
        result = factory.build(sats_block)
        psd = result[1][0].psd.compute()
        grid = factory.frequency_grid_hz()
        j = int(np.argmin(np.abs(grid - 1575.42e6)))
        assert psd[j] > psd.mean()  # pico na portadora (stub gaussiano)
        assert np.isfinite(psd).all()

    def test_canonical_provider_injected(self, signal_factory):
        """A composição injeta o adapter canônico (gnsstools4astro)."""
        from radioskysims_sources.infrastructure.signal.gnsstools_psd import (
            GnsstoolsAnalyticPsdProvider,
        )

        assert isinstance(signal_factory._psd(), GnsstoolsAnalyticPsdProvider)

    def test_missing_provider_raises(self, freq_range):
        """Sem injeção, o domínio falha de forma explícita (não conhece infra)."""
        repo = MagicMock()
        factory = SignalFactory(frequency_range=freq_range, signal_repo=repo)
        with pytest.raises(ValueError, match="psd_provider"):
            factory._psd()
