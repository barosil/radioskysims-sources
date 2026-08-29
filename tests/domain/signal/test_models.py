"""
Módulo: tests/domain/signal/test_models.py

Testes para os modelos de domínio de signal: SignalDefinition e SatSignal.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from radioskysims_sources.domain.signal.models import SatSignal, SignalDefinition


class TestSignalDefinition:
    """Testes para o Value Object SignalDefinition."""

    def test_create_basic(self):
        """Cria SignalDefinition com parâmetros básicos."""
        sd = SignalDefinition(modulation="BPSK(1)", frequency=1575.42)
        assert sd.modulation == "BPSK(1)"
        assert sd.frequency == 1575.42
        assert sd.prn_len == 1023  # valor padrão

    def test_create_with_prn_len(self):
        """Cria SignalDefinition com prn_len personalizado."""
        sd = SignalDefinition(modulation="BOC(1,1)", frequency=1207.14, prn_len=2046)
        assert sd.prn_len == 2046

    def test_different_modulations(self):
        """Suporta diferentes tipos de modulação."""
        bpsk = SignalDefinition(modulation="BPSK(1)", frequency=1575.42)
        boc = SignalDefinition(modulation="BOC(1,1)", frequency=1207.14)
        cboc = SignalDefinition(modulation="CBOC", frequency=1575.42)
        qpsk = SignalDefinition(modulation="QPSK(1)", frequency=1227.60)
        assert all(s.modulation is not None for s in [bpsk, boc, cboc, qpsk])


class TestSatSignal:
    """Testes para o Value Object SatSignal."""

    def test_create_with_signal_definition(self):
        """Cria SatSignal com SignalDefinition e PSD."""
        sd = SignalDefinition(modulation="BPSK(1)", frequency=1575.42)
        psd = [0.0, 0.0]  # mock de PSD
        sig = SatSignal(signal=sd, psd=psd)
        assert sig.signal is sd
        assert sig.psd is psd
        assert sig.signal.modulation == "BPSK(1)"

    def test_signal_definition_immutable_values(self):
        """SignalDefinition deve ter valores consistentes."""
        sd = SignalDefinition(modulation="QPSK(1)", frequency=1227.60)
        assert sd.modulation == "QPSK(1)"
        assert sd.frequency == 1227.60
