"""
Módulo: src/radioskysims/domain/signal/models.py

Modelos de domínio para sinais de emissão de satélites.

Define SignalDefinition (definição do sinal GNSS) e SatSignal
(sinal com PSD lazy), utilizados por SignalFactory e SatelliteAssembler.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SignalDefinition:
    """
    Definição de um sinal GNSS transmitido por um satélite.

    Attributes:
        modulation: Tipo de modulação (ex: "BPSK(1)", "BOC(1,1)", "CBOC").
        frequency: Frequência da portadora em MHz.
        prn_len: Comprimento da sequência PRN (padrão: 1023).
    """

    modulation: str
    frequency: float
    prn_len: int = 1023


class SatSignal:
    """
    Domain value object representing a satellite emission signal.

    This object binds:
    - a signal definition (modulation, carrier frequency, metadata)
    - a lazily-evaluated PSD(freq)

    It does NOT encode time, geometry or instrument response.
    """

    def __init__(
        self,
        *,
        signal: SignalDefinition,
        psd: Any,
    ) -> None:
        self.signal = signal
        self.psd = psd
