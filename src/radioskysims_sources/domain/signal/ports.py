"""
Módulo: src/radioskysims/domain/signal/ports.py

Port (contrato) para consulta de definições de sinais GNSS e para as PSDs
analíticas de modulação (compartilhadas com os satélites RINEX).

Copyright BINGO Collaboration
Last modified: 2026-09-01
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .models import SignalDefinition


class SignalRepositoryPort(ABC):
    """
    Porta para consulta de definições de sinais GNSS por satélite.

    Métodos:
        get_signals_by_norad_ids: Retorna as definições de sinal para
            uma lista de IDs NORAD.
    """

    SignalMap = dict[int, list[SignalDefinition]]

    @abstractmethod
    def get_signals_by_norad_ids(self, norad_ids: list[int]) -> SignalMap:
        """
        Retorna definições de sinal para os satélites especificados.

        Args:
            norad_ids: Lista de identificadores NORAD.

        Returns:
            dict[int, list[SignalDefinition]]: Mapeamento NORAD ID → lista
            de definições de sinal para aquele satélite.
        """
        raise NotImplementedError


class SignalPsdPort(Protocol):
    """Porta para PSDs analíticas de modulações GNSS (BPSK/BOC/CBOC).

    A implementação canônica vive em ``gnsstools4astro.domain.analytic_psd``
    (o MESMO código usado pelos satélites RINEX) e é entregue à
    ``SignalFactory`` por um adapter de infraestrutura — o domínio não
    depende do pacote GNSS diretamente.
    """

    def analytic_psd(
        self,
        modulation: str,
        freq_hz: npt.NDArray[np.floating],
        fc_hz: float,
    ) -> npt.NDArray[np.floating]:
        """PSD normalizada (1/Hz, integral unitária) da modulação no grid."""
        ...

    def psd_db(
        self,
        psd_w_hz: npt.NDArray[np.floating],
        floor_db: float = -300.0,
    ) -> npt.NDArray[np.floating]:
        """Converte PSD para dB com piso numérico (evita -inf)."""
        ...
