"""
Módulo: src/radioskysims/domain/signal/ports.py

Port (contrato) para consulta de definições de sinais GNSS.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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
