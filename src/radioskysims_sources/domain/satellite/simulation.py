"""
Simulação de satélites - serviço de domínio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from ..common.models import TimeRange
    from .models import Satellite


class Simulation:
    """Serviço de domínio para simulações de satélites."""

    def __init__(self, geometry_provider, signal_factory):
        """
        Inicializa a simulação com provedores de geometria e sinal.

        Args:
            geometry_provider: Provedor para cálculo de geometria
            signal_factory: Factory para simulação de sinais
        """
        self.geometry_provider = geometry_provider
        self.signal_factory = signal_factory

    def run(self, satellites: list[Satellite], time_range: TimeRange) -> xr.Dataset:
        """
        Executa simulação de satélites no intervalo de tempo.

        Args:
            satellites: Lista de satélites para simular
            time_range: Intervalo de tempo da simulação

        Returns:
            Dataset xarray com resultados da simulação
        """
        # Calcular geometria
        geometry = self.geometry_provider.compute(satellites, time_range)

        # Simular sinais
        signals = self.signal_factory.simulate(satellites, geometry, time_range)

        return signals
