"""
Simulação de satélites - serviço de domínio.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import xarray as xr

logger = logging.getLogger(__name__)
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
        try:
            logger.info(
                "satellite_simulation_start",
                extra={
                    "num_satellites": len(satellites),
                    "time_range_start": str(time_range.start),
                    "time_range_end": str(time_range.end),
                },
            )

            # Calcular geometria
            geometry = self.geometry_provider.compute(satellites, time_range)
            logger.info(
                "satellite_geometry_computed",
                extra={"num_satellites": len(satellites)},
            )

            # Simular sinais
            signals = self.signal_factory.simulate(satellites, geometry, time_range)
            logger.info(
                "satellite_simulation_success",
                extra={"num_satellites": len(satellites)},
            )

            return signals
        except Exception as e:
            logger.exception(
                "satellite_simulation_failed",
                extra={
                    "num_satellites": len(satellites),
                    "error": str(e),
                },
            )
            raise
