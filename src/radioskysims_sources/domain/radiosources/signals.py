"""
Módulo: src/radioskysims/domain/radiosources/signals.py

Avaliador de sinal para fontes de rádio baseado no modelo espectral
em lei de potência S(ν) = S₀ (ν/ν₀)^(-α).

Fornece a lógica de domínio para obter o fluxo de uma RadioSource
em uma ou mais frequências, sem dependência de infraestrutura.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations


import numpy as np

from radioskysims_sources.domain.radiosources.models import (
    RadioSource,
    RadioSourceFluxSample,
)


class RadioSourceSignalEvaluator:
    """Avaliador de sinal para fontes de rádio.

    Dado um conjunto de fontes com modelo espectral (lei de potência),
    calcula o fluxo esperado em uma ou mais frequências.

    Fórmula:
        S(ν) = S₀ · (ν / ν₀)^(-α)

    onde:
        S₀ — fluxo de referência (Jy) na frequência ν₀ (MHz)
        α  — índice espectral
    """

    @staticmethod
    def evaluate_flux(
        source: RadioSource,
        freqs_mhz: np.ndarray | list[float],
    ) -> np.ndarray:
        """Avalia o fluxo (Jy) de uma única fonte em múltiplas frequências.

        Args:
            source: Fonte de rádio com modelo espectral.
            freqs_mhz: Array de frequências em MHz.

        Returns:
            Array de fluxo em Jy, mesma shape de freqs_mhz.
        """
        freqs = np.asarray(freqs_mhz, dtype=float)
        sm = source.spectral_model
        return sm.s0_jy * (freqs / sm.nu0_mhz) ** (-sm.alpha)

    @staticmethod
    def evaluate_flux_batch(
        sources: list[RadioSource],
        freqs_mhz: np.ndarray | list[float],
    ) -> list[RadioSourceFluxSample]:
        """Avalia o fluxo de múltiplas fontes em uma frequência de referência.

        Útil para determinar a fonte mais brilhante em uma dada frequência.

        Args:
            sources: Lista de fontes de rádio.
            freqs_mhz: Array de frequências em MHz.

        Returns:
            Lista de RadioSourceFluxSample com o fluxo integrado
            (soma sobre todas as frequências) ou fluxo na frequência central.
        """
        freqs = np.asarray(freqs_mhz, dtype=float)
        results: list[RadioSourceFluxSample] = []
        for src in sources:
            flux_arr = RadioSourceSignalEvaluator.evaluate_flux(src, freqs)
            # fluxo médio como métrica representativa
            flux_mean = float(np.mean(flux_arr))
            results.append(
                RadioSourceFluxSample(
                    source_id=src.id,
                    freq_mhz=float(np.mean(freqs)),
                    flux_jy=flux_mean,
                )
            )
        return results

    @staticmethod
    def evaluate_flux_grid(
        sources: list[RadioSource],
        freqs_mhz: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Avalia fluxo de múltiplas fontes em grade de frequências.

        Args:
            sources: Lista de fontes de rádio.
            freqs_mhz: Array 1D de frequências em MHz.

        Returns:
            Dict {source_id: np.ndarray} com fluxo (Jy) em cada frequência.
        """
        return {src.id: RadioSourceSignalEvaluator.evaluate_flux(src, freqs_mhz) for src in sources}
