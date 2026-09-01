"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/services/flux_view.py

Descrição:
    Holder leve de curvas de fluxo × frequência com visualização DEFERIDA
    para ``radioskysims-viz`` (mesmo padrão do SatelliteManager).

    Uso (notebook):
        >>> from radioskysims_sources.application.services.flux_view import FluxDensity
        >>> fd = FluxDensity(freqs_mhz, {"S1": flux_1, "S2": flux_2})
        >>> fd.view()          # matplotlib Figure (import lazy do viz)

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["FluxDensity"]


class FluxDensity:
    """Agregado de curvas de fluxo (Jy) × frequência (MHz).

    Attributes:
        freqs_mhz (np.ndarray): Grade de frequências em MHz.
        flux_by_source (dict[str, np.ndarray]): Fluxo por fonte (mesma
            shape de ``freqs_mhz``).
    """

    def __init__(
        self,
        freqs_mhz: np.ndarray | list[float],
        flux_by_source: dict[str, np.ndarray | list[float]],
    ) -> None:
        self.freqs_mhz = np.asarray(freqs_mhz, dtype=float)
        self.flux_by_source: dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=float) for k, v in flux_by_source.items()
        }

    @property
    def total(self) -> np.ndarray:
        """Soma do fluxo sobre todas as fontes (mesma shape de freqs)."""
        if not self.flux_by_source:
            return np.zeros_like(self.freqs_mhz)
        return np.sum(np.stack(list(self.flux_by_source.values())), axis=0)

    def view(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """Renderiza as curvas de fluxo (matplotlib), import lazy do viz."""
        from radioskysims_viz.sources import radio_sources_flux

        return radio_sources_flux(
            self.freqs_mhz, self.flux_by_source, title=title, **kwargs
        )

    def __len__(self) -> int:
        return len(self.flux_by_source)

    def __repr__(self) -> str:
        n = len(self.flux_by_source)
        return f"<FluxDensity: {n} fonte(s), {self.freqs_mhz.size} frequências>"
