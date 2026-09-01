"""\
Módulo: src/radioskysims_sources/infrastructure/signal/gnsstools_psd.py

Adapter da porta ``SignalPsdPort`` usando o pacote ``gnsstools4astro``.

A PSD analítica (closed-form BPSK/BOC/CBOC) é o MESMO código usado pelos
satélites RINEX (``gnsstools4astro.domain.analytic_psd``). Este adapter é o
único ponto do ``sources`` que importa o pacote GNSS — o domínio
(``SignalFactory``) depende apenas da interface ``SignalPsdPort``.

Copyright BINGO Collaboration
Last modified: 2026-09-01
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gnsstools4astro.domain.analytic_psd import (
    analytic_psd as _analytic_psd,
    psd_db as _psd_db,
)


class GnsstoolsAnalyticPsdProvider:
    """Implementa ``SignalPsdPort`` delegando a ``gnsstools4astro``."""

    def analytic_psd(
        self,
        modulation: str,
        freq_hz: npt.NDArray[np.floating],
        fc_hz: float,
    ) -> npt.NDArray[np.floating]:
        """PSD normalizada (1/Hz) — delega para o módulo compartilhado."""
        return _analytic_psd(modulation, freq_hz, fc_hz)

    def psd_db(
        self,
        psd_w_hz: npt.NDArray[np.floating],
        floor_db: float = -300.0,
    ) -> npt.NDArray[np.floating]:
        """Converte PSD para dB com piso numérico."""
        return _psd_db(psd_w_hz, floor_db)
