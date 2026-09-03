"""
Módulo: src/radioskysims/domain/signal/factory.py

Domain factory for spectral emission models (PSDs analíticas).

As PSDs são calculadas por expressões fechadas (closed-form) de
BPSK/QPSK/BOC/CBOC — compartilhadas com os satélites RINEX. O domínio depende
apenas da **porta** ``SignalPsdPort``; a implementação canônica
(``gnsstools4astro.domain.analytic_psd``) entra por um adapter de
infraestrutura (``GnsstoolsAnalyticPsdProvider``). Substitui a simulação no
domínio do tempo (PRN + Welch), que produzia artefatos e piso de ruído alto
em banda larga.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dask import delayed
from radioskysims_core.domain.common import FrequencyRange
from radioskysims_core.domain.common.logging import get_domain_logger

from radioskysims_sources.domain.signal.models import SatSignal
from radioskysims_sources.domain.signal.ports import (
    SignalPsdPort,
    SignalRepositoryPort,
)

logger = get_domain_logger(__name__)


class SignalFactory:
    """
    Gera espectros de densidade de potência (PSD) para sinais GNSS.

    Usa PSDs analíticas (closed-form) de BPSK/QPSK/BOC/CBOC — compartilhadas
    com os satélites RINEX. Sem PRN nem Welch: determinístico, suave e sem
    piso de ruído em banda larga.

    Características:
    - Stateless (exceto FrequencyRange configurado)
    - Lazy evaluation usando dask.delayed
    - Independente de xarray, tracks e satélites

    Attributes:
        fr (FrequencyRange): Intervalo de frequências para geração de PSD.
    """

    def __init__(
        self,
        frequency_range: FrequencyRange,
        signal_repo: SignalRepositoryPort,
        psd_provider: SignalPsdPort | None = None,
    ) -> None:
        """
        Parameters
        ----------
        frequency_range : FrequencyRange
            Common frequency grid for all generated PSDs.
        signal_repo : SignalRepositoryPort
            Repository providing signal definitions (infrastructure dependency).
        psd_provider : SignalPsdPort | None
            Porta de PSDs analíticas. Deve ser injetada pelo composition root
            (ex.: adapter ``GnsstoolsAnalyticPsdProvider``); o domínio não
            conhece adapters de infraestrutura.
        """

        self._fr: FrequencyRange = frequency_range
        self.signal_repo = signal_repo
        self._psd_provider: SignalPsdPort | None = psd_provider

    def _psd(self) -> SignalPsdPort:
        """Retorna a porta de PSDs injetada (via composition root)."""
        if self._psd_provider is None:
            msg = (
                "psd_provider não injetado no SignalFactory — configure no "
                "composition root (ex.: GnsstoolsAnalyticPsdProvider)."
            )
            raise ValueError(msg)
        return self._psd_provider

    @property
    def fr(self) -> FrequencyRange:
        """Intervalo de frequências configurado."""
        return self._fr

    @fr.setter
    def fr(self, value: FrequencyRange) -> None:
        self._fr = value

    # =========================================================
    # Frequency grid (contrato com o Assembler)
    # =========================================================
    def frequency_grid_hz(self) -> npt.NDArray[np.floating]:
        """
        This grid defines the contract with downstream assemblers (TOD, MAP).
        Gera o grid de frequências comum em Hz.

        Returns:
            npt.NDArray[np.floating]: Array de frequências em Hertz.
        """
        return np.asarray(self.fr.value, dtype=float) * 1e6

    # =========================================================
    # PRN + moduladores (privados)
    # =========================================================
    # PSDs ANALÍTICAS (closed-form) — sem PRN nem Welch. O mesmo código
    # (gnsstools4astro.domain.analytic_psd) é usado pelos satélites RINEX.

    # =========================================================
    # Core: simulate PSD
    # =========================================================
    @delayed
    def _simulate_psd(
        self,
        *,
        modulation: str,
        fc_hz: float,
    ) -> npt.NDArray[np.floating]:
        """
        PSD analítica (dB) da modulação avaliada no grid da FrequencyRange.

        Usa a expressão fechada (BPSK/BOC/CBOC) — determinística e sem piso
        de ruído de Welch em banda larga.
        """
        provider = self._psd()
        norm = provider.analytic_psd(modulation, self.frequency_grid_hz(), fc_hz)
        return provider.psd_db(norm).astype(np.float64)

    # =========================================================
    # Public API
    # =========================================================
    def _build(
        self,
        *,
        modulation: str,
        fc_hz: float,
    ) -> delayed:
        """
        Return a lazy computation that evaluates PSD(freq).

        This method does not trigger computation.
        """
        return self._simulate_psd(
            modulation=modulation,
            fc_hz=fc_hz,
        )

    def build(self, sats=None):
        """
        Build spectral signal models for a collection of satellites.

        Parameters
        ----------
        sats : dict-like
            Mapping containing satellite blocks with a 'satellites' entry.

        Returns
        -------
        dict[int, list[SatSignal]]
            Mapping from NORAD ID to a list of satellite signal models.

        """
        try:
            sat_ids = next(item["satellites"] for item in list(sats.values()))
            norad_ids = [sat.norad_id for sat in sat_ids]

            logger.info(
                "signal_build_start",
                num_satellites=len(norad_ids),
                frequency_range_mhz=f"{self.fr.min}-{self.fr.max}",
            )

            signals = self.signal_repo.get_signals_by_norad_ids(norad_ids)
            sat_signals = {}
            for norad_id, _signals in signals.items():
                PSDs = []
                _sat_signals = []
                for signal in _signals:
                    _PSD = self._build(
                        modulation=signal.modulation,
                        fc_hz=signal.frequency * 1e6,
                    )
                    PSDs.append(_PSD)

                    _signal_obj = SatSignal(signal=signal, psd=_PSD)
                    _sat_signals.append(_signal_obj)

                sat_signals[norad_id] = _sat_signals
                logger.info(
                    "signal_created",
                    norad_id=norad_id,
                    num_signals=len(_sat_signals),
                )

            logger.info("signal_build_success", num_satellites=len(sat_signals))
            return sat_signals
        except Exception as e:
            logger.exception("signal_build_failed", error=str(e))
            raise
