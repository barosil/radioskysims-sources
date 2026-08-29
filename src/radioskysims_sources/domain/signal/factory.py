"""
Módulo: src/radioskysims/domain/signal/factory.py

Domain factory for spectral emission models.

This module defines a factory for generating GNSS-like
power spectral densities (PSD) as a function of frequency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dask import delayed
from scipy.signal import welch

from radioskysims_core.domain.common import FrequencyRange
from radioskysims_core.domain.common.logging import get_domain_logger
from radioskysims_sources.domain.signal.models import SatSignal
from radioskysims_sources.domain.signal.ports import SignalRepositoryPort

logger = get_domain_logger(__name__)


class SignalFactory:
    """
    Gera espectros de densidade de potência (PSD) para sinais GNSS.

    Implementa geração de sinais GPS/Galileo-like com diferentes modulações:
    - BPSK (Binary Phase Shift Keying)
    - QPSK (Quadrature Phase Shift Keying)
    - BOC (Binary Offset Carrier)
    - CBOC (Composite Binary Offset Carrier)

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
    ) -> None:
        """
        Parameters
        ----------
        frequency_range : FrequencyRange
            Common frequency grid for all generated PSDs.
        signal_repo : SignalRepositoryPort
            Repository providing signal definitions (infrastructure dependency).
        """

        self._fr: FrequencyRange = frequency_range
        self.signal_repo = signal_repo

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
    def _generate_prn(self, length: int) -> npt.NDArray[np.floating]:
        """
        Gera sequência pseudo-aleatória (PRN).

        Args:
            length (int): Comprimento da sequência.

        Returns:
            npt.NDArray[np.floating]: Array com valores -1 ou +1.
        """
        rng = np.random.default_rng()
        return rng.choice([-1.0, 1.0], size=length)

    def _bpsk(self, chips: npt.NDArray[np.floating], sps: int) -> npt.NDArray[np.floating]:
        """
        Modulação BPSK (Binary Phase Shift Keying).

        Args:
            chips (npt.NDArray[np.floating]): Sequência de chips.
            sps (int): Samples per symbol.

        Returns:
            npt.NDArray[np.floating]: Sinal modulado.
        """
        return np.repeat(chips, sps)

    def _qpsk(self, chips: npt.NDArray[np.floating], sps: int) -> npt.NDArray[np.complexfloating]:
        """
        Modulação QPSK (Quadrature Phase Shift Keying).

        Args:
            chips (npt.NDArray[np.floating]): Sequência de chips.
            sps (int): Samples per symbol.

        Returns:
            npt.NDArray[np.complexfloating]: Sinal complexo modulado.
        """
        half = chips.size // 2
        i = np.repeat(chips[:half], sps)
        q = np.repeat(chips[half:], sps)
        return i + 1j * q

    def _boc(self, chips: npt.NDArray[np.floating], sps: int, m: float) -> npt.NDArray[np.floating]:
        """
        Modulação BOC (Binary Offset Carrier).

        Args:
            chips (npt.NDArray[np.floating]): Sequência de chips.
            sps (int): Samples per symbol.
            m (float): Parâmetro de modulação BOC.

        Returns:
            npt.NDArray[np.floating]: Sinal modulado BOC.
        """
        base = np.repeat(chips, sps)
        sc = np.sign(np.sin(2 * np.pi * m * np.arange(base.size) / sps))
        return base * sc

    def _cboc(self, chips: npt.NDArray[np.floating], sps: int) -> npt.NDArray[np.floating]:
        """
        Modulação CBOC (Composite Binary Offset Carrier).

        Combina BOC(1,1) e BOC(6,1) com pesos específicos.

        Args:
            chips (npt.NDArray[np.floating]): Sequência de chips.
            sps (int): Samples per symbol.

        Returns:
            npt.NDArray[np.floating]: Sinal modulado CBOC.
        """
        return np.sqrt(10 / 11) * self._boc(chips, sps, m=1) + np.sqrt(1 / 11) * self._boc(
            chips, sps, m=6
        )

    # =========================================================
    # Core: simulate PSD
    # =========================================================
    @delayed
    def _simulate_psd(
        self,
        *,
        modulation: str,
        fc_hz: float,
        prn_len: int,
        fs_factor: int,
    ) -> npt.NDArray[np.floating]:
        """
        Retorna PSD(freq) interpolada no grid da FrequencyRange.
        """
        # -----------------------------------------------------
        # Escolha da modulação
        # -----------------------------------------------------
        # Chip rates fracionários (ex.: "BPSK(0.5)" → 0.5 × 1.023 MHz)
        # exigem parsing numérico em float, não int.
        if modulation.startswith("BPSK"):
            n = float(modulation.split("(")[1].split(")")[0])
            chip_rate = n * 1.023e6
            chips = self._generate_prn(prn_len)
            sig = self._bpsk(chips, fs_factor)

        elif modulation.startswith("QPSK"):
            n = float(modulation.split("(")[1].split(")")[0])
            chip_rate = n * 1.023e6
            chips = self._generate_prn(2 * prn_len)
            sig = self._qpsk(chips, fs_factor)

        elif modulation.startswith("BOC"):
            m, n = map(
                float,
                modulation.split("(")[1].split(")")[0].split(","),
            )
            chip_rate = n * 1.023e6
            chips = self._generate_prn(prn_len)
            sig = self._boc(chips, fs_factor, m=m)

        elif modulation.startswith("CBOC"):
            chip_rate = 1.023e6
            chips = self._generate_prn(prn_len)
            sig = self._cboc(chips, fs_factor)

        else:
            msg = f"Modulação {modulation} não suportada"
            raise NotImplementedError(msg)

        # -----------------------------------------------------
        # PSD (Welch)
        # -----------------------------------------------------
        fs = chip_rate * fs_factor
        f, Pxx = welch(
            sig,
            fs=fs,
            nperseg=4096,
            return_onesided=False,
        )

        f = np.fft.fftshift(f) + fc_hz
        Pxx = np.fft.fftshift(Pxx)
        Pxx_db = 10.0 * np.log10(Pxx + 1e-30)

        # -----------------------------------------------------
        # Interpolação no grid comum
        # -----------------------------------------------------
        f_grid = self.frequency_grid_hz()
        psd_interp = np.interp(f_grid, f, Pxx_db)

        return psd_interp.astype(np.float64)

    # =========================================================
    # Public API
    # =========================================================
    def _build(
        self,
        *,
        modulation: str,
        fc_hz: float,
        prn_len: int = 1023,
        fs_factor: int = 10,
    ) -> delayed:
        """
        Return a lazy computation that evaluates PSD(freq).

        This method does not trigger computation.
        """
        return self._simulate_psd(
            modulation=modulation,
            fc_hz=fc_hz,
            prn_len=prn_len,
            fs_factor=fs_factor,
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
