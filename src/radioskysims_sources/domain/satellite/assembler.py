from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr

from .models import Satellite, SatelliteAssembled


class SatelliteAssembler:
    """
    Boundary entre o domínio cinemático (tracks)
    e o domínio espectral (signals).

    Gera sinais de forma lazy a partir dos tracks,
    delegando a física espectral à SignalFactory.
    """

    @staticmethod
    def assemble(params: list[dict]) -> list[Satellite]:
        """
        Assembla uma lista de instâncias Satellite a partir de parâmetros.

        Método de fábrica que cria Satellite objects a partir de dicionários
        de parâmetros. Implementa tolerância a falhas: se um satélite falhar
        na construção, é silenciosamente ignorado.

        Pre-condições:
        - params: lista de dicts com argumentos válidos para Satellite.__init__

        Comportamento:
        - Entrada válida: retorna list[Satellite] com objetos construídos
        - Entrada inválida (ValueError na construção): ignora e continua
        - Lista vazia: retorna []
        - Nenhum satélite construído com sucesso: retorna []

        Args:
            params: Lista de dicionários, cada um contendo argumentos
                   para construir um Satellite (ex: {"norad_id": 25544})

        Returns:
            list[Satellite]: Satélites construídos com sucesso.
                            Lista vazia se nenhum conseguir ser construído.

        Exemplo:
            >>> params = [{"norad_id": 1}, {"norad_id": 2}]
            >>> sats = SatelliteAssembler.assemble(params)
            >>> len(sats)
            2
        """
        satellites = []
        for param in params:
            try:
                sat = Satellite(**param)
                satellites.append(sat)
            except ValueError:
                # Ignora satélites que falharam na construção
                continue
        return satellites

    def __init__(self, signal_factory, track_factory) -> None:
        self.signal_factory = signal_factory
        self.track_factory = track_factory

    def build_signals(
        self, sats, recalculate=False, sats_assembled=None, frequency_range=None
    ) -> SatelliteAssembled:
        """
        Constrói um SatelliteAssembled com sinais lazy no domínio da frequência.

        Parameters
        ----------
        sats : dict
            Blocos de satélites para construir tracks/signals.
        recalculate : bool
            Se True, recalcula sinais com novo frequency_range.
        sats_assembled : SatelliteAssembled | None
            SatelliteAssembled existente para reuso de tracks.
        frequency_range : FrequencyRange | None
            Novo intervalo de frequência (usado com recalculate=True).

        Returns
        -------
        SatelliteAssembled
        """
        if recalculate:
            tracks = sats_assembled.tracks if sats_assembled else None
            self.signal_factory.fr = frequency_range
        else:
            tracks = self.track_factory.build(sats).dataset

        sat_ids = tracks.satellite.values
        names = tracks.names.values.tolist()
        signals = self.signal_factory.build(sats)

        freq = self._frequency_coord()
        psd_blocks, fc_values, mod_values = self._build_psd_blocks(
            signals, sat_ids, freq
        )

        return self._build_signal_assembled(
            psd_blocks, sat_ids, fc_values, mod_values, freq, names, tracks
        )

    def _build_psd_blocks(self, signals, sat_ids, freq):
        """Constrói blocos de PSD lazy (dask delayed) por satélite."""
        psd_blocks = []
        fc_values = []
        mod_values = []

        for sat in sat_ids:
            sig_objs = signals[int(sat)]
            psd_arrays = [
                da.from_delayed(sig.psd, shape=(freq.size,), dtype=np.float64)
                for sig in sig_objs
            ]
            psd_blocks.append(da.stack(psd_arrays, axis=0))
            fc_values.append([sig.signal.frequency for sig in sig_objs])
            mod_values.append([sig.signal.modulation for sig in sig_objs])

        return psd_blocks, fc_values, mod_values

    def _build_signal_assembled(
        self, psd_blocks, sat_ids, fc_values, mod_values, freq, names, tracks
    ) -> SatelliteAssembled:
        """Monta o dataset de sinais e retorna SatelliteAssembled."""
        signals_list = []
        for psd, sat, fc, mod in zip(
            psd_blocks, sat_ids, fc_values, mod_values, strict=False
        ):
            signal = xr.Dataset(
                data_vars={
                    "PSD": (("satellite", "freq"), psd),
                },
                coords={
                    "satellite": sat * np.ones(psd.shape[0], dtype=int),
                    "freq": freq,
                    "fc": ("satellite", fc),
                    "modulation": ("satellite", mod),
                },
                attrs={"domain": "frequency", "units": "dB/Hz"},
            )
            signals_list.append(signal)

        signals = xr.concat(signals_list, dim="satellite")

        return SatelliteAssembled(name=names, tracks=tracks, signals=signals)

    def _frequency_coord(self) -> xr.DataArray:
        """
        Grade de frequência em MHz.
        """
        freq_hz = self.signal_factory.frequency_grid_hz()
        freq_mhz = freq_hz * 1e-6
        return xr.DataArray(freq_mhz, dims="freq", name="freq")
