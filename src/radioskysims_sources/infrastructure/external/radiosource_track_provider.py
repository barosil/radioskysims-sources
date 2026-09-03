"""
Módulo: src/radioskysims/infrastructure/external/radiosource_track_provider.py

Adapter para cálculo de tracks de fontes de rádio fixas (coordenadas
celestes RA, Dec) usando Skyfield, convertendo para o referencial
alt-az de um observador terrestre ao longo do tempo.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from astropy.time import Time
from skyfield.api import Star

from radioskysims_core.domain.common import Observer, TimeRange
from radioskysims_sources.application.ports.radiosources import RadioSourceTrackProviderPort
from radioskysims_sources.domain.radiosources.models import RadioSource
from radioskysims_sources.domain.radiosources.tracks import RadioSourceTrack
from radioskysims_core.skyfield_loader import load_ephemeris

log = logging.getLogger(__name__)


class SkyfieldRadioSourceTrackProvider(RadioSourceTrackProviderPort):
    """Provider de tracks para fontes de rádio fixas usando efemérides Skyfield.

    Para cada RadioSource (RA, Dec fixos), calcula a posição aparente
    (altitude, azimute, theta, distância) no referencial do observador
    ao longo de um intervalo temporal.
    """

    def __init__(self) -> None:
        self._eph = load_ephemeris()

    @staticmethod
    def _jd_to_iso(jd: float) -> str:
        """Converte JD para string ISO via astropy."""
        from astropy.time import Time as AstroTime

        return str(AstroTime(jd, format="jd").iso)

    def compute_track(
        self,
        sources: list[RadioSource],
        observer: Observer,
        time_range: TimeRange,
    ) -> RadioSourceTrack:
        """Calcula tracks para uma lista de fontes no referencial do observador.

        Args:
            sources: Lista de RadioSource com coordenadas RA/Dec em graus.
            observer: Observador terrestre (lat, lon, altura).
            time_range: Intervalo temporal da simulação.

        Returns:
            RadioSourceTrack com dims (source, time) e variáveis
            theta_rad, az_rad, el_rad, range_m.
        """
        if not sources:
            # Cria dataset vazio MAS com as dimensões (source, time) para
            # que operações como .sizes['time'] não quebrem.
            time_coord = np.array(
                Time(
                    time_range.to_skyfield().tt, format="jd", scale="tt"
                ).utc.datetime64
            )
            return RadioSourceTrack(
                dataset=xr.Dataset(
                    coords={"source": [], "time": time_coord[:0]},
                ),
                source_ids=[],
                time_range=(
                    self._jd_to_iso(time_range.start_jd),
                    self._jd_to_iso(time_range.end_jd),
                ),
            )

        obs_loc, _ = observer.to_skyfield()
        times = time_range.to_skyfield()
        earth = self._eph["earth"]

        theta_list: list[np.ndarray] = []
        az_list: list[np.ndarray] = []
        el_list: list[np.ndarray] = []
        range_list: list[np.ndarray] = []
        source_ids: list[str] = []

        for src in sources:
            # Cria objeto Star com as coordenadas da fonte
            star = Star(
                ra_hours=src.ra_deg / 15.0,
                dec_degrees=src.dec_deg,
            )

            # Posição aparente no referencial do observador
            apparent = (earth + obs_loc).at(times).observe(star).apparent()
            alt, az, distance = apparent.altaz()

            theta = np.pi / 2.0 - alt.radians
            theta_list.append(np.asarray(theta, dtype=float))
            az_list.append(np.asarray(az.radians, dtype=float))
            el_list.append(np.asarray(alt.radians, dtype=float))
            range_list.append(np.asarray(distance.m, dtype=float))
            source_ids.append(src.id)

        time_coord = np.array(Time(times.tt, format="jd", scale="tt").utc.datetime64)

        ds = xr.Dataset(
            data_vars={
                "theta_rad": (["source", "time"], np.stack(theta_list, axis=0)),
                "az_rad": (["source", "time"], np.stack(az_list, axis=0)),
                "el_rad": (["source", "time"], np.stack(el_list, axis=0)),
                "range_m": (["source", "time"], np.stack(range_list, axis=0)),
            },
            coords={
                "source": source_ids,
                "time": time_coord,
            },
            attrs={"provider": "skyfield"},
        )

        t0 = str(time_coord[0]) if len(time_coord) > 0 else ""
        t1 = str(time_coord[-1]) if len(time_coord) > 0 else ""

        return RadioSourceTrack(
            dataset=ds,
            source_ids=source_ids,
            time_range=(t0, t1),
        )
