"""\
Módulo: src/radioskysims/infrastructure/external/local_source_track_skyfield_provider.py

Descrição:
    Adapter externo para cálculo de tracks de fontes locais (MVP: Sun)
    usando Skyfield no referencial do observador terrestre.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from astropy.time import Time

from radioskysims_core.domain.common import Observer, TimeRange
from radioskysims_sources.domain.localsources.models import LocalSourceTrack
from radioskysims_core.skyfield_loader import load_ephemeris
from radioskysims_sources.infrastructure.ports.localsources import LocalSourceTrackProviderPort


class SkyfieldLocalSourceTrackProvider(LocalSourceTrackProviderPort):
    """Provider de tracks para fontes locais usando efemérides Skyfield."""

    def __init__(self) -> None:
        self._eph = load_ephemeris()

    def compute_track(
        self,
        source_name: str,
        observer: Observer,
        time_range: TimeRange,
    ) -> LocalSourceTrack:
        source_key = source_name.strip().lower()
        if source_key != "sun":
            msg = (
                f"Fonte local '{source_name}' ainda não suportada. "
                "Use 'Sun' no MVP atual."
            )
            raise NotImplementedError(msg)

        obs_loc, _ = observer.to_skyfield()
        times = time_range.to_skyfield()

        earth = self._eph["earth"]
        sun = self._eph["sun"]

        app = (earth + obs_loc).at(times).observe(sun).apparent()
        alt, az, distance = app.altaz()

        theta = np.pi / 2.0 - alt.radians
        az_rad = az.radians
        el_rad = alt.radians
        range_m = distance.m

        time_coord = np.array(Time(times.tt, format="jd", scale="tt").utc.datetime64)

        ds = xr.Dataset(
            data_vars={
                "theta_rad": ("time", np.asarray(theta, dtype=float)),
                "az_rad": ("time", np.asarray(az_rad, dtype=float)),
                "el_rad": ("time", np.asarray(el_rad, dtype=float)),
                "range_m": ("time", np.asarray(range_m, dtype=float)),
            },
            coords={"time": time_coord},
            attrs={"provider": "skyfield", "source": source_name},
        )

        return LocalSourceTrack(source_name=source_name, dataset=ds)
