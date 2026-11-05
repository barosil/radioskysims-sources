from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from radioskysims_core.domain.common.models import TimeRange
from radioskysims_sources.domain.satellite.models import Satellite
from radioskysims_sources.infrastructure.db.satellite.models import OrbitalElement

TLE_LINE1 = "1 25544U 98067A   21275.51857639  .00002182  00000-0  50362-4 0  9991"
TLE_LINE2 = "2 25544  51.6443  21.4184 0004295  35.2895 324.8335 15.48942522302788"


@pytest.mark.xfail(
    reason="load_ephemeris() não implementado em Satellite core. "
    "Função de cálculo de efemérides aguarda integração com bibliotecas "
    "de propagação orbital (Skyfield, Astropy, etc.)",
    strict=False,
)
def test_satellite_ephemeris_generation():
    tr = TimeRange.from_iso("2021-09-01T00:00:00", "2021-09-01T00:10:00", resolution=60)

    # Criar elemento orbital para usar no satélite
    element = OrbitalElement(
        norad_id=25544,
        object_name="ISS",
        epoch=datetime(2021, 9, 1, 0, 0, 0),
        tle_line1=TLE_LINE1,
        tle_line2=TLE_LINE2,
    )

    # Criar satélite com os novos campos obrigatórios
    sat = Satellite(
        norad_id=25544,
        name="ISS",
        epoch=datetime(2021, 9, 1, 0, 0, 0),
        element=element,
    )
    eph = sat.load_ephemeris(tr)

    assert eph.positions.shape[0] == 3
    assert eph.positions.shape[1] == len(tr.value)
    assert np.allclose(eph.positions.attrs["units"], "km")
