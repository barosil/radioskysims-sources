"""
Módulo: tests/domain/radiosources/test_selection.py

Descrição:
    Testes para a seleção de radiofontes por proximidade angular ao pointing
    do telescópio (domain/radiosources/selection.py).

Copyright BINGO Collaboration
Last modified: 2026-08-15
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from radioskysims_sources.domain.radiosources.models import RadioSource, SpectralModel
from radioskysims_sources.domain.radiosources.selection import (
    angular_separation_deg,
    min_separation_to_pointing,
    pointing_region,
    select_sources_in_beam,
)


def _src(src_id: str, ra_deg: float, dec_deg: float) -> RadioSource:
    return RadioSource(
        id=src_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        spectral_model=SpectralModel(nu0_mhz=1400.0, s0_jy=1.0, alpha=0.75),
    )


def _pointing_track(ra_deg, dec_deg) -> xr.Dataset:
    """Cria pointing track com ra/dec em GRAUS (formato do projeto).

    ``compute_pointing_track`` já fornece RA/Dec em graus com
    ``units="deg"`` — o filtro angular do feixe não aplica ``rad2deg``
    (aplicar inflaria os valores em ~57×).
    """
    return xr.Dataset(
        data_vars={
            "ra": ("time", np.asarray(ra_deg, dtype=float)),
            "dec": ("time", np.asarray(dec_deg, dtype=float)),
        },
        coords={"time": np.arange(len(ra_deg))},
    )


class TestAngularSeparation:
    def test_zero_separation(self):
        assert angular_separation_deg(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)

    def test_equator_ra_separation(self):
        # 90° em RA no equador → 90° de separação
        sep = angular_separation_deg(0.0, 0.0, 90.0, 0.0)
        assert sep == pytest.approx(90.0)

    def test_pole_to_any_point(self):
        # Do polo norte para qualquer ponto do equador → 90°
        sep = angular_separation_deg(0.0, 90.0, 120.0, 0.0)
        assert sep == pytest.approx(90.0)

    def test_ra_wrap(self):
        # RA 359° e 1° (no equador) → 2° de separação (wrap natural)
        sep = angular_separation_deg(359.0, 0.0, 1.0, 0.0)
        assert sep == pytest.approx(2.0, abs=1e-6)

    def test_vectorized(self):
        seps = angular_separation_deg(
            np.array([0.0, 10.0]), np.array([0.0, 0.0]), 90.0, 0.0
        )
        assert seps[0] == pytest.approx(90.0)


class TestMinSeparationToPointing:
    def test_fixed_pointing(self):
        # No equador, 2° de diferença em RA = 2° de arco de círculo máximo
        sep = min_separation_to_pointing(
            12.0, 0.0, pointing_ra_deg=10.0, pointing_dec_deg=0.0
        )
        assert sep == pytest.approx(2.0, abs=1e-6)

    def test_fixed_pointing_offset_dec(self):
        # Fora do equador, a separação de arco difere da diferença em RA
        sep = min_separation_to_pointing(
            12.0, -5.0, pointing_ra_deg=10.0, pointing_dec_deg=-5.0
        )
        assert 0.0 < sep < 2.0

    def test_temporal_pointing_min_over_time(self):
        # Fonte em (ra=30, dec=0); pointing varre RA de 0 a 60 → mínimo em 30 → 0°
        track = _pointing_track([0.0, 15.0, 30.0, 45.0, 60.0], [0.0] * 5)
        sep = min_separation_to_pointing(30.0, 0.0, pointing_track=track)
        assert sep == pytest.approx(0.0, abs=1e-6)

    def test_no_pointing(self):
        assert min_separation_to_pointing(10.0, 10.0) is None


class TestSelectSourcesInBeam:
    def _sources(self):
        return [
            _src("near", 10.0, 0.0),
            _src("far", 80.0, 0.0),
        ]

    def test_fixed_pointing_fwhm(self):
        kept, seps = select_sources_in_beam(
            self._sources(),
            pointing_ra_deg=10.0,
            pointing_dec_deg=0.0,
            fwhm_deg=5.0,
            factor=3.0,  # limiar = 15°
        )
        assert [s.id for s in kept] == ["near"]
        assert seps[0] == pytest.approx(0.0)

    def test_temporal_pointing_keeps_sources_along_path(self):
        track = _pointing_track([0.0, 10.0, 20.0], [0.0] * 3)
        srcs = [_src("on_path", 10.0, 0.0), _src("off_path", 80.0, 0.0)]
        kept, _ = select_sources_in_beam(
            srcs, pointing_track=track, fwhm_deg=5.0, factor=3.0
        )
        assert [s.id for s in kept] == ["on_path"]

    def test_radius_overrides_fwhm(self):
        kept, _ = select_sources_in_beam(
            self._sources(),
            pointing_ra_deg=10.0,
            pointing_dec_deg=0.0,
            fwhm_deg=5.0,
            factor=3.0,
            radius_deg=2.0,
        )
        # 'far' a 70° também é descartado; 'near' a 0° mantido
        assert [s.id for s in kept] == ["near"]

    def test_no_pointing_keeps_all(self):
        kept, seps = select_sources_in_beam(
            self._sources(), fwhm_deg=5.0, factor=3.0
        )
        assert len(kept) == 2
        assert all(s is None for s in seps)

    def test_empty_input(self):
        kept, seps = select_sources_in_beam([], pointing_ra_deg=10.0, pointing_dec_deg=0.0)
        assert kept == []
        assert seps == []


class TestPointingRegion:
    def test_fixed_pointing(self):
        ra, dec, radius = pointing_region(
            pointing_ra_deg=45.0, pointing_dec_deg=-10.0
        )
        assert (ra, dec) == pytest.approx((45.0, -10.0))
        assert radius == pytest.approx(0.0)

    def test_temporal_coverage_radius(self):
        # Pontos espalhados em RA; o raio deve cobrir todos
        track = _pointing_track([0.0, 30.0, 60.0], [0.0] * 3)
        ra, dec, radius = pointing_region(pointing_track=track)
        for r, d in zip([0.0, 30.0, 60.0], [0.0] * 3, strict=False):
            sep = angular_separation_deg(ra, dec, r, d)
            assert sep <= radius + 1e-9

    def test_no_pointing(self):
        assert pointing_region() is None
