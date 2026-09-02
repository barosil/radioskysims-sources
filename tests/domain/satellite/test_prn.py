"""\
Módulo: tests/domain/satellite/test_prn.py

Descrição:
    Testes do mapeamento NORAD ↔ Nome ↔ PRN (ponte TLE ↔ RINEX):
    - ``parse_prn_from_name`` (CelesTrak);
    - ``prn_to_sv`` / ``sv_to_prn`` (esquema RINEX);
    - ``norad_to_prn`` (mapa canônico + derivação do nome).

Copyright BINGO Collaboration
"""

from __future__ import annotations

import pytest
from radioskysims_sources.domain.satellite.prn import (
    NORAD_TO_PRN,
    norad_to_prn,
    parse_prn_from_name,
    prn_to_sv,
    sv_to_prn,
)


# ----------------------------------------------------------------------
# parse_prn_from_name
# ----------------------------------------------------------------------
class TestParsePrnFromName:
    @pytest.mark.parametrize(
        ("name", "cons", "expected"),
        [
            # GPS — "(PRN NN)"
            ("GPS BIIR-5 (PRN 22)", "GPS", 22),
            ("GPS BIII-7 (PRN 01)", "GPS", 1),
            ("GPS BIIF-12 (PRN 32)", "GPS", 32),
            # BeiDou — "(C NN)"
            ("BEIDOU-3 M25 (C47)", "BEIDOU", 47),
            ("BEIDOU-3 G1 (C19)", "BEIDOU", 19),
            # SBAS/outros com "(PRN NNN)"
            ("GSAT-8 (GAGAN/PRN 127)", "GALILEO", 127),
            ("EUTELSAT 117 WEST B (WAAS/PRN 131)", "EUTELSAT", 131),
            ("QZS-2 (QZSS/PRN 194)", "QZS", 194),
            # GLONASS — número Glonass "(7xx)"
            ("COSMOS 2433 (720)", "GLONASS", 720),
            # GALILEO — número do veículo "(GALILEO N)"
            ("GSAT0201 (GALILEO 5)", "GALILEO", 5),
            ("GSAT0225 (GALILEO 29)", "GALILEO", 29),
        ],
    )
    def test_parse_valid(self, name, cons, expected):
        assert parse_prn_from_name(name, cons) == expected

    @pytest.mark.parametrize(
        ("name", "cons"),
        [
            # Sem PRN no nome
            ("GSAT0101 (GALILEO-PFM)", "GALILEO"),
            ("IRNSS-1A", "IRNSS"),
            # Nome vazio/None
            (None, "GPS"),
            ("", "GPS"),
            # Nome sem padrão conhecido
            ("UNKNOWN SATELLITE", "GPS"),
        ],
    )
    def test_parse_unknown(self, name, cons):
        assert parse_prn_from_name(name, cons) is None


# ----------------------------------------------------------------------
# prn_to_sv / sv_to_prn (ponte RINEX)
# ----------------------------------------------------------------------
class TestSvBridge:
    @pytest.mark.parametrize(
        ("prn", "cons", "sv"),
        [
            (22, "GPS", "G22"),
            (1, "GPS", "G01"),
            (5, "GALILEO", "E05"),
            (7, "GLONASS", "R07"),
            (47, "BEIDOU", "C47"),
        ],
    )
    def test_prn_to_sv(self, prn, cons, sv):
        assert prn_to_sv(prn, cons) == sv

    def test_prn_to_sv_unknown_constellation(self):
        with pytest.raises(ValueError):
            prn_to_sv(1, "STARLINK")

    @pytest.mark.parametrize(
        ("sv", "cons", "prn"),
        [
            ("G22", "GPS", 22),
            ("G01", "GPS", 1),
            ("E05", "GALILEO", 5),
            ("R07", "GLONASS", 7),
            ("C47", "BEIDOU", 47),
        ],
    )
    def test_sv_to_prn(self, sv, cons, prn):
        assert sv_to_prn(sv) == (cons, prn)

    @pytest.mark.parametrize("sv", ["X99", "G", "", "G0X", None])
    def test_sv_to_prn_invalid(self, sv):
        assert sv_to_prn(sv) is None


# ----------------------------------------------------------------------
# norad_to_prn
# ----------------------------------------------------------------------
class TestNoradToPrn:
    def test_uses_canonical_override_first(self, monkeypatch):
        monkeypatch.setitem(NORAD_TO_PRN, 99999, 42)
        assert norad_to_prn(99999, name="irrelevante") == 42

    def test_falls_back_to_name(self):
        # GPS real do catálogo: NORAD 26407 = GPS BIIR-5 (PRN 22)
        assert norad_to_prn(26407, name="GPS BIIR-5 (PRN 22)", constellation="GPS") == 22

    def test_unknown_returns_none(self):
        assert norad_to_prn(12345, name="NOME GENÉRICO", constellation="GPS") is None
