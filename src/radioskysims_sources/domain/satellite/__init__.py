# domain/satellite/__init__.py
from __future__ import annotations

from importlib import import_module

from .models import Satellite, SatelliteAssembled
from .prn import (
    NORAD_TO_PRN,
    SV_PREFIX,
    norad_to_prn,
    parse_prn_from_name,
    prn_to_sv,
    sv_to_prn,
)
from .selector import TLESelector

__all__ = [
    "Satellite",
    "SatelliteAssembled",
    "SatelliteAssembler",
    "SatelliteFactory",
    "TLESelector",
    # Mapeamento NORAD ↔ Nome ↔ PRN (ponte TLE ↔ RINEX)
    "SV_PREFIX",
    "NORAD_TO_PRN",
    "parse_prn_from_name",
    "norad_to_prn",
    "prn_to_sv",
    "sv_to_prn",
]


def __getattr__(name: str):
    if name == "SatelliteAssembler":
        return import_module(".assembler", __name__).SatelliteAssembler
    if name == "SatelliteFactory":
        return import_module(".factory", __name__).SatelliteFactory
    if name == "TLESelector":
        return import_module(".selector", __name__).TLESelector
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
