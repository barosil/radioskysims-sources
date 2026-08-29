# domain/satellite/__init__.py
from __future__ import annotations

from importlib import import_module

from .models import Satellite, SatelliteAssembled
from .selector import TLESelector

__all__ = [
    "Satellite",
    "SatelliteAssembled",
    "SatelliteAssembler",
    "SatelliteFactory",
    "TLESelector",
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
