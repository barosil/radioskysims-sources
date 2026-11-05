from __future__ import annotations

import importlib

from radioskysims_sources.infrastructure.db.satellite.models import Signal


def test_signal_module_reexports_persistence_model():
    module = importlib.import_module("radioskysims_sources.infrastructure.db.signal.models")
    assert module.Signal is Signal
    assert module.__all__ == ["Signal"]


def test_satellite_package_resolves_lazy_exports():
    module = importlib.import_module("radioskysims_sources.domain.satellite")
    for name in ("SatelliteAssembler", "SatelliteFactory", "TLESelector"):
        module.__dict__.pop(name, None)
        assert getattr(module, name).__name__ == name


def test_satellite_package_rejects_unknown_export():
    module = importlib.import_module("radioskysims_sources.domain.satellite")

    try:
        module.__getattr__("MissingExport")
    except AttributeError as error:
        assert "MissingExport" in str(error)
    else:
        message = "unknown export should raise AttributeError"
        raise AssertionError(message)
