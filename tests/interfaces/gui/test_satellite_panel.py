from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from radioskysims_sources.interfaces.gui.satellite_panel import SatellitePanel


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _satellite(identifier: int):
    return SimpleNamespace(
        id=identifier,
        coords=SimpleNamespace(
            data={"ra": SimpleNamespace(values=np.array([0.1, 0.3]))}
        ),
    )


class _Factory:
    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = []

    def create_from_norad_ids(self, ids, epoch):
        self.calls.append(("ids", ids, epoch))
        if self.error:
            raise self.error
        return self.result

    def create_from_constellations(self, names, epoch):
        self.calls.append(("constellations", names, epoch))
        return self.result


def test_panel_submit_by_norad_updates_output():
    factory = _Factory(result=[_satellite(123)])
    panel = SatellitePanel(factory)
    panel.norad_input.value = "123\n456\n"

    panel._on_submit(None)

    assert factory.calls[0][0:2] == ("ids", [123, 456])
    assert "Satélites encontrados" in panel.output.object
    assert "**123**" in panel.output.object


def test_panel_submit_by_constellation_updates_output():
    factory = _Factory(result=[_satellite(7)])
    panel = SatellitePanel(factory)
    panel.constellation_input.value = ["GPS"]

    panel._on_submit(None)

    assert factory.calls[0][0:2] == ("constellations", ["GPS"])
    assert "**7**" in panel.output.object


def test_panel_submit_without_selection_requests_input():
    panel = SatellitePanel(_Factory())

    panel._on_submit(None)

    assert panel.output.object == "Forneça NORAD IDs ou constelações."


def test_panel_submit_reports_factory_error_and_show_returns_panel():
    panel = SatellitePanel(_Factory(error=RuntimeError("backend offline")))
    panel.norad_input.value = "1"

    panel._on_submit(None)

    assert panel.output.object == "Erro: backend offline"
    assert panel.show() is panel.panel
