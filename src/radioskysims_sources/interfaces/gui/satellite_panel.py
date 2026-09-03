# interfaces/gui/satellite_panel.py
from __future__ import annotations

from datetime import datetime

import panel as pn

pn.extension("tabulator")


class SatellitePanel:
    def __init__(self, factory):
        self.factory = factory

        self.norad_input = pn.widgets.TextAreaInput(
            name="NORAD IDs (1 por linha)", placeholder="12345\n23456"
        )
        self.constellation_input = pn.widgets.MultiSelect(name="Constelações", options=[], size=6)
        self.epoch_picker = pn.widgets.DatetimePicker(name="Época", value=datetime.utcnow())

        self.output = pn.pane.Markdown("Pronto para buscar.")
        self.button = pn.widgets.Button(name="Buscar Satélites", button_type="primary")
        self.button.on_click(self._on_submit)

        self.panel = pn.Column(
            pn.pane.Markdown("### Consulta de Satélites"),
            self.norad_input,
            self.constellation_input,
            self.epoch_picker,
            self.button,
            self.output,
        )

    def _on_submit(self, _event):
        epoch = self.epoch_picker.value
        ids = [int(x) for x in self.norad_input.value.strip().splitlines() if x.strip()]
        consts = list(self.constellation_input.value)

        try:
            if ids:
                sats = self.factory.create_from_norad_ids(ids, epoch)
            elif consts:
                sats = self.factory.create_from_constellations(consts, epoch)
            else:
                self.output.object = "Forneça NORAD IDs ou constelações."
                return

            md = "\n".join(
                [
                    f"**{sat.id}** — RA: {sat.coords.data['ra'].values.mean():.3f} rad"
                    for sat in sats
                ]
            )
            self.output.object = f"Satélites encontrados:\n\n{md}"

        except Exception as e:
            self.output.object = f"Erro: {e}"

    def show(self):
        return self.panel
