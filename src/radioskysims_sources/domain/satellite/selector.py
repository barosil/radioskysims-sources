from __future__ import annotations

import logging

import pandas as pd
from astropy.time import Time

from radioskysims_sources.infrastructure.db.satellite.models import SatID

from .models import OrbitalElement

logger = logging.getLogger(__name__)


class TLESelector:
    @staticmethod
    def select(
        satids: list[SatID],
        epoch_jd: float,
    ) -> list[dict]:
        rows = []
        satellites = []

        for sat in satids:
            for elem in sat.elements:
                rows.append(
                    {
                        "norad_id": sat.norad_id,
                        "name": sat.name,
                        "constellation": sat.constellation,
                        "prn": getattr(sat, "prn", None),
                        "epoch": elem.epoch,
                        "delta_epoch": abs(Time(elem.epoch).jd - epoch_jd),
                        "element": elem,
                    }
                )

            _df_sats = pd.DataFrame(rows)
            idx = _df_sats.groupby("norad_id")["delta_epoch"].idxmin().to_numpy()
            # orbital = [(_df_sats.loc[ii]["element"].model_dump()) for ii in idx]
            params = (
                _df_sats[
                    ["norad_id", "name", "constellation", "prn", "epoch", "element"]
                ]
                .loc[idx]
                .to_dict(orient="records")
            )
            # for param, orbital_ in zip(params, orbital):
            for param in params:
                element = param["element"]

                # Converter época para datetime se necessário (helper function)
                def convert_epoch(obj):
                    if hasattr(obj, "iso"):  # Astropy Time object
                        return obj.datetime
                    return obj

                # Converter elemento para OrbitalElement se necessário
                if isinstance(element, OrbitalElement):
                    _orbital = element
                elif isinstance(element, dict):
                    if "epoch" in element:
                        element["epoch"] = convert_epoch(element["epoch"])
                    _orbital = OrbitalElement.model_validate(element)
                # Tentar converter para dict e depois validar
                elif hasattr(element, "model_dump"):
                    elem_dict = element.model_dump()
                    if "epoch" in elem_dict:
                        elem_dict["epoch"] = convert_epoch(elem_dict["epoch"])
                    _orbital = OrbitalElement.model_validate(elem_dict)
                else:
                    # Se é um objeto simples com atributos, extrair como dict
                    elem_dict = {}
                    if hasattr(element, "epoch"):
                        elem_dict["epoch"] = convert_epoch(element.epoch)
                    # Se tem mais atributos, tentar extrair
                    for attr in dir(element):
                        if not attr.startswith("_") and attr != "epoch":
                            try:
                                val = getattr(element, attr)
                                if not callable(val):
                                    elem_dict[attr] = val
                            except Exception:
                                pass
                    if elem_dict:
                        _orbital = OrbitalElement.model_validate(elem_dict)
                    else:
                        _orbital = element

                # Converter epoch do param também
                epoch = convert_epoch(param["epoch"])

                # Retornar como dicionário (conforme tipo de retorno list[dict])
                satellites.append(
                    {
                        "norad_id": param["norad_id"],
                        "name": param["name"],
                        "constellation": param["constellation"],
                        "prn": param.get("prn"),
                        "epoch": epoch,
                        "element": _orbital,
                    }
                )
        return satellites
