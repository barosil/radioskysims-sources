"""
Módulo: src/radioskysims/domain/radiosources/tracks.py

Modelo de domínio para tracks de fontes de rádio fixas (celestes).

Uma RadioSourceTrack representa a posição aparente (theta, az, el)
de uma ou mais fontes de rádio com coordenadas celestes fixas (RA, Dec)
ao longo do tempo, no referencial de um observador terrestre.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)


class RadioSourceTrack(BaseModel):
    """Track temporal de fontes de rádio fixas no referencial do observador.

    Encapsula um xarray.Dataset com dimensões (source, time) contendo
    as coordenadas horárias (theta, az, el, range) para cada fonte.

    Attributes:
        dataset: xr.Dataset com variáveis theta_rad, az_rad, el_rad, range_m.
        source_ids: Lista de identificadores das fontes.
        time_range: Tupla (início, fim) ISO do intervalo.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Any = Field(
        ...,
        description=(
            "xr.Dataset com dims (source, time) e variáveis theta_rad, az_rad, el_rad, range_m."
        ),
    )
    source_ids: list[str] = Field(..., description="Identificadores das fontes no dataset.")
    time_range: tuple[str, str] = Field(..., description="Intervalo temporal (início, fim) em ISO.")

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, ds: Any) -> Any:
        """Valida estrutura mínima do dataset.

        Permite dataset vazio (sem variáveis) para representar
        track sem fontes — útil quando um filtro não retorna resultados.
        """
        try:
            import xarray as xr
        except ImportError:
            return ds

        if not isinstance(ds, xr.Dataset):
            msg = "dataset deve ser um xr.Dataset"
            raise TypeError(msg)

        # Permite dataset vazio (ex.: filtro sem resultados)
        if not ds.data_vars:
            return ds

        required_vars = {"theta_rad", "az_rad", "el_rad", "range_m"}
        missing = required_vars - set(ds.data_vars)
        if missing:
            msg = f"Dataset sem variáveis obrigatórias: {sorted(missing)}"
            raise ValueError(msg)

        required_dims = {"source", "time"}
        if not required_dims.issubset(ds.sizes):
            missing_dims = required_dims - set(ds.sizes)
            msg = f"Dimensões ausentes no dataset: {missing_dims}"
            raise ValueError(msg)

        return ds

    def evaluate(self, t_utc: np.datetime64) -> dict[str, np.ndarray]:
        """Interpola tracks para um instante UTC e retorna métricas.

        Args:
            t_utc: Instante de tempo em np.datetime64.

        Returns:
            Dict com arrays theta_rad, az_rad, el_rad, range_m
            indexados por source.
        """
        ds = self.dataset.assign_coords(time=self.dataset["time"].astype("datetime64[ns]"))
        t_ns = np.datetime64(t_utc, "ns")
        point = ds.interp(time=t_ns, kwargs={"fill_value": "extrapolate"})
        return {
            "theta_rad": np.asarray(point["theta_rad"].values),
            "az_rad": np.asarray(point["az_rad"].values),
            "el_rad": np.asarray(point["el_rad"].values),
            "range_m": np.asarray(point["range_m"].values),
        }

    def filter_by_source(self, source_id: str) -> RadioSourceTrack:
        """Retorna novo track filtrado para uma única fonte.

        Mantém explicitamente a dimensão ``source`` com tamanho 1,
        preservando o contrato estrutural de ``RadioSourceTrack``.
        """
        ds = self.dataset.sel(source=[source_id])
        return RadioSourceTrack(
            dataset=ds,
            source_ids=[source_id],
            time_range=self.time_range,
        )

    def filter_by_flux(self, fluxes_jy: dict[str, float], min_flux_jy: float) -> RadioSourceTrack:
        """Retorna novo track apenas com fontes acima de um fluxo mínimo."""
        keep = [sid for sid, f in fluxes_jy.items() if f >= min_flux_jy]
        if not keep:
            import xarray as xr

            return RadioSourceTrack(
                dataset=xr.Dataset(),
                source_ids=[],
                time_range=self.time_range,
            )
        ds = self.dataset.sel(source=keep)
        return RadioSourceTrack(dataset=ds, source_ids=keep, time_range=self.time_range)

    def __len__(self) -> int:
        return len(self.source_ids)
