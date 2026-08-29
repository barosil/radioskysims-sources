"""\
Módulo: src/radioskysims/domain/localsources/models.py

Descrição:
    Modelos de domínio para fontes locais (agregado LocalSources),
    com suporte a sinal solar parametrizado por tempo/frequência.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field


class SolarFlux(BaseModel):
    """Sinal solar em grade (tempo x frequência) com interpolação contínua."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_utc: datetime = Field(..., description="Início do intervalo UTC.")
    end_utc: datetime = Field(..., description="Fim do intervalo UTC.")
    dataset: xr.Dataset = Field(..., description="Dataset com variável 'flux_jy'.")
    source_url: str | None = Field(default=None, description="URL de origem dos dados.")

    def _flux_var(self) -> xr.DataArray:
        """Retorna a variável de fluxo em Jy (converte SFU se necessário)."""
        if "flux_jy" in self.dataset:
            return self.dataset["flux_jy"]
        if "flux_sfu" in self.dataset:
            return self.dataset["flux_sfu"] * 1.0e4
        msg = "Dataset SolarFlux deve conter variável 'flux_jy' ou 'flux_sfu'."
        raise ValueError(msg)

    def evaluate(self, t_utc: datetime, freq_mhz: float) -> float:
        """Avalia fluxo solar (Jy) em um ponto (tempo, frequência).

        Para avaliação em grade completa, prefira ``evaluate_grid()``
        que é vetorizada e muito mais eficiente.
        """
        flux_var = self._flux_var()
        t_ref = t_utc.astimezone(UTC).replace(tzinfo=None)
        t_ns = np.datetime64(t_ref, "ns")
        flux_var = flux_var.assign_coords(time=flux_var["time"].astype("datetime64[ns]"))
        value = flux_var.interp(
            time=t_ns,
            frequency_mhz=freq_mhz,
            # Sem extrapolação: fora da cobertura dos dados → NaN (a
            # extrapolação temporal produz fluxos negativos irreais).
            kwargs={"fill_value": None},
        )
        return float(value.values)

    def evaluate_grid(
        self,
        times: np.ndarray,
        freqs_mhz: np.ndarray,
    ) -> np.ndarray:
        """Avalia o fluxo solar em grade completa (tempo × frequência).

        Args:
            times: Array 1-D de ``np.datetime64`` (ns) representando os
                   instantes UTC de avaliação.
            freqs_mhz: Array 1-D de frequências em MHz.

        Returns:
            Array 2-D ``(len(times), len(freqs_mhz))`` com fluxo em Jy.
        """
        flux_var = self._flux_var()
        flux_var = flux_var.assign_coords(time=flux_var["time"].astype("datetime64[ns]"))
        grid = flux_var.interp(
            time=times,
            frequency_mhz=freqs_mhz,
            # Sem extrapolação: fora da cobertura dos dados → NaN (a
            # extrapolação temporal produz fluxos negativos irreais).
            kwargs={"fill_value": None},
        )
        return np.asarray(grid.values, dtype=float)


class LocalSourceTrack(BaseModel):
    """Track temporal de uma fonte local (ex.: Sun) no referencial do observador."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_name: str = Field(..., description="Nome da fonte local.")
    dataset: xr.Dataset = Field(
        ...,
        description=(
            "Dataset com variáveis `theta_rad`, `az_rad`, `el_rad`, `range_m` e dimensão `time`."
        ),
    )

    def evaluate(self, t_utc: datetime) -> dict[str, float]:
        """Interpola o track no instante UTC e retorna métricas geométricas."""
        required = {"theta_rad", "az_rad", "el_rad", "range_m"}
        missing = required - set(self.dataset.data_vars)
        if missing:
            msg = f"Dataset LocalSourceTrack sem variáveis obrigatórias: {sorted(missing)}"
            raise ValueError(msg)

        t_ref = t_utc.astimezone(UTC).replace(tzinfo=None)
        t_ns = np.datetime64(t_ref, "ns")
        ds = self.dataset.assign_coords(time=self.dataset["time"].astype("datetime64[ns]"))
        point = ds.interp(time=t_ns, kwargs={"fill_value": "extrapolate"})
        return {
            "theta_rad": float(point["theta_rad"].values),
            "az_rad": float(point["az_rad"].values),
            "el_rad": float(point["el_rad"].values),
            "range_m": float(point["range_m"].values),
        }


class LocalSource(BaseModel):
    """Entidade genérica de fonte local (ex.: Sun)."""

    name: str = Field(..., description="Nome da fonte local.")
    kind: str = Field(..., description="Tipo da fonte (sun, moon, planet, etc).")
    has_track: bool = Field(default=True, description="Fonte possui track disponível.")
    has_signal: bool = Field(default=True, description="Fonte possui sinal disponível.")


class LocalSources(BaseModel):
    """Agregado de fontes locais para um cenário de simulação."""

    sources: list[LocalSource] = Field(default_factory=list)
