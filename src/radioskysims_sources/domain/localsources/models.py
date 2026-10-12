"""\
Módulo: src/radioskysims/domain/localsources/models.py

Descrição:
    Modelos de domínio para fontes locais (agregado LocalSources),
    com suporte a sinal solar parametrizado por tempo/frequência.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

import warnings
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

    def daily_mean_grid(
        self,
        times: np.ndarray,
        freqs_mhz: np.ndarray,
    ) -> np.ndarray:
        r"""Média **diária** do fluxo solar por frequência (NaN-aware).

        Para cada instante em ``times`` retorna a média dos dados do MESMO dia
        UTC (ignorando amostras ``NaN``), em vez de interpolar hora a hora.

        Motivação: o SRD/SRS é medido em instantes UTC discretos que não
        coincidem com a hora local de observação do radiotelescópio; a
        interpolação temporal compara instantes de medição do Sol com a grade
        de observação e produz artefatos nas bordas (extrapolação/»NaN«). A
        média diária é uma grandeza mais estável e fisicamente representativa
        do fluxo do dia.

        Regras de borda:

        - A interpolação é feita **apenas em frequência** (curva suave); o
          tempo é reduzido por média.
        - Dias sem cobertura no dataset, ou frequências totalmente ``NaN``
          nesse dia, caem para a média geral (ignorando ``NaN``).
        - Se não houver nenhuma amostra finita, retorna ``NaN`` (o chamador
          decide o fallback físico).

        Parameters
        ----------
        times : array
            Instantes (``np.datetime64``, ns) da grade de simulação.
        freqs_mhz : array
            Frequências (MHz) da grade de simulação.

        Returns
        -------
        ndarray
            Grade ``(len(times), len(freqs_mhz))`` de fluxo em Jy.
        """
        times = np.asarray(times).astype("datetime64[ns]")
        freqs = np.asarray(freqs_mhz, dtype=float)

        flux_var = self._flux_var()
        flux_var = flux_var.assign_coords(time=flux_var["time"].astype("datetime64[ns]"))
        # Interpola só em frequência (sem extrapolação temporal → fora de
        # cobertura fica NaN e é ignorado pela média).
        flux_f = flux_var.interp(
            frequency_mhz=freqs,
            kwargs={"fill_value": None},
        ).transpose("time", "frequency_mhz")
        values = np.asarray(flux_f.values, dtype=float)
        day_of = np.asarray(flux_f["time"].values).astype("datetime64[D]")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            overall = np.nanmean(values, axis=0)

        out = np.empty((times.size, freqs.size), dtype=float)
        times_day = times.astype("datetime64[D]")
        for i, day in enumerate(times_day):
            mask = day_of == day
            if np.any(mask):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_day = np.nanmean(values[mask], axis=0)
                # Frequências totalmente NaN no dia → média geral.
                out[i] = np.where(np.isfinite(mean_day), mean_day, overall)
            else:
                out[i] = overall
        return out


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
