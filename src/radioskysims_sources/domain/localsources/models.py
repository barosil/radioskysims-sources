"""\
Módulo: src/radioskysims/domain/localsources/models.py

Descrição:
    Modelos de domínio para fontes locais (agregado LocalSources),
    com suporte a sinal solar parametrizado por tempo/frequência.

Copyright BINGO Collaboration
Last modified: 2026-10-12
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
        r"""Mediana **diária** do fluxo solar por frequência (NaN-aware).

        Para cada instante em ``times`` retorna a mediana dos dados do MESMO dia
        UTC (ignorando amostras ``NaN``), em vez de interpolar hora a hora.

        Motivação: o SRD/SRS é medido em instantes UTC discretos que não
        coincidem com a hora local de observação do radiotelescópio; a
        interpolação temporal compara instantes de medição do Sol com a grade
        de observação e produz artefatos nas bordas (extrapolação/»NaN«). A
        mediana diária é uma grandeza mais estável e fisicamente representativa
        do fluxo do dia.

        Regras de borda:

        - A interpolação é feita **apenas em frequência** (curva suave); o
          tempo é reduzido por mediana.
        - Dias sem cobertura no dataset, ou frequências totalmente ``NaN``
          nesse dia, caem para a mediana geral (ignorando ``NaN``).
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
        flux_var = flux_var.transpose("time", "frequency_mhz")

        # DESEMPENHO: a redução diária é feita ANTES da interpolação em
        # frequência, na grade NATIVA do arquivo. O NoRP tem cadência de ~1 s
        # (ex.: ~93 mil amostras em 3 dias) e o grid do BINGO tem 4096 canais:
        # interpolar primeiro materializaria ~3 GB e, como a mediana precisa ser
        # recalculada para CADA instante solicitado (721 no caso de 24 h), o
        # custo real chegava a ~700 GB de tráfego de memória. Reduzindo nos
        # poucos canais nativos (ex.: 6) o custo cai ~2 ordens de grandeza.
        native = np.asarray(flux_var.values, dtype=float)
        native_freqs = np.asarray(flux_var["frequency_mhz"].values, dtype=float)
        native_day = np.asarray(flux_var["time"].values).astype("datetime64[D]")

        unique_days = np.unique(native_day)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            overall_native = np.nanmedian(native, axis=0)
            day_medians = np.vstack(
                [np.nanmedian(native[native_day == day], axis=0) for day in unique_days]
            )

        # Interpolação APENAS em frequência (curva suave) e apenas sobre a
        # matriz reduzida ``(n_dias + 1, n_freq_nativa)`` — alguns kB.
        # Extrapolação linear nas bordas do grid nativo (o espectro solar é
        # suave; ex.: o NoRP começa em 1 GHz e o BINGO começa em ~995 MHz).
        stacked = xr.DataArray(
            np.vstack([day_medians, overall_native[None, :]]),
            coords={
                "day_index": np.arange(unique_days.size + 1),
                "frequency_mhz": native_freqs,
            },
            dims=("day_index", "frequency_mhz"),
        )
        interp_f = np.asarray(
            stacked.interp(frequency_mhz=freqs, kwargs={"fill_value": "extrapolate"}).values,
            dtype=float,
        )
        overall = interp_f[-1]
        # Frequências totalmente NaN no dia → mediana geral.
        day_medians_f = np.where(np.isfinite(interp_f[:-1]), interp_f[:-1], overall)

        # Mapeia cada instante solicitado para a mediana do seu dia UTC.
        times_day = times.astype("datetime64[D]")
        pos = np.clip(np.searchsorted(unique_days, times_day), 0, unique_days.size - 1)
        covered = unique_days[pos] == times_day
        return np.where(covered[:, None], day_medians_f[pos], overall)


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
