"""\
Módulo: src/radioskysims/application/radiosources_use_cases.py

Casos de uso (application layer) para o contexto de RadioSources.

Orquestram ingestão de catálogos, persistência em banco de dados e
consulta/avaliação de fluxos de fontes de rádio, sem depender de
implementações concretas de infraestrutura.

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import pandas as pd

from radioskysims_sources.application.ports.radiosources import (
    CatalogIngestPort,
    RadioSourcesCatalogPort,
    RadioSourcesFluxPort,
    RadioSourcesRepositoryPort,
)
from radioskysims_sources.domain.radiosources.models import (
    RadioSource,
    RadioSourceFluxSample,
    SpectralModel,
)


@dataclass
class BuildRadioSourcesCatalogInput:
    """Dados de entrada para construção/atualização do catálogo de fontes."""

    survey: str
    adql_query: str


class BuildRadioSourcesCatalogUseCase:
    """Caso de uso para construir/atualizar o catálogo de fontes.

    Fluxo típico:
    1. Usa :class:`CatalogIngestPort` para obter dados brutos.
    2. Converte linhas em :class:`RadioSource`.
    3. Persiste via :class:`RadioSourcesRepositoryPort`.
    """

    def __init__(
        self,
        ingest_port: CatalogIngestPort,
        repository: RadioSourcesRepositoryPort,
    ) -> None:
        self._ingest_port = ingest_port
        self._repository = repository

    def execute(self, data: BuildRadioSourcesCatalogInput) -> None:
        """Executa o fluxo de construção de catálogo.

        A implementação de conversão DataFrame → RadioSource será
        adicionada em iterações futuras.
        """

        _raw_table = self._ingest_port.ingest_from_tap(
            survey=data.survey,
            adql_query=data.adql_query,
        )
        # TODO: conversão _raw_table → list[RadioSource]
        sources: list[RadioSource] = []
        self._repository.save_sources(sources)


# ---------------------------------------------------------------------------
# Fluxo específico NVSS + SPECFIND
# ---------------------------------------------------------------------------


@dataclass
class BuildNvssSpecfindCatalogInput:
    """Parâmetros para construir catálogo via NVSS + SPECFIND.

    Todos os ângulos em graus e fluxos em Jy.
    """

    ra_min_deg: float
    ra_max_deg: float
    dec_min_deg: float
    dec_max_deg: float
    smin_jy: float


class BuildNvssSpecfindCatalogUseCase:
    """Caso de uso para construir catálogo a partir de NVSS + SPECFIND.

    Fluxo resumido:

    1. Monta e executa uma query ADQL em NVSS para um retângulo (RA, Dec)
       com corte em fluxo (S1.4 > ``smin_jy``).
    2. Usa os nomes NVSS retornados para consultar SPECFIND e obter
       medidas espectrais adicionais.
    3. Normaliza o resultado para entidades de domínio :class:`RadioSource`.
    4. Persiste via :class:`RadioSourcesRepositoryPort`.
    """

    NVSS_TABLE = '"VIII/65/nvss"'
    SPECFIND_TABLE = '"VIII/104/spectra"'

    def __init__(
        self,
        ingest_port: CatalogIngestPort,
        repository: RadioSourcesRepositoryPort,
    ) -> None:
        self._ingest_port = ingest_port
        self._repository = repository

    @staticmethod
    def _adql_escape(col: str) -> str:
        """Escapa nomes de colunas potencialmente problemáticos."""

        if any(c in col for c in [".", "(", ")"]) or col[0].isdigit():
            return f'"{col}"'
        return col

    def _build_nvss_query(self, params: BuildNvssSpecfindCatalogInput) -> str:
        nvss_cols = ["RAJ2000", "DEJ2000", "NVSS", "S1.4"]
        cols_expr = ", ".join(self._adql_escape(c) for c in nvss_cols)
        # NVSS fornece S1.4 tipicamente em mJy, convertemos corte de Jy -> mJy
        smin_mjy = params.smin_jy * 1000.0
        return f"""
        SELECT {cols_expr}
        FROM {self.NVSS_TABLE}
        WHERE
            RAJ2000 BETWEEN {params.ra_min_deg} AND {params.ra_max_deg}
        AND DEJ2000 BETWEEN {params.dec_min_deg} AND {params.dec_max_deg}
        AND "S1.4" > {smin_mjy}
        """

    def _build_specfind_query(self, nvss_names: list[str]) -> str:
        spec_cols = ["Name", "nu", "S(nu)"]
        cols_expr = ", ".join(self._adql_escape(c) for c in spec_cols)
        # SPECFIND armazena nomes no formato "NVSS 125631+020229"
        names_sql = ", ".join(f"'NVSS {n}'" for n in nvss_names)
        return f"""
        SELECT {cols_expr}
        FROM {self.SPECFIND_TABLE}
        WHERE Name IN ({names_sql})
        """

    @staticmethod
    def _normalize_specfind_df(df: pd.DataFrame) -> pd.DataFrame:
        """Filtra e normaliza nomes NVSS no DataFrame SPECFIND.

        Mantém apenas linhas cujo ``Name`` começa com ``"NVSS "`` e remove
        o prefixo, colocando o nome limpo na coluna ``NVSS``.
        """

        if "Name" not in df.columns:
            return df.iloc[0:0]
        spec_df = df.copy()
        spec_df = spec_df[spec_df["Name"].str.startswith("NVSS ")].copy()
        spec_df["NVSS"] = spec_df["Name"].str.replace("NVSS ", "", regex=False)
        if spec_df.empty:
            return spec_df.iloc[0:0]
        return spec_df

    @staticmethod
    def _dataframes_to_sources(
        nvss_df: pd.DataFrame, spec_df: pd.DataFrame
    ) -> list[RadioSource]:
        """Converte DataFrames NVSS + SPECFIND em entidades RadioSource.

        - Usa RAJ2000/DEJ2000 em graus como posição.
        - Define S0 como o fluxo NVSS S1.4 convertido para Jy.
        - Estima o índice espectral a partir de um ponto extra em SPECFIND,
          se disponível; caso contrário, usa valor padrão (0.7).
        """

        if nvss_df.empty:
            return []

        merged = nvss_df.merge(
            spec_df, on="NVSS", how="left", suffixes=("_nvss", "_spec")
        )

        sources: list[RadioSource] = []
        grouped = merged.groupby("NVSS")

        for nvss_name, group in grouped:
            row0 = group.iloc[0]

            ra_deg = float(row0["RAJ2000"])
            dec_deg = float(row0["DEJ2000"])
            s1_4_mjy = float(row0["S1_4"])
            s0_jy = s1_4_mjy / 1000.0
            nu0_mhz = 1400.0

            alpha = 0.7  # valor padrão caso não haja informação espectral

            # tenta usar uma linha SPECFIND válida para estimar alpha
            valid_spec = (
                group.dropna(subset=["nu", "S_nu_"])
                if "nu" in group.columns and "S_nu_" in group.columns
                else group.iloc[0:0]
            )
            if not valid_spec.empty:
                # escolhe a linha com frequência mais distante de 1400 MHz para melhor alavancagem
                valid_spec = valid_spec.assign(
                    _dist=(valid_spec["nu"] - nu0_mhz).abs()
                ).sort_values("_dist", ascending=False)
                spec_row = valid_spec.iloc[0]
                nu_mhz = float(spec_row["nu"])
                s_nu_jy = float(spec_row["S_nu_"])
                if nu_mhz > 0 and s_nu_jy > 0 and nu_mhz != nu0_mhz:
                    try:
                        alpha = -log(s_nu_jy / s0_jy) / log(nu_mhz / nu0_mhz)
                    except ValueError:
                        alpha = 0.7

            spectral_model = SpectralModel(
                nu0_mhz=nu0_mhz,
                s0_jy=s0_jy,
                alpha=alpha,
            )

            src = RadioSource(
                id=str(nvss_name),
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                spectral_model=spectral_model,
            )
            sources.append(src)

        return sources

    def execute(self, params: BuildNvssSpecfindCatalogInput) -> list[RadioSource]:
        """Executa o fluxo completo NVSS + SPECFIND.

        Retorna a lista de :class:`RadioSource` criadas e também as
        persiste via repositório configurado.
        """

        # 1) NVSS na janela definida
        nvss_query = self._build_nvss_query(params)
        nvss_df_any: Any = self._ingest_port.ingest_from_tap(
            survey="NVSS",
            adql_query=nvss_query,
        )
        nvss_df = pd.DataFrame(nvss_df_any)
        if nvss_df.empty:
            self._repository.save_sources([])
            return []

        if "NVSS" not in nvss_df.columns:
            msg = "Resultado NVSS não contém coluna 'NVSS'."
            raise ValueError(msg)

        names: list[str] = list(map(str, nvss_df["NVSS"].values))
        if not names:
            self._repository.save_sources([])
            return []

        # 2) SPECFIND apenas para esses nomes
        spec_query = self._build_specfind_query(names)
        spec_df_any: Any = self._ingest_port.ingest_from_tap(
            survey="SPECFIND",
            adql_query=spec_query,
        )
        spec_raw_df = pd.DataFrame(spec_df_any)
        spec_df = self._normalize_specfind_df(spec_raw_df)

        # 3) Normaliza para entidades de domínio
        sources = self._dataframes_to_sources(nvss_df, spec_df)
        self._repository.save_sources(sources)
        return sources


@dataclass
class ComputeRadioSourcesForegroundInput:
    """Entrada mínima para cálculo de foreground discreto de fontes."""

    theta_deg: float
    phi_deg: float
    freqs_mhz: list[float]
    radius_deg: float
    min_flux_jy: float


class ComputeRadioSourcesForegroundUseCase:
    """Caso de uso para calcular contribuição de fontes pontuais.

    Consome os ports de catálogo e fluxo definidos em
    :mod:`radioskysims_sources.application.ports.radiosources`.
    """

    def __init__(
        self,
        catalog_port: RadioSourcesCatalogPort,
        flux_port: RadioSourcesFluxPort,
    ) -> None:
        self._catalog_port = catalog_port
        self._flux_port = flux_port

    def execute(
        self, data: ComputeRadioSourcesForegroundInput
    ) -> list[RadioSourceFluxSample]:
        """Executa o cálculo de foreground discreto.

        A implementação inicial delega seleção de fontes + avaliação
        espectral para os ports, mantendo o caso de uso fino.
        """

        sources: list[RadioSource] = self._catalog_port.query_sources(
            ra_deg=data.phi_deg,  # conversão angular fina ficará para adapters/geo
            dec_deg=data.theta_deg,
            radius_deg=data.radius_deg,
            min_flux_jy=data.min_flux_jy,
        )
        return self._flux_port.evaluate_flux(sources, data.freqs_mhz)
