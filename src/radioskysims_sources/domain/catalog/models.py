"""
Módulo: src/radioskysims/domain/catalog/models.py

Modelos de domínio para o agregado Catalog — metadados de ingestão
de catálogos externos de fontes de rádio (NVSS, SPECFIND, etc.).

Define:
    - CatalogQuery: parâmetros de consulta utilizados.
    - CatalogIngestion: registro de uma operação de ingestão.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class CatalogQuery(BaseModel):
    """Parâmetros de uma consulta utilizada para obter dados do catálogo.

    Attributes:
        survey: Nome do survey (ex.: "NVSS", "SPECFIND").
        query: Cadeia da query ADQL ou identificador da consulta.
        params: Parâmetros adicionais da consulta (RA range, Dec range, flux cutoff).
    """

    survey: str = Field(..., description="Nome do survey (ex.: NVSS).")
    query: str = Field(..., description="Query ADQL enviada ao serviço TAP.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parâmetros adicionais da consulta.",
    )


class CatalogIngestion(BaseModel):
    """Registro de uma ingestão de catálogo.

    Attributes:
        catalog_name: Nome lógico do catálogo (ex.: "nvss_specfind").
        ingested_at: Timestamp UTC da ingestão.
        num_sources: Número de fontes ingeridas.
        query: Parâmetros da consulta realizada.
        status: "success", "partial", "failed".
        error: Mensagem de erro, se houver.
    """

    catalog_name: str = Field(..., description="Nome lógico do catálogo.")
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp UTC da ingestão.",
    )
    num_sources: int = Field(0, ge=0, description="Número de fontes ingeridas.")
    query: CatalogQuery | None = Field(
        default=None, description="Parâmetros da consulta realizada."
    )
    status: str = Field(
        "success",
        pattern=r"^(success|partial|failed)$",
        description="Status da ingestão.",
    )
    error: str | None = Field(default=None, description="Mensagem de erro, se houver.")
