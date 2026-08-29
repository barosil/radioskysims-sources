"""
Módulo: src/radioskysims/domain/catalog/__init__.py

Modelos de domínio para o agregado Catalog — metadados de ingestão
de catálogos externos de fontes de rádio.

Copyright BINGO Collaboration
Last modified: 2026-07-26
"""

from __future__ import annotations

from .models import CatalogIngestion, CatalogQuery

__all__ = [
    "CatalogIngestion",
    "CatalogQuery",
]
