"""\
Módulo: src/radioskysims/infrastructure/db/solar_flux_db_model.py

Descrição:
    Modelo SQLModel para metadados de arquivos SolarFlux em cache local.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class SolarFluxDB(SQLModel, table=True):
    """Registro de cache SolarFlux persistido em NetCDF."""

    __tablename__ = "solar_flux_cache"

    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(index=True, unique=True)
    start_utc: datetime = Field(index=True)
    end_utc: datetime = Field(index=True)
    source_url: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
