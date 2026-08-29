"""\
Módulo: src/radioskysims/infrastructure/db/solar_flux_repository.py

Descrição:
    Repositório de SolarFlux com cache em NetCDF e índice opcional em SQLModel.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import xarray as xr
from sqlmodel import SQLModel, desc, select

from radioskysims_sources.domain.localsources import SolarFlux
from radioskysims_sources.infrastructure.db.solar_flux_db_model import SolarFluxDB
from radioskysims_sources.infrastructure.ports.localsources import SolarFluxRepositoryPort
from radioskysims_core.paths import ROOT

logger = logging.getLogger(__name__)


class SolarFluxRepository(SolarFluxRepositoryPort):
    """Repositório para cache e persistência de ``SolarFlux``."""

    def __init__(
        self,
        engine=None,
        cache_dir: str | Path = "data/aux/signals/solar_flux",
    ) -> None:
        self.engine = engine
        cd = Path(cache_dir).expanduser()
        # Caminho relativo é ancorado na raiz do projeto (notebooks têm CWD próprio)
        if not cd.is_absolute():
            cd = ROOT / cd
        self.cache_dir = cd.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.engine is not None:
            SQLModel.metadata.create_all(
                self.engine.engine,
                tables=[SolarFluxDB.__table__],
            )

    def load_interval(self, start_utc: datetime, end_utc: datetime) -> SolarFlux | None:
        """Busca cache que cubra completamente o intervalo requisitado."""
        if self.engine is None:
            return None

        with self.engine.session() as session:
            stmt = (
                select(SolarFluxDB)
                .where(SolarFluxDB.start_utc <= start_utc)
                .where(SolarFluxDB.end_utc >= end_utc)
                .order_by(desc(SolarFluxDB.created_at))
            )
            row = session.exec(stmt).first()

        if row is None:
            return None

        file_path = self.cache_dir / row.filename
        if not file_path.exists():
            logger.warning("Arquivo SolarFlux registrado não encontrado: %s", file_path)
            return None

        dataset = xr.load_dataset(file_path)
        return SolarFlux(
            start_utc=row.start_utc,
            end_utc=row.end_utc,
            dataset=dataset,
            source_url=row.source_url,
        )

    def save(self, solar_flux: SolarFlux) -> SolarFlux:
        """Persiste ``SolarFlux`` em NetCDF e registra metadados no DB (opcional)."""
        start_tag = solar_flux.start_utc.strftime("%Y%m%dT%H%M%S")
        end_tag = solar_flux.end_utc.strftime("%Y%m%dT%H%M%S")
        filename = f"solar_flux_{start_tag}_{end_tag}.nc"
        file_path = self.cache_dir / filename

        solar_flux.dataset.to_netcdf(file_path)

        if self.engine is not None:
            with self.engine.session() as session:
                existing = session.exec(
                    select(SolarFluxDB).where(SolarFluxDB.filename == filename)
                ).first()
                if existing is None:
                    existing = SolarFluxDB(
                        filename=filename,
                        start_utc=solar_flux.start_utc,
                        end_utc=solar_flux.end_utc,
                        source_url=solar_flux.source_url,
                    )
                    session.add(existing)
                else:
                    existing.start_utc = solar_flux.start_utc
                    existing.end_utc = solar_flux.end_utc
                    existing.source_url = solar_flux.source_url
                session.commit()

        return solar_flux
