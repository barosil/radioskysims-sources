"""\
Módulo: src/radioskysims/infrastructure/db/radiosources_repository.py

Implementações de repositório para o contexto de RadioSources.

Inclui uma implementação em memória útil para testes rápidos e
notebooks, e serve de referência para uma futura implementação
sobre PostgreSQL usando SQLModel/SQLAlchemy, alinhada às práticas
já usadas para o contexto de Beam e Satellite.

Unidades:
- Frequência em MHz
- Fluxo em Jy
- Ângulos em graus

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from radioskysims_sources.domain.radiosources.models import RadioSource
from radioskysims_sources.infrastructure.db.radiosources_db_model import RadioSourceDB
from radioskysims_sources.infrastructure.ports.radiosources import RadioSourcesRepositoryPort

logger = logging.getLogger(__name__)


@dataclass
class InMemoryRadioSourcesRepository(RadioSourcesRepositoryPort):
    """Repositório simples em memória para :class:`RadioSource`.

    Esta implementação é útil como "adapter de referência" e para
    testes locais, sem depender de um banco PostgreSQL configurado.
    """

    _sources: list[RadioSource] = field(default_factory=list)

    def save_sources(self, sources: list[RadioSource]) -> None:
        """Persiste ou atualiza fontes em memória.

        A lógica atual substitui qualquer fonte com o mesmo ``id``.
        """

        existing_by_id = {s.id: s for s in self._sources}
        for src in sources:
            existing_by_id[src.id] = src
        self._sources = list(existing_by_id.values())
        logger.info("Repositório in-memory agora contém %d fontes", len(self._sources))

    def list_sources_in_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        min_flux_jy: float | None = None,
    ) -> list[RadioSource]:
        """Retorna fontes dentro de uma região circular aproximada em graus.

        A distância angular é calculada usando a fórmula do cosseno
        esférico em graus. Para pequenos ângulos, é adequada como
        aproximação inicial.
        """

        def _angsep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
            ra1_rad = math.radians(ra1)
            ra2_rad = math.radians(ra2)
            dec1_rad = math.radians(dec1)
            dec2_rad = math.radians(dec2)
            cos_sep = math.sin(dec1_rad) * math.sin(dec2_rad) + math.cos(dec1_rad) * math.cos(
                dec2_rad
            ) * math.cos(ra1_rad - ra2_rad)
            cos_sep = max(-1.0, min(1.0, cos_sep))
            return math.degrees(math.acos(cos_sep))

        results: list[RadioSource] = []
        for src in self._sources:
            sep = _angsep_deg(ra_deg, dec_deg, src.ra_deg, src.dec_deg)
            if sep > radius_deg:
                continue
            # Corte de fluxo mínimo (Jy) aplicado sobre o fluxo de referência
            # s0_jy (S1.4 para NVSS) — critério vem do settings.toml (S > X).
            if min_flux_jy is not None and src.spectral_model.s0_jy <= min_flux_jy:
                continue
            results.append(src)

        logger.info(
            "Consulta in-memory em RA=%.3f, Dec=%.3f, raio=%.3f deg retornou %d fontes",
            ra_deg,
            dec_deg,
            radius_deg,
            len(results),
        )
        return results


class PostgresRadioSourcesRepository(RadioSourcesRepositoryPort):
    """Stub para repositório PostgreSQL de RadioSources.

    A implementação concreta deverá seguir o padrão de ``BeamRepository``
    e ``SatelliteRepository``, utilizando SQLModel/SQLAlchemy para mapear
    entidades de banco e conversões ``DB <-> domínio``.
    """

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Cria um repositório PostgreSQL para ``RadioSource``.

        Parameters
        ----------
        session_factory:
            Callable que retorna um objeto :class:`sqlmodel.Session`,
            tipicamente algo como ``Database().session`` da camada de
            infraestrutura de banco de dados.
        """

        self._session_factory = session_factory

    def save_sources(self, sources: list[RadioSource]) -> None:
        """Persiste ou atualiza fontes no banco PostgreSQL.

        A estratégia atual é *upsert* por ``id``: se a fonte já existe,
        seus campos são atualizados; caso contrário, um novo registro é
        criado.
        """

        if not sources:
            return

        try:
            with self._session_factory() as session:
                for src in sources:
                    existing = session.get(RadioSourceDB, src.id)
                    if existing is None:
                        db_obj = RadioSourceDB.from_domain(src)
                        session.add(db_obj)
                    else:
                        updated = RadioSourceDB.from_domain(src)
                        for field_name, value in updated.model_dump(exclude={"id"}).items():
                            setattr(existing, field_name, value)

                session.commit()

            logger.info("Persistidas %d fontes em PostgreSQL", len(sources))
        except SQLAlchemyError as exc:  # pragma: no cover - log + rethrow
            logger.exception("Erro ao salvar RadioSources no PostgreSQL: %s", exc)
            raise

    def list_sources_in_region(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        min_flux_jy: float | None = None,
    ) -> list[RadioSource]:
        """Consulta fontes em uma região aproximada no banco PostgreSQL.

        Por simplicidade e previsibilidade científica, esta primeira
        implementação carrega os registros do banco e aplica o mesmo
        filtro angular em Python usado pelo repositório in-memory.
        """

        # Corte de fluxo mínimo (Jy) aplicado em Python sobre s0_jy — o mesmo
        # critério do repositório in-memory (settings: S > X Jy).
        def _angsep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
            ra1_rad = math.radians(ra1)
            ra2_rad = math.radians(ra2)
            dec1_rad = math.radians(dec1)
            dec2_rad = math.radians(dec2)
            cos_sep = math.sin(dec1_rad) * math.sin(dec2_rad) + math.cos(dec1_rad) * math.cos(
                dec2_rad
            ) * math.cos(ra1_rad - ra2_rad)
            cos_sep = max(-1.0, min(1.0, cos_sep))
            return math.degrees(math.acos(cos_sep))

        try:
            with self._session_factory() as session:
                statement = select(RadioSourceDB)
                db_sources = list(session.exec(statement).all())

            domain_sources: list[RadioSource] = []
            for db_obj in db_sources:
                src = db_obj.to_domain()
                sep = _angsep_deg(ra_deg, dec_deg, src.ra_deg, src.dec_deg)
                if sep > radius_deg:
                    continue
                # Corte de fluxo mínimo (Jy) sobre s0_jy (S1.4 para NVSS).
                if min_flux_jy is not None and src.spectral_model.s0_jy <= min_flux_jy:
                    continue
                domain_sources.append(src)

            logger.info(
                "Consulta PostgreSQL em RA=%.3f, Dec=%.3f, raio=%.3f deg retornou %d fontes",
                ra_deg,
                dec_deg,
                radius_deg,
                len(domain_sources),
            )
            return domain_sources
        except SQLAlchemyError as exc:  # pragma: no cover - log + rethrow
            logger.exception(
                "Erro ao consultar RadioSources no PostgreSQL para RA=%.3f, Dec=%.3f: %s",
                ra_deg,
                dec_deg,
                exc,
            )
            raise
