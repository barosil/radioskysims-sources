"""
Módulo: src/radioskysims/infrastructure/db/satellite/repository_adapter.py

Descrição:
    Adapter SQLAlchemy para o Port SatelliteRepository.
    Implementa a interface definida no domínio utilizando SQLModel/SQLAlchemy.

Padrão Port & Adapter:
    Este módulo representa o Adapter que conecta a interface Port do domínio
    à implementação concreta com SQLAlchemy. Isolado na camada de infraestrutura,
    permitindo múltiplos adapters (InMemory, REST, etc) sem afetar o domínio.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from radioskysims.application.ports import (
    SatelliteRepositoryPort as SatelliteRepository,
)
from radioskysims_sources.infrastructure.db.satellite.models import OrbitalElement, SatID

if TYPE_CHECKING:
    from radioskysims_sources.domain.satellite.models import Satellite


logger = logging.getLogger(__name__)


class SqlAlchemySatelliteRepository(SatelliteRepository):
    """
    Adapter: Implementação de SatelliteRepository com SQLAlchemy/SQLModel.

    Este adapter implementa o Port SatelliteRepository, permitindo acesso
    a dados de satélites persistidos em banco de dados SQL.

    Responsabilidades:
    - Consultar dados de satélites no banco de dados
    - Mapear modelos ORM (SatID, OrbitalElement) → Domínio (Satellite)
    - Garantir tipagem correta
    - Isolar complexidade de SQL do domínio

    Attributes:
        session: Sessão SQLAlchemy para operações no banco de dados.
    """

    def __init__(self, session: Session) -> None:
        """
        Inicializa o adapter com uma sessão SQLAlchemy.

        Args:
            session: Sessão SQLAlchemy ativa para operações no banco.
                    Deve estar vinculada ao engine correto.

        Raises:
            TypeError: Se session não for uma instância de Session.
        """
        if not isinstance(session, Session):
            msg = f"Expected Session, got {type(session)}"
            raise TypeError(msg)
        self.session: Session = session

    def query_sats(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        """
        Recupera satélites pelos seus identificadores NORAD.

        Implementação:
        - Consulta base de dados por SatID com norad_id em norad_ids
        - Carrega elementos orbitais relacionados
        - Mapeia resultados ORM para domínio via _to_domain

        Args:
            norad_ids: Lista de identificadores NORAD dos satélites.
            epoch: Época de referência opcional para elementos orbitais.
                  Se None, usa o elemento orbital mais recente.

        Returns:
            Lista de satélites encontrados. Retorna lista vazia se nenhum
            satélite com os IDs fornecidos existir no banco.

        Raises:
            ValueError: Se a lista de IDs estiver vazia.

        Example:
            >>> adapter = SqlAlchemySatelliteRepository(session)
            >>> satellites = adapter.query_sats([25544, 27424])
            >>> len(satellites) >= 0
            True
        """
        if not norad_ids:
            msg = "norad_ids list cannot be empty"
            raise ValueError(msg)

        # Query: SELECT * FROM satid WHERE norad_id IN (norad_ids)
        stmt = select(SatID).where(SatID.norad_id.in_(norad_ids))
        rows = self.session.exec(stmt).all()

        # Mapeia cada linha ORM para entidade de domínio
        satellites = [self._to_domain(row, epoch) for row in rows]
        return satellites

    def query_constellation(
        self,
        constellations: list[str],
        epoch: datetime | None = None,
    ) -> list[Satellite]:  # type: ignore[name-defined]
        """
        Recupera satélites por constelação.

        Implementação:
        - Consulta base de dados por SatID com constellation em constellations
        - Carrega elementos orbitais relacionados
        - Mapeia resultados ORM para domínio via _to_domain

        Args:
            constellations: Lista de nomes de constelações (ex: GPS, Galileo, Starlink).
            epoch: Época de referência opcional para elementos orbitais.
                  Se None, usa o elemento orbital mais recente.

        Returns:
            Lista de satélites pertencentes às constelações especificadas.
            Retorna lista vazia se nenhum satélite for encontrado.

        Raises:
            ValueError: Se a lista de nomes estiver vazia.

        Example:
            >>> adapter = SqlAlchemySatelliteRepository(session)
            >>> gps_sats = adapter.query_constellation(["GPS"])
            >>> len(gps_sats) > 0
            True
        """
        if not constellations:
            msg = "constellations list cannot be empty"
            raise ValueError(msg)

        # Query: SELECT * FROM satid WHERE constellation IN (constellations)
        stmt = select(SatID).where(SatID.constellation.in_(constellations))
        rows = self.session.exec(stmt).all()

        # Mapeia cada linha ORM para entidade de domínio
        satellites = [self._to_domain(row, epoch) for row in rows]
        return satellites

    # ------------------------------------------------------------------
    # Compat wrappers (nome antigo)
    # ------------------------------------------------------------------
    def get_by_norad_ids(
        self,
        ids: list[int],
        epoch: datetime | None = None,
    ) -> list[Satellite]:  # type: ignore[name-defined]
        """Wrapper legado para compatibilidade com chamadas antigas."""
        return self.query_sats(norad_ids=ids, epoch=epoch)

    def get_by_constellations(
        self,
        names: list[str],
        epoch: datetime | None = None,
    ) -> list[Satellite]:  # type: ignore[name-defined]
        """Wrapper legado para compatibilidade com chamadas antigas."""
        return self.query_constellation(constellations=names, epoch=epoch)

    # ------------------------------------------------------------------
    # Contrato canônico de curadoria/sincronização
    # ------------------------------------------------------------------
    def update_orbital_elements_sync(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
    ) -> None:
        msg = (
            "SqlAlchemySatelliteRepository não implementa sincronização externa de TLEs. "
            "Use infrastructure.db.satellite.satelliterepository.SatelliteRepository "
            "para operações de curadoria/sincronização."
        )
        raise NotImplementedError(msg)

    def update_satellites_from_celestrak(self) -> None:
        msg = (
            "SqlAlchemySatelliteRepository não implementa ingestão de catálogo Celestrak. "
            "Use infrastructure.db.satellite.satelliterepository.SatelliteRepository "
            "para operações de curadoria/sincronização."
        )
        raise NotImplementedError(msg)

    def verify(self, epoch: datetime | None = None) -> bool:
        """Verifica se há satélites com elementos orbitais disponíveis."""
        _ = epoch  # parte da interface do port
        try:
            stmt = select(SatID)
            rows = self.session.exec(stmt).all()
            if not rows:
                return False
            return any(bool(getattr(row, "elements", None)) for row in rows)
        except Exception:
            logger.exception("satellite_repo_verify_failed")
            return False

    def reset(self, kind: str = "all") -> None:
        """Adapter stateless: reset não possui efeito colateral."""
        _ = kind  # parte da interface do port
        return

    @staticmethod
    def _to_domain(db_row: SatID, epoch: datetime | None = None) -> Satellite:
        """
        Mapeia modelo ORM (SatID) → Modelo de Domínio (Satellite).

        Realiza a transformação de dados persistidos para entidade de domínio,
        selecionando o elemento orbital apropriado baseado na época.

        Args:
            db_row: Linha do banco de dados (SatID com OrbitalElements carregados).
            epoch: Época de referência para seleção de elemento orbital.
                  Se None, usa o elemento mais recente.

        Returns:
            Instância de Satellite com dados do banco mapeados.

        Raises:
            ValueError: Se db_row não tiver elementos orbitais.
            RuntimeError: Se mapeamento de dados for inconsistente.

        Notes:
            - Seleção de OrbitalElement: usa o mais próximo à epoch, ou mais recente
            - Campos None são preservados (ex: name, constellation)
        """
        # Import local para evitar circular imports
        from radioskysims_sources.domain.satellite.models import Satellite

        # Seleciona elemento orbital apropriado
        element = _select_orbital_element(db_row, epoch)

        if element is None:
            msg = (
                f"Satellite {db_row.norad_id} ({db_row.name}) "
                "has no orbital elements in database"
            )
            raise ValueError(msg)

        # Mapeia campos do modelo ORM para domínio
        satellite = Satellite(
            norad_id=db_row.norad_id,
            name=db_row.name,
            constellation=db_row.constellation,
            epoch=element.epoch if element.epoch else datetime.utcnow(),
            element=element,
        )

        return satellite


def _select_orbital_element(
    sat_db: SatID,
    epoch: datetime | None = None,
) -> OrbitalElement | None:
    """
    Seleciona o elemento orbital mais apropriado.

    Estratégia:
    1. Se epoch não é None: retorna elemento mais próximo à época
    2. Se epoch é None: retorna elemento mais recente (maior epoch)
    3. Se lista vazia: retorna None

    Args:
        sat_db: Satélite ORM com elementos carregados.
        epoch: Época de referência para seleção.

    Returns:
        OrbitalElement selecionado, ou None se lista vazia.
    """
    if not sat_db.elements:
        return None

    if epoch is None:
        # Retorna elemento mais recente
        return max(
            sat_db.elements,
            key=lambda e: e.epoch if e.epoch else datetime.min,
        )

    # Retorna elemento mais próximo à época fornecida
    return min(
        sat_db.elements,
        key=lambda e: abs((e.epoch - epoch).total_seconds())
        if e.epoch
        else float("inf"),
    )
