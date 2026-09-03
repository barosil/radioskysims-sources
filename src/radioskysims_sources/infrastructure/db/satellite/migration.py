"""Migração idempotente da coluna ``prn`` em ``SatID``.

A tabela ``satid`` foi criada antes do mapeamento NORAD ↔ Nome ↔ PRN existir.
Como o projeto não usa Alembic (as tabelas são criadas via
``SQLModel.metadata.create_all``, que não adiciona colunas), este módulo
fornece migrações leves e idempotentes:

- ``ensure_prn_column``: ``ALTER TABLE satid ADD COLUMN IF NOT EXISTS prn``;
- ``backfill_prn``: popula ``prn`` a partir do ``name``/``constellation`` já
  persistidos (CelesTrak) para registros existentes.

Ambas são seguras para rodar múltiplas vezes (idempotentes).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlmodel import select

logger = logging.getLogger(__name__)

_DDL = "ALTER TABLE satid ADD COLUMN IF NOT EXISTS prn INTEGER"


def ensure_prn_column(db: Any) -> bool:
    """Garante a coluna ``prn`` na tabela ``satid`` (idempotente).

    Aceita um objeto ``Database`` (radioskysims_core.db.engine) — acessa
    ``db.engine`` para o DDL.

    Returns
    -------
    bool
        ``True`` se a coluna já existia ou foi criada com sucesso.
    """
    engine = getattr(db, "engine", db)
    try:
        with engine.begin() as conn:
            conn.execute(text(_DDL))
        return True
    except Exception as exc:  # noqa: BLE001 — SQLite/outros dialetos podem não suportar
        logger.warning("Falha ao garantir coluna prn em satid: %s", exc)
        return False


def backfill_prn(db: Any) -> int:
    """Popula ``prn`` de registros ``SatID`` ainda sem PRN.

    Usa o nome CelesTrak e a constelação já persistidos para derivar o PRN
    (mesma lógica da ingestão). Aceita um objeto ``Database`` (com
    ``.session()``).

    Returns
    -------
    int
        Quantidade de registros atualizados.
    """
    from radioskysims_sources.domain.satellite.prn import parse_prn_from_name
    from radioskysims_sources.infrastructure.db.satellite.models import SatID

    session_ctx = getattr(db, "session", None)
    if session_ctx is None:
        msg = "backfill_prn requer um objeto Database (com .session())."
        raise TypeError(msg)

    with session_ctx() as session:
        rows = session.execute(select(SatID).where(SatID.prn.is_(None))).scalars().all()

        updated = 0
        for sat in rows:
            prn = parse_prn_from_name(sat.name, sat.constellation)
            if prn is not None:
                sat.prn = prn
                updated += 1
        session.commit()
    if updated:
        logger.info("backfill_prn: %d registros atualizados", updated)
    return updated
