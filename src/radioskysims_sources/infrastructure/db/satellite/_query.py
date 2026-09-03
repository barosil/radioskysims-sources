"""Mixin de consulta/loaders de satélites (E8/TD-011).

Agrupa os métodos de resumo, query por ids/constelação, conversão para o
domínio e loaders do ``SatelliteRepository``. Extraído de
``satelliterepository.py``.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from astropy.time import Time
from sqlalchemy.orm import selectinload
from sqlmodel import distinct, func, select

from radioskysims_sources.domain.satellite.models import (
    OrbitalElement as DomainOrbitalElement,
)
from radioskysims_sources.domain.satellite.models import (
    Satellite as DomainSatellite,
)
from radioskysims_sources.domain.satellite.models import (
    as_utc,
)
from radioskysims_sources.infrastructure.db.satellite.models import (
    OrbitalElement,
    SatID,
    SatSignalLink,
)

logger = logging.getLogger(__name__)


class QueryMixin:
    """Métodos de resumo, consulta e loaders do repositório de satélites."""

    def _load_sats_summary(self) -> None:
        """
        Carrega resumo com SatID (norad_id), contagem de elementos e sinais.
        A versão async preenche self.sats_df (pandas).
        """
        with self.engine.session() as session:
            stmt = (
                select(
                    SatID.norad_id.label("norad_id"),
                    SatID.name,
                    SatID.constellation,
                    SatID.type,
                    func.count(distinct(OrbitalElement.id)).label("num_elements"),
                    func.count(distinct(SatSignalLink.signal_id)).label("num_signals"),
                )
                .outerjoin(OrbitalElement, SatID.norad_id == OrbitalElement.norad_id)
                .outerjoin(SatSignalLink, SatSignalLink.sat_id == SatID.norad_id)
                .group_by(SatID.norad_id, SatID.name, SatID.constellation, SatID.type)
            )
            result = session.execute(stmt)
            rows = result.all()

        if not rows:
            logger.warning("Nenhum satélite encontrado no banco.")
            self.sats_df = pd.DataFrame(
                columns=[
                    "norad_id",
                    "name",
                    "constellation",
                    "type",
                    "num_elements",
                    "num_signals",
                ]
            )
            return

        self.sats_df = pd.DataFrame([dict(zip(result.keys(), r, strict=False)) for r in rows])
        if not self.norad_ids:
            self.norad_ids = self.sats_df["norad_id"].tolist()

    def _epoch_from_tr(self) -> datetime | None:
        """
        Época de referência (datetime) derivada do time_range da simulação
        (``self.tr``), caso esteja definido.

        Permite que seleção de TLEs e checagens de frescor sejam relativas à
        janela da simulação em vez de "agora". Retorna ``None`` quando não há
        ``tr`` configurado.
        """
        if self.tr is not None:
            return Time(self.tr.to_tuple()[:2]).mean().to_datetime()
        return None

    def _query_satids(
        self, norad_ids: list[int] | None, epoch: datetime | None = None
    ) -> list[SatID]:
        """
        Retorna SatID pelo norad_id, incluindo sinais e os elementos (ou o elemento mais próximo do epoch).
        """
        if not norad_ids:
            norad_ids = self.norad_ids
        if not norad_ids and self.sats_df is not None:
            norad_ids = self.sats_df["norad_id"].tolist()
        if not norad_ids:
            return []
        if epoch is not None and not isinstance(epoch, datetime):
            msg = "epoch deve ser um objeto datetime ou None."
            raise ValueError(msg)
        if epoch is None:
            # Sem época explícita, usa a janela da simulação (self.tr) quando
            # disponível — assim o elemento "mais próximo" é relativo à data da
            # simulação, e não a "agora".
            epoch = self.epoch or self._epoch_from_tr()
        with self.engine.session() as session:
            if epoch is None:
                stmt = (
                    select(SatID)
                    .where(SatID.norad_id.in_(norad_ids))
                    .options(selectinload(SatID.elements), selectinload(SatID.signals))
                )
                res = session.execute(stmt)
                sats: list[SatID] = list(res.scalars().all())
            else:
                # carregamos os satélites com elementos e sinais, e depois escolhemos o elemento mais próximo em Python
                stmt = (
                    select(SatID)
                    .where(SatID.norad_id.in_(norad_ids))
                    .options(selectinload(SatID.elements), selectinload(SatID.signals))
                )
                res = session.execute(stmt)
                sats = list(res.scalars().all())

                for sat in sats:
                    if sat.elements:
                        # filtrar elementos com epoch definido
                        valid = [e for e in sat.elements if getattr(e, "epoch", None) is not None]
                        if valid:
                            # Datas no banco podem ser naive → normaliza para
                            # UTC antes da comparação (TODO: timezone-aware).
                            epoch_utc = as_utc(epoch)
                            closest = min(
                                valid,
                                key=lambda e: abs((as_utc(e.epoch) - epoch_utc).total_seconds()),
                            )
                            sat.elements = [closest]
        return sats

    def _to_domain_satellite(self, sat: SatID) -> DomainSatellite:
        if not sat.elements:
            msg = f"Satellite {sat.norad_id} has no orbital elements."
            raise ValueError(msg)
        elem = sat.elements[0]
        elem_data = elem.model_dump()
        domain_elem = DomainOrbitalElement.model_validate(elem_data)
        return DomainSatellite(
            norad_id=sat.norad_id,
            name=sat.name,
            constellation=sat.constellation,
            prn=getattr(sat, "prn", None),
            epoch=elem.epoch,
            element=domain_elem,
        )

    def _ensure_elements(self, sats: list[SatID], *, epoch: datetime | None = None) -> list[SatID]:
        """
        Best-effort: if a satellite is missing TLE/elements, try to fetch from Space-Track
        and re-query. This keeps selection APIs usable even when the DB is incomplete.
        """
        missing = [sat.norad_id for sat in sats if not getattr(sat, "elements", None)]
        if not missing:
            return sats

        try:
            # SatelliteRepository implements a richer signature than the port.
            self.update_orbital_elements_sync(norad_ids=missing, epoch=epoch)
        except TypeError:
            # Fallback to older/port-like signatures.
            try:
                self.update_orbital_elements_sync(epoch=epoch)
            except Exception as e:  # pragma: no cover
                logger.warning("Falha ao atualizar TLEs via Space-Track: %s", e)
                return sats
        except Exception as e:
            logger.warning("Falha ao atualizar TLEs via Space-Track: %s", e)
            return sats

        # Re-query only the missing ones and merge.
        refreshed = self._query_satids(missing, epoch=epoch)
        by_id = {sat.norad_id: sat for sat in refreshed}
        out: list[SatID] = []
        for sat in sats:
            out.append(by_id.get(sat.norad_id, sat))
        return out

    def query_sats(
        self, norad_ids: list[int] | None = None, epoch: datetime | None = None
    ) -> list[DomainSatellite]:
        sats = self._query_satids(norad_ids, epoch=epoch)
        sats = self._ensure_elements(sats, epoch=epoch)
        out: list[DomainSatellite] = []
        for sat in sats:
            try:
                out.append(self._to_domain_satellite(sat))
            except ValueError:
                continue
        return out

    def query_constellation(
        self, constellations: list[str], epoch: datetime | None = None
    ) -> list[DomainSatellite]:
        """Retorna satélites de uma ou mais constelações (usa query_sats para incluir relações)."""
        if not constellations:
            return []

        # Mapeamento reverso: nome padrão → nomes Celestrak
        _REVERSE_ALIAS: dict[str, list[str]] = {
            "GALILEO": ["GSAT", "GALILEO"],
            "GLONASS": ["COSMOS", "GLONASS"],
            "BEIDOU": ["BEIDOU"],
            "GPS": ["GPS"],
        }

        def _norm(s: str) -> str:
            return "".join(str(s).upper().split()).replace("-", "")

        # Expande constelações pedidas para incluir aliases
        expanded: list[str] = []
        for c in constellations:
            key = _norm(c)
            if key in _REVERSE_ALIAS:
                expanded.extend(_REVERSE_ALIAS[key])
            else:
                expanded.append(key)
        wanted = list({_norm(e) for e in expanded})

        norm_db = func.upper(func.replace(func.replace(SatID.constellation, " ", ""), "-", ""))

        with self.engine.session() as session:
            stmt = select(SatID).where(norm_db.in_(wanted))
            res = session.execute(stmt)
            sats = res.scalars().all()

        norad_ids = [sat.norad_id for sat in sats]
        # Popula o estado em memória para que atualizações posteriores de
        # elementos orbitais (update_orbital_elements) tenham os NORAD IDs.
        self.satellites = list(sats)
        return self.query_sats(norad_ids, epoch=epoch)

    def load_elements(
        self,
        norad_ids: list[int] | None = None,
        epoch: datetime | None = None,
        append: bool = False,
    ) -> list[SatID] | None:
        """Carrega satélites e elementos por norad_ids (usa self.norad_ids se None)."""
        if not norad_ids:
            norad_ids = self.norad_ids
        if not norad_ids:
            norad_ids = self.sats_df.norad_id.tolist() if self.sats_df is not None else None
        if not norad_ids:
            logger.warning("Nenhum norad_id fornecido.")
            return None
        sats = self._query_satids(norad_ids, epoch)
        if append:
            self.satellites.extend(sats)
        else:
            self.satellites = sats
        return sats

    def load_elements_from_constellation(
        self,
        constellations: list[str],
        epoch: datetime | None = None,
        append: bool = False,
    ) -> list[SatID] | None:
        if not constellations:
            return None

        def _norm(s: str) -> str:
            return "".join(str(s).upper().split()).replace("-", "")

        wanted = [_norm(c) for c in constellations]
        norm_db = func.upper(func.replace(func.replace(SatID.constellation, " ", ""), "-", ""))
        with self.engine.session() as session:
            stmt = select(SatID).where(norm_db.in_(wanted))
            res = session.execute(stmt)
            sats_basic = res.scalars().all()
        norad_ids = [sat.norad_id for sat in sats_basic]
        sats = self._query_satids(norad_ids, epoch=epoch)
        if append:
            self.satellites.extend(sats)
        else:
            self.satellites = sats
        return sats
