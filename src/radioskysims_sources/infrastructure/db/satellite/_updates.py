"""Mixin de atualizações externas do repositório de satélites (E8/TD-011).

Agrupa os métodos de atualização via Celestrak (catálogo) e SpaceTrack
(elementos orbitais, async) do ``SatelliteRepository``. Extraído de
``satelliterepository.py``.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import pandas as pd
from astropy.time import Time
from radioskysims_core.logging_config import get_logger
from sqlmodel import select

from radioskysims_sources.domain.satellite.prn import parse_prn_from_name
from radioskysims_sources.infrastructure.db.satellite._convert import _safe_float, _safe_int
from radioskysims_sources.infrastructure.db.satellite.migration import ensure_prn_column
from radioskysims_sources.infrastructure.db.satellite.models import OrbitalElement, SatID
from radioskysims_sources.infrastructure.external.satellite_downloader import (
    CelestrakClient,
    SpaceTrackClient,
)

logger = logging.getLogger(__name__)
slogger = get_logger(__name__)


class UpdatesMixin:
    """Métodos de atualização externa (Celestrak/SpaceTrack)."""

    def update_satellites_from_celestrak(self) -> None:
        """Baixa lista de satélites do Celestrak e insere/atualiza a base."""
        try:
            ensure_prn_column(self.engine)
            celestrak = CelestrakClient().fetch_sats()
            sats_df = celestrak  # espera-se um pandas.DataFrame

            def _to_satid(row: pd.Series) -> SatID:
                # Mapeamento explícito: o DataFrame do CelesTrak tem colunas
                # além das aceitas pelo SatID (ex.: 'id' = designador
                # internacional) — não usar ``SatID(**row.to_dict())``.
                return SatID(
                    norad_id=int(row["norad_id"]),
                    name=str(row["name"]),
                    period=_safe_float(row.get("period")),
                    inclination=_safe_float(row.get("inclination")),
                    apogee=_safe_float(row.get("apogee")),
                    perigee=_safe_float(row.get("perigee")),
                    eccentricity=_safe_float(row.get("eccentricity")),
                    gp_age=_safe_float(row.get("gp_age")),
                    constellation=str(row.get("constellation") or None),
                    prn=_safe_int(row.get("prn")),
                )

            sats = [_to_satid(row) for _, row in sats_df.iterrows()]

            with self.engine.session() as session:
                existing_res = session.execute(select(SatID))
                existing = {s.norad_id: s for s in existing_res.scalars().all()}

                new_count = 0
                update_count = 0
                for sat in sats:
                    norad = int(sat.norad_id)
                    if norad in existing:
                        # Atualiza registro existente (ex.: nome da constelação)
                        old = existing[norad]
                        if old.constellation != sat.constellation:
                            old.constellation = sat.constellation
                            update_count += 1
                        # Backfill do PRN quando ainda ausente
                        if getattr(old, "prn", None) is None and sat.prn is not None:
                            old.prn = sat.prn
                            update_count += 1
                    else:
                        session.add(sat)
                        new_count += 1

                session.commit()
            logger.info(
                "Celestrak: %d novos, %d atualizados (total=%d).",
                new_count,
                update_count,
                len(sats),
            )
        except (ValueError, TypeError) as e:
            logger.exception("Erro ao atualizar satélites do Celestrak: %s", e)

    def update_orbital_elements_sync(
        self,
        norad_ids: list[int] | None = None,
        sats: list[SatID] | None = None,
        epoch: datetime | None = None,
    ) -> None:
        """Wrapper síncrono para chamar o método async update_orbital_elements."""
        asyncio.run(self.update_orbital_elements(norad_ids=norad_ids, sats=sats, epoch=epoch))

    async def update_orbital_elements(
        self,
        norad_ids: list[int] | None = None,
        sats: list[SatID] | None = None,
        epoch: datetime | None = None,
    ) -> None:
        """Atualiza elementos orbitais via SpaceTrackClient (async)."""
        ensure_prn_column(self.engine)
        # intervalo de tempo
        if self.tr:
            epoch = Time(self.tr.to_tuple()[:2]).mean().to_datetime()
        elif epoch is None:
            epoch = self.epoch if self.epoch is not None else datetime.now()

        # IDs alvo
        if sats is not None:
            norad_ids = [sat.norad_id for sat in sats]
        elif norad_ids is None or not norad_ids:
            norad_ids = [sat.norad_id for sat in self.satellites or []]

        if not norad_ids:
            logger.warning("Nenhum norad_id fornecido ou carregado em self.satellites")
            return

        slogger.info(
            "spacetrack_update_start",
            requested_count=len(norad_ids),
            norad_ids=norad_ids,
            epoch=str(epoch),
        )
        st_client = SpaceTrackClient(epoch=epoch)
        fetched = await st_client.fetch_elements(norad_ids)  # espera DataFrame pandas

        if fetched is None or fetched.empty:
            slogger.warning(
                "spacetrack_update_empty_result",
                requested_count=len(norad_ids),
                norad_ids=norad_ids,
                detail=(
                    "SpaceTrack retornou 0 TLEs. Pode significar que: "
                    "(1) todos os NORAD IDs já foram consultados hoje "
                    "(cache diário); (2) as credenciais são inválidas; "
                    "(3) não há TLEs disponíveis para os NORAD IDs "
                    "solicitados no período."
                ),
            )
            return

        # Log detalhado do que foi retornado
        returned_norad_ids = (
            fetched["norad_id"].unique().tolist() if "norad_id" in fetched.columns else []
        )
        missing_norad_ids = list(set(norad_ids) - set(returned_norad_ids))
        slogger.info(
            "spacetrack_update_received",
            returned_count=len(fetched),
            unique_norad_ids=len(returned_norad_ids),
            norad_ids_with_data=returned_norad_ids,
            missing_norad_ids=missing_norad_ids,
            detail=(
                f"De {len(norad_ids)} NORAD IDs solicitados, "
                f"{len(returned_norad_ids)} retornaram TLEs e "
                f"{len(missing_norad_ids)} não retornaram dados."
            ),
        )

        # Ajustando dtypes
        dtypes = [
            {field: str(info.annotation).split("|")[0].strip()}
            for field, info in OrbitalElement.model_fields.items()
        ]
        for col in fetched.columns:
            if col in dtypes:
                fetched = fetched.astype(dtypes[col])
        fetched = fetched.where(pd.notnull(fetched), None)
        elements = []

        for _, row in fetched.iterrows():
            # converter nomes/colunas conforme seu modelo OrbitalElement
            d = row.to_dict()
            # Converte NaT/NaN (pandas) para None: colunas datetime64
            # reconvertem None→NaT dentro do .where() acima, e o Postgres
            # rejeita a string "NaT" como timestamp.
            d = {key: (None if pd.isna(value) else value) for key, value in d.items()}
            elements.append(OrbitalElement(**d))

        # associar no banco
        with self.engine.session() as session:
            for elem in elements:
                sat_row = session.execute(select(SatID).where(SatID.norad_id == elem.norad_id))
                sat = sat_row.scalars().first()
                if not sat:
                    logger.warning(
                        "Satélite com norad_id=%s não encontrado no banco",
                        elem.norad_id,
                    )
                    continue

                # Backfill do PRN quando ainda ausente — o OBJECT_NAME do
                # Space-Track segue o mesmo padrão CelesTrak (ex.:
                # "GPS BIIR-5 (PRN 22)").
                if getattr(sat, "prn", None) is None:
                    prn = parse_prn_from_name(
                        getattr(elem, "object_name", None) or sat.name,
                        sat.constellation,
                    )
                    if prn is not None:
                        sat.prn = prn

                # checar existência por nome + freq + modulation
                existing_q = select(OrbitalElement).where(OrbitalElement.epoch == elem.epoch)
                res = session.execute(existing_q)
                existing_element = res.scalars().first()

                if not existing_element:
                    session.add(elem)
                    session.flush()  # para obter id

            session.commit()

        logger.info("Atualizados %d elementos orbitais via SpaceTrack.", len(elements))
