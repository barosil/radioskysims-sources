from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
from astropy.time import Time
from sqlalchemy.orm import selectinload
from sqlmodel import distinct, func, select

from radioskysims.application.ports import SatelliteRepositoryPort
from radioskysims_core.domain.common import TimeRange
from radioskysims_sources.domain.satellite.models import OrbitalElement as DomainOrbitalElement
from radioskysims_sources.domain.satellite.models import Satellite as DomainSatellite
from radioskysims_core.db.engine import Database
from radioskysims_sources.infrastructure.db.satellite.models import (
    OrbitalElement,
    SatID,
    SatSignalLink,
    Signal,
)
from radioskysims_sources.infrastructure.external.satellite_downloader import (
    CONSTELLATION_ALIAS,
    CelestrakClient,
    SpaceTrackClient,
)
from radioskysims_core.logging_config import get_logger

logger = logging.getLogger(__name__)
slogger = get_logger(__name__)


class SatelliteRepository(SatelliteRepositoryPort):
    def __init__(
        self,
        engine: Database,
        tr: TimeRange | None = None,
        norad_ids: list[int] | None = None,
        constellations: list[str] | None = None,
    ):
        self.engine = engine
        self.tr = tr
        self.norad_ids: list[int] = norad_ids or []
        self.constellations = constellations or []
        self.epoch: datetime | None = None

        # caches / in-memory structures
        self.sats_df: pd.DataFrame | None = None
        self.failed_sats_df: pd.DataFrame | None = None
        self.satellites: list[SatID] = []
        self._load_sats_summary()
        self.processed_sats: list[SatID] = []

    # ---------------------------
    # Signals JSON helpers
    # ---------------------------
    def __repr__(self) -> str:
        """
        Retorna um resumo legível do catálogo de satélites.

        Inclui uma visão geral das constelações disponíveis, com:
            - Número total de satélites
            - Número de NORAD IDs distintos
            - Mínimo de `num_elements`
            - Mínimo de `num_signals`

        Returns
        -------
        str
            Texto formatado resumindo o conteúdo de `sats_df`.
        """
        if not hasattr(self, "sats_df") or self.sats_df is None or self.sats_df.empty:
            return "<Catalog vazio — nenhum satélite carregado>"

        sats_df = self.sats_df
        agg_df = (
            sats_df.groupby("constellation")
            .agg(
                distinct_norad_ids=("norad_id", "nunique"),
                min_num_elements=("num_elements", "min"),
                min_num_signals=("num_signals", "min"),
            )
            .reset_index()
            .sort_values("distinct_norad_ids", ascending=False)
        )

        formatted = agg_df.to_string(
            index=False,
            formatters={
                "distinct_norad_ids": "{:>6}".format,
                "min_num_elements": "{:>6}".format,
                "min_num_signals": "{:>6}".format,
            },
        )

        total = len(sats_df)
        header = f"<Catalog — {total} satélites em {agg_df.constellation.nunique()} constelações>"
        line = "─" * len(header)

        return f"{header}\n{line}\n{formatted}\n"

    def _read_signals_json(self, signals_file: Path | None = None) -> list[dict[str, Any]]:
        if signals_file is None:
            resource = files("radioskysims.resources").joinpath("satellites/sat_signals.json")
            signals_file = Path(str(resource))
        try:
            with signals_file.open(encoding="utf-8") as f:
                result: list[dict[str, Any]] = json.load(f)
                return result
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Erro ao carregar JSON de sinais (%s): %s", signals_file, e)
            return []

    def _make_signals(
        self, satellites: Iterable[SatID], signals_file: Path | None = None
    ) -> tuple[list[Any], list[Signal]]:
        """
        Gera objetos Signal a partir do JSON, associando ao atributo constellation dos SatID passados.
        Retorna (norad_ids_list, [Signal, ...])
        """
        sat_signals = self._read_signals_json(signals_file)
        if not sat_signals:
            return [], []

        # construir DataFrame consolidado de sinais por constelação
        dfs = []
        for entry in sat_signals:
            cst = entry.get("constellation")
            sig_df = pd.DataFrame(entry.get("signals", []))
            sig_df["constellation"] = cst
            dfs.append(sig_df)

        sig_df = pd.concat(dfs, ignore_index=True)
        # normalize columns (a ordem no JSON pode variar)
        # esperamos colunas: center_frequency, modulation, name
        # renomear/filtrar por segurança
        expected = ["frequency", "modulation", "signal_name", "constellation"]
        for col in expected:
            if col not in sig_df.columns:
                sig_df[col] = None
        sig_df = sig_df[expected]

        # para cada sat, filtrar por constelação, normalizando o nome do banco
        # (ex.: GALILEO/GLONASS) para o nome usado no JSON (GSAT/COSMOS)
        reverse_map = {v: k for k, v in CONSTELLATION_ALIAS.items()}
        dfs = []
        norad_ids = []
        for sat in satellites:
            json_const = reverse_map.get(sat.constellation, sat.constellation)
            sat_df = sig_df[sig_df["constellation"] == json_const].copy()
            if sat_df.empty:
                continue
            sat_df["norad_id"] = sat.norad_id
            dfs.append(sat_df)
            norad_ids.extend([sat.norad_id] * len(sat_df))

        if not dfs:
            return [], []

        sat_df = pd.concat(dfs, ignore_index=True).dropna(subset=["signal_name", "frequency"])

        # converte frequências (ex: "1575.42 MHz" -> 1575.42)
        def _to_float(s):
            try:
                return float(str(s).split()[0])
            except ValueError:
                logger.warning("Não foi possível converter frequência para float: %s", s)
                return None

        sat_df["frequency"] = sat_df["frequency"].apply(_to_float)
        sat_df = sat_df.dropna(subset=["frequency"])

        signals = [
            Signal(
                signal_name=row["signal_name"],
                modulation=row["modulation"] or "",
                frequency=float(row["frequency"]),
            )
            for _, row in sat_df.iterrows()
        ]
        norad_list = sat_df["norad_id"].tolist()
        return norad_list, signals

    def _signals_from_json(
        self,
        satellites: Iterable[int] | Iterable[SatID],
        signals_file: Path | None = None,
    ) -> None:
        """
        Carrega sinais GNSS a partir de um JSON e associa a cada SatID.
        `satellites` pode ser:
         - lista de SatID objects
         - OU lista de NORAD_ID ints
        """
        # garantir objetos SatID
        sat_objs: list[SatID] = []
        satellites_list: list[int | SatID] = list(satellites)
        with self.engine.session() as session:
            # detecta tipo dos itens
            if not satellites_list:
                return
            if all(isinstance(x, int) for x in satellites_list):
                # buscar SatID para cada NORAD_ID
                sat_ids_list: list[int] = [x for x in satellites_list if isinstance(x, int)]
                result = session.execute(select(SatID).where(SatID.norad_id.in_(sat_ids_list)))
                sat_objs = list(result.scalars().all())
            else:
                sat_objs = [x for x in satellites_list if isinstance(x, SatID)]

            if not sat_objs:
                logger.warning("Nenhum SatID encontrado para carregar sinais.")
                return

        # gerar objetos Signal localmente
        norad_ids, signals = self._make_signals(sat_objs, signals_file)
        if not signals:
            logger.warning("Nenhum sinal definido no JSON para as constelações fornecidas.")
            return

        # associar no banco
        with self.engine.session() as session:
            for norad, sig in zip(norad_ids, signals, strict=False):
                sat_row = session.execute(select(SatID).where(SatID.norad_id == norad))
                sat = sat_row.scalars().first()
                if not sat:
                    logger.warning("Satélite com norad_id=%s não encontrado no banco", norad)
                    continue

                # checar existência por nome + freq + modulation
                existing_q = select(Signal).where(
                    Signal.signal_name == sig.signal_name,
                    func.abs(Signal.frequency - sig.frequency) < 1e-6,
                    Signal.modulation == sig.modulation,
                )
                res = session.execute(existing_q)
                existing_signal = res.scalars().first()

                if existing_signal:
                    sig_to_assoc = existing_signal
                else:
                    session.add(sig)
                    session.flush()  # para obter id
                    sig_to_assoc = sig

                # link via SatSignalLink se não existir
                link_q = select(SatSignalLink).where(
                    SatSignalLink.sat_id == sat.norad_id,
                    SatSignalLink.signal_id == sig_to_assoc.id,
                )
                r = session.execute(link_q)
                link = r.scalars().first()
                if not link:
                    new_link = SatSignalLink(sat_id=sat.norad_id, signal_id=sig_to_assoc.id)
                    session.add(new_link)

            session.commit()

    # ---------------------------
    # Summary loader (async)
    # ---------------------------
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

    # ---------------------------
    # Query methods requested
    # ---------------------------
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
            raise ValueError("epoch deve ser um objeto datetime ou None.")
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
                            closest = min(
                                valid,
                                key=lambda e: abs((e.epoch - epoch).total_seconds()),
                            )
                            sat.elements = [closest]
        return sats

    def _to_domain_satellite(self, sat: SatID) -> DomainSatellite:
        if not sat.elements:
            raise ValueError(f"Satellite {sat.norad_id} has no orbital elements.")
        elem = sat.elements[0]
        elem_data = elem.model_dump()
        domain_elem = DomainOrbitalElement.model_validate(elem_data)
        return DomainSatellite(
            norad_id=sat.norad_id,
            name=sat.name,
            constellation=sat.constellation,
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

    # ---------------------------
    # Loaders / orchestrators
    # ---------------------------
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

    def load_signals(self, sats: list[SatID] | None = None) -> list[SatID] | None:
        """
        Para os satélites carregados em self.satellites, identifica quais não têm sinais
        e tenta carregá-los a partir do JSON de sinais.
        """
        if sats is None:
            sats = self.satellites
            return None
        selected_sats = [sat.norad_id for sat in sats]
        if self.sats_df is None:
            self._load_sats_summary()
        if self.sats_df is None:
            return None
        # considerar 0 ou NaN como sem sinais
        no_signals = self.sats_df[self.sats_df["num_signals"].fillna(0) == 0]["norad_id"].tolist()
        sats_to_load = list(set(selected_sats) & set(no_signals))
        if not sats_to_load:
            return None
        self._signals_from_json(sats_to_load)
        # atualizar resumo
        self._load_sats_summary()
        return self._query_satids(sats_to_load)

    def _process(
        self,
        norad_ids: list[int] | None = None,
        constellations: list[str] | None = None,
        epoch: datetime | None = None,
    ) -> list[SatID] | None:
        """Processa os satélites com base nos NORAD_IDs/constelações fornecidas."""
        sats_from_ids: list[SatID] | None = []
        sats_from_constellations: list[SatID] | None = []
        if not norad_ids:
            norad_ids = self.norad_ids
        if not constellations:
            constellations = self.constellations or []
        sats_from_ids = self.load_elements(norad_ids, append=True, epoch=epoch)
        if len(constellations) > 0:
            sats_from_constellations = self.load_elements_from_constellation(
                constellations, append=True, epoch=epoch
            )
        sats = (sats_from_ids or []) + (sats_from_constellations or [])
        return self.load_signals(sats=sats)

    def reset(self, kind: str = "all") -> SatelliteRepository:
        """Reseta o estado interno do repositório."""
        if kind == "all":
            self.satellites = []
            self.failed_sats_df = pd.DataFrame()
            self.norad_ids = []
            self.constellations = []
            self.sats_df = pd.DataFrame()
        elif kind == "satellites":
            self.satellites = []
        elif kind == "norad_ids":
            self.norad_ids = []
        elif kind == "constellations":
            self.constellations = []
        elif kind == "selection":
            self.failed_sats_df = pd.DataFrame()
        elif kind == "processed_sats":
            self.processed_sats = []
        return self

    def process(
        self,
        norad_ids: list[int] | None = None,
        constellations: list[str] | None = None,
        epoch: datetime | None = None,
    ) -> None:
        """
        Orquestra processo de carregamento: elementos -> sinais -> verificação -> update se necessário -> recarregar.
        """
        self.reset("satellites").reset("norad_ids")
        sats_01 = self._process(norad_ids, constellations, epoch=epoch) or []
        sats_01_ok, sats_01_no_elements, sats_01_no_signals = self.verify(sats_01)
        self.processed_sats = [sat for sat in self.satellites if sat.norad_id in sats_01_ok]
        if sats_01_no_elements:
            logger.info(
                "Número de satélites sem elementos orbitais: %s",
                len(sats_01_no_elements),
            )
            # rodar a atualização (sync wrapper que executa o async internamente)
            self.update_orbital_elements_sync(norad_ids=sats_01_no_elements)
        # recarregar elementos
        self.reset("satellites").reset("norad_ids")
        sats_02 = self._process(norad_ids, constellations, epoch=epoch) or []
        sats_02_ok, sats_02_no_elements, sats_02_no_signals = self.verify(sats_02)

    def verify(
        self,
        satellites: list[SatID] | None = None,
        epoch: datetime | None = None,
    ) -> bool | tuple[list[int], list[int], list[int]]:
        """
        Verifica integridade e/ou atualidade dos dados carregados.

        - Se `satellites` for informado: retorna tuple
          (sats_ok, sats_no_elements, sats_no_signals)
        - Se `satellites` for None: retorna bool para uso do serviço de aplicação
          (`True` quando há satélites com elementos válidos e recentes para epoch).
        """
        if satellites is None:
            # A "atualidade" deve ser medida em relação à janela da simulação
            # (self.tr), não a "agora": TLEs recentes hoje podem ser inválidos
            # para uma simulação em outra data (ex.: 2025).
            epoch_ref = epoch or self.epoch or self._epoch_from_tr() or datetime.utcnow()
            # Escopo: verifica apenas os satélites atualmente selecionados
            # (self.satellites), com fallback para norad_ids/sats_df. Antes
            # verificava todos os satélites do catálogo, o que disparava
            # atualizações desnecessárias para constelações fora do escopo.
            selected = self.satellites or []
            if selected:
                norad_ids = [s.norad_id for s in selected]
            else:
                norad_ids = self.norad_ids or None
            sats = self._query_satids(norad_ids, epoch=epoch_ref)
            if not sats:
                return False

            max_age_sec = 7 * 24 * 3600  # 7 dias
            for sat in sats:
                if not getattr(sat, "elements", None):
                    return False
                elem = sat.elements[0]
                elem_epoch = getattr(elem, "epoch", None)
                if elem_epoch is None:
                    return False
                if abs((elem_epoch - epoch_ref).total_seconds()) > max_age_sec:
                    return False
            return True

        sats_ok = []
        sats_no_elements = []
        sats_no_signals = []
        for sat in satellites or []:
            has_elements = bool(getattr(sat, "elements", None))
            has_signals = bool(getattr(sat, "signals", None))
            if not has_elements:
                sats_no_elements.append(sat.norad_id)
            if not has_signals:
                sats_no_signals.append(sat.norad_id)
            if has_elements and has_signals:
                sats_ok.append(sat.norad_id)
        return sats_ok, sats_no_elements, sats_no_signals

    # ---------------------------
    # External Updates
    # ---------------------------
    def update_satellites_from_celestrak(self) -> None:
        """Baixa lista de satélites do Celestrak e insere/atualiza a base."""
        try:
            celestrak = CelestrakClient().fetch_sats()
            sats_df = celestrak  # espera-se um pandas.DataFrame
            sats = [SatID(**row.to_dict()) for _, row in sats_df.iterrows()]

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

                # checar existência por nome + freq + modulation
                existing_q = select(OrbitalElement).where(OrbitalElement.epoch == elem.epoch)
                res = session.execute(existing_q)
                existing_element = res.scalars().first()

                if not existing_element:
                    session.add(elem)
                    session.flush()  # para obter id

            session.commit()

        logger.info("Atualizados %d elementos orbitais via SpaceTrack.", len(elements))


# End of file: src/radioskysims/infrastructure/db/satellite/satelliterepository.py
