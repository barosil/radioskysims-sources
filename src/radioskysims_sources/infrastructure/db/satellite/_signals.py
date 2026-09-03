"""Mixin de carregamento de sinais GNSS (E8/TD-011).

Agrupa os métodos de leitura/parse do JSON de sinais e associação aos
``SatID`` do ``SatelliteRepository``. Extraído de ``satelliterepository.py``.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
from sqlmodel import func, select

from radioskysims_sources.infrastructure.db.satellite.models import (
    SatID,
    SatSignalLink,
    Signal,
)
from radioskysims_sources.infrastructure.external.satellite_downloader import (
    CONSTELLATION_ALIAS,
)

logger = logging.getLogger(__name__)


class SignalsMixin:
    """Métodos de carregamento/associação de sinais GNSS (JSON)."""

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
