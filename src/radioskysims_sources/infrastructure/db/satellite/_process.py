"""Mixin de processamento/verificação do repositório de satélites (E8/TD-011).

Agrupa ``_process``, ``reset``, ``process`` e ``verify`` do
``SatelliteRepository``. Extraído de ``satelliterepository.py``.

Copyright BINGO Collaboration
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from radioskysims_sources.infrastructure.db.satellite.models import SatID

if TYPE_CHECKING:
    from radioskysims_sources.infrastructure.db.satellite.satelliterepository import (
        SatelliteRepository,
    )

logger = logging.getLogger(__name__)


class ProcessMixin:
    """Métodos de orquestração do processo de carregamento/verificação."""

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
