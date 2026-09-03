"""Repositório de satélites (casca; E8/TD-011).

``SatelliteRepository`` (antes ~890 l. num único módulo) foi decomposto por
responsabilidade em mixins, mantendo a API pública e o caminho de import:

- ``_signals.py`` → ``SignalsMixin``  (leitura/parse do JSON de sinais)
- ``_query.py``   → ``QueryMixin``    (resumo, query por ids/constelação,
  loaders e conversão para o domínio)
- ``_process.py`` → ``ProcessMixin``  (orquestração do carregamento,
  ``reset``/``process``/``verify``)
- ``_updates.py`` → ``UpdatesMixin``  (atualizações Celestrak/SpaceTrack)
- ``_convert.py`` → helpers ``_safe_float``/``_safe_int``

Copyright BINGO Collaboration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from radioskysims_sources.application.ports.satellite import SatelliteRepositoryPort
from radioskysims_sources.infrastructure.db.satellite._process import ProcessMixin
from radioskysims_sources.infrastructure.db.satellite._query import QueryMixin
from radioskysims_sources.infrastructure.db.satellite._signals import SignalsMixin
from radioskysims_sources.infrastructure.db.satellite._updates import UpdatesMixin

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd
    from radioskysims_core.db.engine import Database
    from radioskysims_core.domain.common import TimeRange

    from radioskysims_sources.infrastructure.db.satellite.models import SatID


class SatelliteRepository(
    SignalsMixin,
    QueryMixin,
    ProcessMixin,
    UpdatesMixin,
    SatelliteRepositoryPort,
):
    """Repositório SQLModel/SQLAlchemy de satélites (TLE, sinais e órbitas)."""

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
