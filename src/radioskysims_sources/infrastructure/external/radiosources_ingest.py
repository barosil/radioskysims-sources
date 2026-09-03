"""\
Módulo: src/radioskysims/infrastructure/external/radiosources_ingest.py

Adapters de ingestão para o contexto de RadioSources.

Implementa acesso básico a serviços TAP/ADQL (ex.: VizieR) e leitura
simples de arquivos locais, retornando tabelas pandas compatíveis com
os ports definidos em ``infrastructure/ports/radiosources.py``.

Frequências são mantidas na unidade nativa retornada pelo serviço
(geralmente MHz para catálogos de rádio), cabendo à normalização de
:class:`RadioSource` garantir a conversão apropriada quando necessário.

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:  # pragma: no cover - dependência opcional
    from astroquery.utils.tap.core import TapPlus
except ImportError:  # pragma: no cover
    TapPlus = None  # type: ignore[assignment]

try:  # pragma: no cover - dependência opcional
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from radioskysims_sources.infrastructure.ports.radiosources import CatalogIngestPort

logger = logging.getLogger(__name__)


class VizierTapIngestAdapter(CatalogIngestPort):
    """Adapter para ingestão de catálogos via TAP/ADQL (ex.: VizieR).

    Este adapter fornece uma implementação concreta de
    :class:`CatalogIngestPort` usando ``astroquery`` (``TapPlus``).
    O objetivo é ser robusto mas minimalista, adequado para uso em
    notebooks e pipelines científicos leves.
    """

    def __init__(self, tap_url: str) -> None:
        if TapPlus is None:  # pragma: no cover
            msg = (
                "astroquery não está instalado. Instale 'astroquery' para usar "
                "VizierTapIngestAdapter."
            )
            raise RuntimeError(msg)
        if pd is None:  # pragma: no cover
            msg = "pandas não está instalado. Instale 'pandas' para usar VizierTapIngestAdapter."
            raise RuntimeError(msg)

        self._tap = TapPlus(url=tap_url)
        self._tap_url = tap_url

    def ingest_from_tap(self, survey: str, adql_query: str) -> Any:
        """Executa uma query TAP/ADQL e retorna um ``pandas.DataFrame``.

        Args:
            survey: Nome lógico do survey (usado apenas para logging).
            adql_query: Query ADQL completa a ser enviada ao serviço TAP.
        """

        logger.info("Executando TAP query para survey '%s' em %s", survey, self._tap_url)
        job = self._tap.launch_job(adql_query)
        table = job.get_results()
        catalog_df = table.to_pandas()
        logger.info("TAP query retornou %d linhas", len(catalog_df))
        return catalog_df

    def ingest_from_file(self, path: Path, format: str) -> Any:
        """Lê um catálogo local e retorna um ``pandas.DataFrame``.

        Suporta inicialmente apenas CSV; outros formatos podem ser
        adicionados em iterações futuras.
        """

        if pd is None:  # pragma: no cover
            msg = "pandas não está instalado. Instale 'pandas' para usar ingest_from_file."
            raise RuntimeError(msg)

        if format.lower() == "csv":
            logger.info("Lendo catálogo local CSV em %s", path)
            return pd.read_csv(path)

        msg = f"Formato de arquivo não suportado para ingestão: {format!r}"
        raise ValueError(msg)
