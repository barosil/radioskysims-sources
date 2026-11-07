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
from collections.abc import Sequence
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

    def __init__(self, tap_url: str | Sequence[str]) -> None:
        if TapPlus is None:  # pragma: no cover
            msg = (
                "astroquery não está instalado. Instale 'astroquery' para usar "
                "VizierTapIngestAdapter."
            )
            raise RuntimeError(msg)
        if pd is None:  # pragma: no cover
            msg = "pandas não está instalado. Instale 'pandas' para usar VizierTapIngestAdapter."
            raise RuntimeError(msg)

        # Aceita uma única URL ou uma sequência de servidores (fallback).
        self._tap_urls: list[str] = [tap_url] if isinstance(tap_url, str) else list(tap_url)
        if not self._tap_urls:
            msg = "É necessário informar ao menos um servidor TAP."
            raise ValueError(msg)

        # Mantém ``_tap_url``/``_tap`` apontando para o primeiro servidor
        # (compatibilidade com a API anterior).
        self._tap_url = self._tap_urls[0]
        self._tap = TapPlus(url=self._tap_url)
        self._tap_clients: dict[str, Any] = {self._tap_url: self._tap}

    def _tap_for(self, tap_url: str) -> Any:
        """Retorna (criando sob demanda) o cliente TapPlus de um servidor."""
        if tap_url not in self._tap_clients:
            self._tap_clients[tap_url] = TapPlus(url=tap_url)
        return self._tap_clients[tap_url]

    def ingest_from_tap(self, survey: str, adql_query: str) -> Any:
        """Executa uma query TAP/ADQL e retorna um ``pandas.DataFrame``.

        Tenta cada servidor TAP configurado em ordem; em caso de falha de um
        servidor, faz fallback para o próximo. Só falha se todos os servidores
        da lista falharem.

        Args:
            survey: Nome lógico do survey (usado apenas para logging).
            adql_query: Query ADQL completa a ser enviada ao serviço TAP.
        """

        last_exc: Exception | None = None
        for tap_url in self._tap_urls:
            try:
                logger.info("Executando TAP query para survey '%s' em %s", survey, tap_url)
                job = self._tap_for(tap_url).launch_job(adql_query)
                table = job.get_results()
                catalog_df = table.to_pandas()
                logger.info("TAP query retornou %d linhas", len(catalog_df))
                return catalog_df
            except Exception as exc:  # noqa: BLE001 - fallback para o próximo servidor
                last_exc = exc
                logger.warning(
                    "Servidor TAP %s falhou (%s: %s); tentando próximo da lista.",
                    tap_url,
                    type(exc).__name__,
                    exc,
                )

        msg = (
            "Todos os servidores TAP VizieR falharam "
            f"({len(self._tap_urls)} tentativas). Último erro: "
            f"{type(last_exc).__name__}: {last_exc}"
        )
        raise RuntimeError(msg) from last_exc

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
