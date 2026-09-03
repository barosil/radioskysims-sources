"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/orchestrators/satellite_orchestrator.py

Descrição:
    Fachada de alto nível (UX — Onda 3/R6) para o contexto de satélites GNSS.

    Esconde a montagem do repositório (``build_repositories``) e do
    ``SatelliteManager``: o usuário pede a constelação e recebe tracks/sinais
    sem conhecer bootstrap/repositórios.

    Uso (notebook):
        >>> orch = SatelliteOrchestrator.create_from_constellation("GPS")
        >>> tracks = orch.tracks(timerange)          # dataset ECI (km)
        >>> assembled = orch.build(timerange, freq)  # tracks + sinais

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from radioskysims_core.domain.common import FrequencyRange, TimeRange

from radioskysims_sources.application.services.satellite_manager import SatelliteManager
from radioskysims_sources.domain.satellite.models import Satellite, SatelliteAssembled

__all__ = ["SatelliteOrchestrator"]


class SatelliteOrchestrator:
    """Orquestrador/fachada pública de satélites GNSS.

    Encapsula a construção do ``SatelliteManager`` (com o repositório default
    de TLE via banco/Celestrak) e o ciclo constelação → tracks → sinais.
    """

    def __init__(self, manager: SatelliteManager) -> None:
        self._manager = manager

    # ──────────────────────────────────────────────────────────
    # Estado
    # ──────────────────────────────────────────────────────────
    @property
    def satellites(self) -> list[Satellite]:
        """Satélites da seleção atual (última constelação criada)."""
        return self._manager.satellites

    def get_tle(self) -> list[dict[str, Any]]:
        """Linhas TLE (2LE) dos satélites selecionados."""
        return self._manager.get_tle()

    # ──────────────────────────────────────────────────────────
    # Fábricas
    # ──────────────────────────────────────────────────────────
    @classmethod
    def from_manager(cls, manager: SatelliteManager) -> "SatelliteOrchestrator":
        """Embrulha um ``SatelliteManager`` já montado (injeção em testes/uso)."""
        return cls(manager)

    @classmethod
    def _default_manager(
        cls,
        *,
        epoch: datetime | None,
        settings_file: str | None = None,
    ) -> SatelliteManager:
        """Monta o ``SatelliteManager`` default (Database + build_repositories).

        O banco (PostgreSQL) é o repositório de TLE; se estiver indisponível,
        a primeira consulta falhará com erro claro.
        """
        try:
            from radioskysims_core.config.config_loader import set_settings_file
            from radioskysims_core.db.engine import Database

            from radioskysims_sources.bootstrap.builders import build_repositories
        except Exception as exc:  # noqa: BLE001 - erro de montagem amigável
            msg = (
                "Falha ao montar dependências de satélite (módulos de banco/"
                "bootstrap não disponíveis): "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc

        if settings_file is not None:
            set_settings_file(settings_file)
        db = Database()
        repos = build_repositories(db)
        return SatelliteManager(
            repository=repos["satellite_repository"],  # type: ignore[arg-type]
            signal_repository=repos["signal_repository"],  # type: ignore[arg-type]
            epoch=epoch,
        )

    @classmethod
    def create_from_constellation(
        cls,
        constellation: str | list[str],
        *,
        epoch: datetime | None = None,
        method: str = "TLE",
        manager: SatelliteManager | None = None,
    ) -> "SatelliteOrchestrator":
        """Cria satélites de uma constelação e retorna o orquestrador pronto.

        Args:
            constellation: Nome(s) da constelação (ex.: ``"GPS"`` ou
                ``["GPS", "GALILEO"]``).
            epoch: Época UTC de referência (default: agora).
            method: Fonte dos elementos orbitais — ``"TLE"`` (default, via
                banco/Celestrak) ou ``"RINEX"`` (integração futura com
                ``gnsstools4astro``, repo próprio).
            manager: ``SatelliteManager`` opcional (injeção); quando omitido,
                monta o default com banco.

        Raises:
            NotImplementedError: se ``method="RINEX"`` (integração pendente).
            ValueError/RuntimeError: se a constelação não puder ser carregada.
        """
        method_norm = (method or "TLE").upper()
        if method_norm == "RINEX":
            msg = (
                "Fonte RINEX ainda não integrada ao SatelliteOrchestrator; a "
                "integração virá via gnsstools4astro (repo próprio). Use "
                "method='TLE' (Celestrak/Space-Track via banco)."
            )
            raise NotImplementedError(msg)

        names = [constellation] if isinstance(constellation, str) else list(constellation)
        if not names:
            msg = "Informe ao menos uma constelação (ex.: 'GPS')."
            raise ValueError(msg)

        if manager is None:
            manager = cls._default_manager(epoch=epoch)
        orch = cls(manager)
        try:
            orch._manager.create_from_constellations(names, epoch=epoch)
        except Exception as exc:  # noqa: BLE001 - mensagem amigável (R3)
            msg = (
                f"Não foi possível carregar a constelação {names!r}. Verifique "
                f"se o banco de TLE está acessível e populado "
                f"(SatelliteRepository/Celestrak): {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        return orch

    @classmethod
    def from_rinex(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> "SatelliteOrchestrator":
        """(Futuro) Monta o orquestrador com elementos de arquivos RINEX.

        A integração RINEX vive no repositório próprio ``gnsstools4astro``
        (georinex) e será plugada aqui via port — acompanhar o plano de
        consolidação (Anexo A).
        """
        _ = (args, kwargs)
        msg = (
            "SatelliteOrchestrator.from_rinex ainda não está disponível; a "
            "integração RINEX virá via gnsstools4astro (repo próprio)."
        )
        raise NotImplementedError(msg)

    # ──────────────────────────────────────────────────────────
    # Fluxo (delega ao SatelliteManager)
    # ──────────────────────────────────────────────────────────
    def tracks(self, timerange: TimeRange):
        """Constrói o dataset de tracks (posições ECI em km) no intervalo."""
        return self._manager.build_tracks(timerange)

    def signals(
        self,
        timerange: TimeRange,
        frequency_range: FrequencyRange,
    ):
        """Constrói o dataset de sinais (PSD por satélite/sinal)."""
        return self._manager.build_signals(timerange, frequency_range)

    def build(
        self,
        timerange: TimeRange,
        frequency_range: FrequencyRange,
    ) -> SatelliteAssembled:
        """Monta o ``SatelliteAssembled`` completo (tracks + sinais)."""
        return self._manager.build(timerange, frequency_range)
