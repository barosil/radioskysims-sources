from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from radioskysims_sources.application.ports.satellite import SatelliteRepositoryPort
from radioskysims_sources.domain.satellite import Satellite


class _FallbackStructuredLogger:
    """Fallback no-op para manter API quando logger nao for injetado."""

    def info(self, event: str, **fields: Any) -> None:
        _ = (event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        _ = (event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        _ = (event, fields)


class SatelliteService:
    """
    Serviço de aplicação para acesso e curadoria de satélites.

    Responsabilidades:
    - Selecionar satélites por critérios científicos
    - Garantir consistência dos dados orbitais
    - Delegar persistência ao repositório
    """

    def __init__(
        self,
        repository: SatelliteRepositoryPort,
        logger: Any | None = None,
    ) -> None:
        self._repository = repository
        self._logger = logger or _FallbackStructuredLogger()

    # ------------------------------------------------------------------
    # Casos de uso de leitura
    # ------------------------------------------------------------------

    def get_all(self) -> list[Satellite]:
        """
        Retorna todos os satélites disponíveis e válidos.
        """
        try:
            self._logger.info("satellite_get_all_start")
            satellites = self._repository.query_sats()
            self._logger.info("satellite_get_all_success", count=len(satellites))
            return satellites
        except Exception:
            self._logger.exception("satellite_get_all_failed")
            raise

    def get_by_ids(self, norad_ids: Iterable[int]) -> list[Satellite]:
        """
        Retorna satélites correspondentes aos NORAD IDs informados.
        """
        try:
            norad_ids_list = list(norad_ids)
            self._logger.info("satellite_get_by_ids_start", num_requested=len(norad_ids_list))
            satellites = self._repository.query_sats(norad_ids=norad_ids_list)
            self._logger.info("satellite_get_by_ids_success", num_returned=len(satellites))
            return satellites
        except Exception:
            self._logger.exception("satellite_get_by_ids_failed")
            raise

    def get_by_constellations(self, constellations: Iterable[str]) -> list[Satellite]:
        """
        Retorna satélites pertencentes às constelações informadas.

        Procedimento:
        1. Consulta o banco e popula ``self.satellites`` do repositório
           (a própria ``query_constellation`` cuida disso).
        2. Se o banco não tem os satélites, atualiza o catálogo via Celestrak
           e consulta novamente — a nova consulta volta a popular
           ``self.satellites``.
        3. Garante elementos orbitais recentes usando ``epoch`` como critério
           (via ``verify``), com os NORAD IDs já carregados em ``self.satellites``.
        """
        const_list = list(constellations)
        try:
            self._logger.info(
                "satellite_get_by_constellations_start",
                constellations=const_list,
            )

            # 1) Query ao banco → popula self.satellites do repositório
            sats = self._repository.query_constellation(constellations=const_list)

            # 2) Banco sem satélites → atualiza catálogo (Celestrak) e re-consulta
            if not sats:
                self._logger.info(
                    "satellite_get_by_constellations_empty",
                    constellations=const_list,
                    action="refresh_catalog_and_orbit",
                )
                # Atualiza catálogo de satélites (inclui novas constelações)
                self.update_from_celestrak()

                # Nova query de constelações → popula self.satellites novamente
                sats = self._repository.query_constellation(constellations=const_list)
                if not sats:
                    self._logger.warning(
                        "satellite_get_by_constellations_catalog_empty",
                        constellations=const_list,
                        detail=("Celestrak retornou 0 satélites para as constelações solicitadas"),
                    )
                    return sats

            # 3) Garante elementos orbitais recentes — epoch (via verify) é o
            #    critério. Como query_constellation populou self.satellites,
            #    o update usa os NORAD IDs corretos em vez de avisar
            #    "nenhum norad_id carregado".
            self.update_orbital_elements(epoch=None, force=False)

            # 4) Re-consulta após a atualização orbital: a query da etapa 1 é
            #    anterior ao update e não refletiria TLEs recém-buscados (ex.:
            #    quando a época da simulação difere de "agora", como uma
            #    simulação em 2025 com TLEs atuais em 2026).
            sats = self._repository.query_constellation(constellations=const_list)

            self._logger.info(
                "satellite_get_by_constellations_success",
                constellations=const_list,
                num_returned=len(sats),
                norad_ids=[s.norad_id for s in sats],
            )
            return sats
        except Exception:
            self._logger.exception(
                "satellite_get_by_constellations_failed",
                constellations=const_list,
            )
            raise

    # ------------------------------------------------------------------
    # Casos de uso de sincronização / curadoria
    # ------------------------------------------------------------------

    def update_orbital_elements(
        self,
        epoch: datetime | None = None,
        *,
        force: bool = False,
    ) -> None:
        """
        Atualiza os elementos orbitais a partir das fontes externas.

        Usa ``epoch`` como critério (via ``verify``) para decidir se a
        atualização é necessária. Os NORAD IDs são derivados do estado
        carregado no repositório (``self.satellites``), garantindo que a
        atualização atinja exatamente os satélites das constelações pedidas.

        Parameters
        ----------
        epoch : datetime | None
            Epoch de referência para os TLEs.
        force : bool
            Força atualização mesmo se os dados estiverem recentes.
        """
        try:
            self._logger.info(
                "satellite_update_orbital_elements_start", force=force, epoch=str(epoch)
            )
            # Deriva os NORAD IDs do estado carregado no repositório
            loaded = getattr(self._repository, "satellites", None) or []
            norad_ids = [sat.norad_id for sat in loaded]

            if force or not self._repository.verify(epoch=epoch):
                self._repository.update_orbital_elements_sync(
                    norad_ids=norad_ids or None,
                    epoch=epoch,
                )
                self._logger.info(
                    "satellite_update_orbital_elements_success",
                    num_satellites=len(norad_ids),
                )
            else:
                self._logger.info("satellite_update_orbital_elements_skipped", reason="data_recent")
        except Exception:
            self._logger.exception("satellite_update_orbital_elements_failed")
            raise

    def update_from_celestrak(self) -> None:
        """
        Atualiza o catálogo de satélites a partir do Celestrak.
        """
        try:
            self._logger.info("satellite_update_from_celestrak_start")
            self._repository.update_satellites_from_celestrak()
            self._logger.info("satellite_update_from_celestrak_success")
        except Exception:
            self._logger.exception("satellite_update_from_celestrak_failed")
            raise

    def reset(self) -> None:
        """
        Limpa o estado interno do repositório (útil para testes).
        """
        try:
            self._logger.info("satellite_reset_start")
            self._repository.reset()
            self._logger.info("satellite_reset_success")
        except Exception:
            self._logger.exception("satellite_reset_failed")
            raise
