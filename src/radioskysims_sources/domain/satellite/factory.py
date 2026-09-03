from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from astropy.time import Time

from radioskysims_core.domain.common import TimeRange
from radioskysims_core.domain.common.logging import get_domain_logger

from .models import Satellite
from .selector import TLESelector

logger = get_domain_logger(__name__)


class SatelliteRepositoryProtocol(Protocol):
    """Protocolo de domínio para o que a SatelliteFactory precisa de um repositório."""

    def load_elements(self, ids: list[int]) -> list[Any]: ...
    def load_elements_from_constellation(self, constellations: list[str]) -> list[Any]: ...
    def query_satellites(self, ids: list[int]) -> list[Any]: ...
    def load_from_constellation(self, constellation: str) -> list[Any]: ...
    def reset(self, kind: str) -> None: ...
    def process(self, norad_ids: list[int], epoch: datetime) -> None: ...


class SatelliteFactory:
    def __init__(self, repository: SatelliteRepositoryProtocol | None = None):
        """
        Inicializa a factory de satélites.

        Args:
            repository: Repositório de satélites (opcional, pode ser passado em métodos)
        """
        self.repository = repository

    def create_from_norad_ids(
        self,
        ids: list[int],
        epoch: datetime,
    ) -> list[Satellite]:
        """
        Cria satélites baseado em IDs NORAD.

        Args:
            ids: Lista de IDs NORAD
            epoch: Época para selecionar elementos orbitais

        Returns:
            Lista de instâncias Satellite

        Raises:
            ValueError: Se nenhum satélite encontrado ou sem elementos
        """
        if not self.repository:
            msg = "Repositório não foi configurado"
            raise ValueError(msg)

        # Carregar SatIDs do repositório
        if hasattr(self.repository, "load_elements"):
            satids = self.repository.load_elements(ids)
        else:
            satids = self.repository.query_satellites(ids)

        if not satids:
            msg = f"Nenhum SatID encontrado para IDs {ids}"
            raise ValueError(msg)

        # Validar que todos têm elementos orbitais
        for sat in satids:
            if not sat.elements:
                msg = "Sem elementos orbitais disponíveis"
                raise ValueError(msg)

        # Converter época para JD
        epoch_jd = Time(epoch).jd

        # Selecionar TLEs mais próximos da época
        selected = TLESelector.select(satids, epoch_jd)

        # Converter para Satellite objects
        satellites = []
        for sat_dict in selected:
            try:
                sat = Satellite.model_validate(sat_dict)
                satellites.append(sat)
            except Exception:
                # Se não conseguir validar, continuar com o próximo
                pass

        if not satellites:
            msg = "Nenhum satélite válido encontrado"
            raise ValueError(msg)

        return satellites

    def create_from_constellations(
        self,
        constellations: list[str],
        epoch: datetime,
    ) -> list[Satellite]:
        """
        Cria satélites baseado em constelações.

        Args:
            constellations: Lista de nomes de constelações
            epoch: Época para selecionar elementos orbitais

        Returns:
            Lista de instâncias Satellite

        Raises:
            ValueError: Se nenhum satélite encontrado
        """
        if not self.repository:
            msg = "Repositório não foi configurado"
            raise ValueError(msg)

        # Carregar SatIDs do repositório por constelação
        all_satids = []
        for constellation in constellations:
            if hasattr(self.repository, "load_elements_from_constellation"):
                satids = self.repository.load_elements_from_constellation([constellation])
            else:
                satids = self.repository.load_from_constellation(constellation)
            all_satids.extend(satids if isinstance(satids, list) else [satids])

        if not all_satids:
            msg = f"Nenhum SatID encontrado para constelações {constellations}"
            raise ValueError(msg)

        # Converter época para JD
        epoch_jd = Time(epoch).jd

        # Selecionar TLEs mais próximos da época
        selected = TLESelector.select(all_satids, epoch_jd)

        # Converter para Satellite objects
        satellites = []
        for sat_dict in selected:
            try:
                sat = Satellite.model_validate(sat_dict)
                satellites.append(sat)
            except Exception:
                pass

        if not satellites:
            msg = "Nenhum satélite válido encontrado"
            raise ValueError(msg)

        return satellites

    def build(
        self,
        *,
        tr: TimeRange,
        repository: SatelliteRepositoryProtocol,
        ids: list[int] | None = None,
        constellations: list[str] | None = None,
        interval: int = 5,
    ) -> dict[TimeRange, list[Satellite]]:
        """
        Constrói múltiplas instâncias de Satellite para intervalos de tempo.

        Método público (API contrato) para criar satélites com base em:
        - Identificadores NORAD (ids)
        - Nomes de constelações (constellations)
        - Intervalo temporal com slices regulares

        Args:
            tr: TimeRange — Intervalo temporal para simulação
            repository: SatelliteRepository — Fonte de dados de satélites
            ids: IDs NORAD dos satélites (ex: [25544, 33591] para ISS e NOAA)
            constellations: Nomes de constelações (ex: ["iridium", "starlink"])
            interval: Intervalo em dias para slicing (padrão: 5 dias)

        Returns:
            dict[int, list[Satellite]]: Dicionário mapeando índice de slice
            a uma lista de Satellite instâncias processadas para aquele período.

        Raises:
            ValueError: Se nem ids nem constellations fornecidos
            ValueError: Se nenhum satélite encontrado com critérios dados

        Exemplo:
            >>> factory = SatelliteFactory()
            >>> tr = TimeRange(start=Time('2024-01-01'), end=Time('2024-01-10'))
            >>> result = factory.build(
            ...     tr=tr,
            ...     repository=repo,
            ...     ids=[25544],  # ISS
            ...     interval=5,
            ... )
            >>> for timerange, satellites in result.items():
            ...     print(f"Period {timerange}: {len(satellites)} satellites")
        """
        if not ids and not constellations:
            msg = "Deve fornecer ids ou constellations"
            raise ValueError(msg)

        # 1️⃣ carregar SatIDs
        satids = []

        if ids:
            satids.extend(repository.load_elements(ids))

        if constellations:
            satids.extend(repository.load_elements_from_constellation(constellations))

        if not satids:
            msg = "Nenhum SatID encontrado"
            raise ValueError(msg)

        # remover duplicados por norad_id
        _satids = {s.norad_id: s for s in satids}

        # 2️⃣ gerar epochs
        slices = tr.slice(interval)
        epochs = [sl.range.mean().compute().values for sl in slices]

        result: dict[int, list[Satellite]] = {}

        # 3️⃣ processar cada epoch
        for epoch_idx, (_sl, epoch_jd) in enumerate(zip(slices, epochs, strict=False)):
            epoch_dt = Time(epoch_jd, format="jd").to_datetime()

            repository.reset("processed_sats").process(
                norad_ids=list(_satids.keys()),
                epoch=epoch_dt,
            )
            satids = repository.satellites

            selected = TLESelector.select(
                satids=satids,
                epoch_jd=epoch_jd,
            )
            result[epoch_idx] = selected

        logger.info(
            "epoch_processed",
            epoch=epoch_dt.strftime("%Y-%m-%d"),
            num_satellites=len(selected),
        )

        return result
