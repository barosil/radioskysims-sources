# application/services/satellite_manager.py
"""\
Facade de alto nível (UX) para o ciclo de vida de satélites.

Encapsula criação/consulta, tracks, signals e o montado final
(``SatelliteAssembled``) usando as factories e repositórios do pacote —
sem expor os detalhes de montagem ao usuário.

Exemplo de uso (notebook):
    >>> from datetime import datetime
    >>> mgr = SatelliteManager(repository=repo, signal_repository=sig_repo,
    ...                        epoch=datetime(2026, 8, 30, 12, 0))
    >>> mgr.create_from_constellations(["GPS"])
    >>> assembled = mgr.build(tr, fr)
    >>> assembled.plot_tracks(smooth=True)
    >>> assembled.select_Constellation("GPS").plot_signals()

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from radioskysims_core.domain.common import FrequencyRange, TimeRange
from radioskysims_core.domain.tracks.factory import TrackFactory
from radioskysims_sources.domain.satellite.assembler import SatelliteAssembler
from radioskysims_sources.domain.satellite.factory import SatelliteFactory
from radioskysims_sources.domain.satellite.models import Satellite, SatelliteAssembled, as_utc
from radioskysims_sources.domain.signal.factory import SignalFactory


class SatelliteManager:
    """Gerencia satélites com uma API única e de alto nível.

    Attributes:
        factory (SatelliteFactory): Fábrica de satélites (criação/consulta).
        signal_repository: Repositório de sinais (obrigatório para signals).
        epoch (datetime | None): Época UTC de referência (opcional).
        satellites (list[Satellite]): Última seleção criada/consultada.
    """

    def __init__(
        self,
        factory: SatelliteFactory | None = None,
        *,
        repository: Any = None,
        signal_repository: Any = None,
        epoch: datetime | None = None,
    ) -> None:
        if factory is None:
            factory = SatelliteFactory(repository)
        self.factory = factory
        self.signal_repository = signal_repository
        self.epoch: datetime | None = as_utc(epoch) if epoch is not None else None
        self.satellites: list[Satellite] = []

    # ------------------------------------------------------------------
    # Época
    # ------------------------------------------------------------------
    def _epoch(self, epoch: datetime | None) -> datetime:
        """Resolve a época efetiva (arg > self.epoch > 'agora')."""
        if epoch is not None:
            return as_utc(epoch)
        if self.epoch is not None:
            return self.epoch
        return datetime.now(timezone.utc)

    @staticmethod
    def _dedupe(
        satellites: list[Satellite], epoch: datetime
    ) -> list[Satellite]:
        """Deduplica por NORAD ID, mantendo o elemento mais próximo da época.

        O catálogo guarda várias épocas por satélite; para a experiência de
        uso, cada satélite aparece uma única vez com o TLE mais próximo do
        ``epoch`` solicitado.
        """
        best: dict[int, Satellite] = {}
        for sat in satellites:
            current = best.get(sat.norad_id)
            if current is None or abs((sat.epoch - epoch).total_seconds()) < abs(
                (current.epoch - epoch).total_seconds()
            ):
                best[sat.norad_id] = sat
        return list(best.values())

    # ------------------------------------------------------------------
    # Criação / consulta (TODO: create / query / TLE)
    # ------------------------------------------------------------------
    def create_from_constellations(
        self,
        constellations: list[str],
        *,
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        """Cria satélites das constelações dadas (ex.: ``["GPS"]``)."""
        eff = self._epoch(epoch)
        sats = self.factory.create_from_constellations(constellations, eff)
        self.satellites = self._dedupe(sats, eff)
        return self.satellites

    def create_from_norad_ids(
        self,
        norad_ids: list[int],
        *,
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        """Cria satélites por NORAD ID."""
        eff = self._epoch(epoch)
        sats = self.factory.create_from_norad_ids(norad_ids, eff)
        self.satellites = self._dedupe(sats, eff)
        return self.satellites

    def query_sats(
        self,
        norad_ids: list[int] | None = None,
        *,
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        """Consulta o repositório (elementos orbitais mais próximos da época)."""
        if self.factory.repository is None:
            msg = "Repositório não configurado no SatelliteManager."
            raise ValueError(msg)
        eff = self._epoch(epoch)
        sats = self.factory.repository.query_sats(norad_ids=norad_ids, epoch=eff)
        self.satellites = self._dedupe(sats, eff)
        return self.satellites

    def get_tle(self) -> list[dict[str, Any]]:
        """Linhas TLE (2LE) dos satélites selecionados (TODO: retrieve TLE)."""
        return [
            {
                "norad_id": s.norad_id,
                "name": s.name,
                "constellation": s.constellation,
                "epoch": s.epoch.isoformat(),
                "tle_line1": s.element.tle_line1,
                "tle_line2": s.element.tle_line2,
            }
            for s in self.satellites
        ]

    # ------------------------------------------------------------------
    # Tracks / Signals / Assembled (TODO: build_*)
    # ------------------------------------------------------------------
    def _block(self, timerange: TimeRange, sats: list[Satellite] | None) -> dict:
        sats = sats or self.satellites
        if not sats:
            msg = (
                "Nenhum satélite selecionado — use create_from_constellations, "
                "create_from_norad_ids ou query_sats antes."
            )
            raise ValueError(msg)
        return {float(timerange.start_jd): {"timerange": timerange, "satellites": sats}}

    def _assembler(self, frequency_range: FrequencyRange) -> SatelliteAssembler:
        if self.signal_repository is None:
            msg = "signal_repository não configurado no SatelliteManager."
            raise ValueError(msg)
        signal_factory = SignalFactory(
            frequency_range=frequency_range, signal_repo=self.signal_repository
        )
        return SatelliteAssembler(
            signal_factory=signal_factory, track_factory=TrackFactory()
        )

    def build_tracks(
        self,
        timerange: TimeRange,
        *,
        sats: list[Satellite] | None = None,
    ):
        """Constrói o dataset de tracks (posições ECI em km) para o intervalo."""
        block = self._block(timerange, sats)
        return TrackFactory().build(block).dataset

    def build_signals(
        self,
        timerange: TimeRange,
        frequency_range: FrequencyRange,
        *,
        sats: list[Satellite] | None = None,
    ):
        """Constrói o dataset de sinais (PSD por satélite/sinal)."""
        block = self._block(timerange, sats)
        return self._assembler(frequency_range).build_signals(block)

    def build(
        self,
        timerange: TimeRange,
        frequency_range: FrequencyRange,
        *,
        sats: list[Satellite] | None = None,
    ) -> SatelliteAssembled:
        """Monta o ``SatelliteAssembled`` completo (tracks + signals)."""
        block = self._block(timerange, sats)
        return self._assembler(frequency_range).build_signals(block)
