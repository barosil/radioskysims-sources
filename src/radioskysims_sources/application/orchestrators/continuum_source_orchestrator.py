"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/orchestrators/continuum_source_orchestrator.py

Descrição:
    Fachada de alto nível (UX — Onda 3/R6) para fontes de contínuo (Galáxia).

    Encapsula o ``GenerateGalaxyMapsUseCase`` (mapas difusos da Galáxia por
    bin de frequência). O usuário não precisa conhecer o use case nem os
    adapters (PySM3) — recebe o orquestrador com o use case montado.

    Uso (notebook):
        >>> from radioskysims_sources import ContinuumSourceOrchestrator
        >>> orch = ContinuumSourceOrchestrator(use_case=galaxy_uc)
        >>> result = orch.galaxy_maps(source=galaxy, frequency_range=fr, nside=64)

Copyright BINGO Collaboration
Last modified: 2026-09-03
"""

from __future__ import annotations

from typing import Any

from radioskysims_core.domain.common import FrequencyRange

from radioskysims_sources.application.skymap.generate_continuum import (
    GalaxyMapSet,
    GenerateGalaxyMapsUseCase,
)
from radioskysims_sources.domain.radiosources.continuum import Galaxy

__all__ = ["ContinuumSourceOrchestrator"]


class ContinuumSourceOrchestrator:
    """Orquestrador/fachada pública de fontes de contínuo (mapas da Galáxia)."""

    def __init__(self, use_case: GenerateGalaxyMapsUseCase) -> None:
        self._use_case = use_case

    # ──────────────────────────────────────────────────────────
    # Fábricas
    # ──────────────────────────────────────────────────────────
    @classmethod
    def from_use_case(
        cls,
        use_case: GenerateGalaxyMapsUseCase,
    ) -> "ContinuumSourceOrchestrator":
        """Embrulha um ``GenerateGalaxyMapsUseCase`` já montado (composition)."""
        return cls(use_case)

    @classmethod
    def from_generator(
        cls,
        generator: Any,
        repository: Any | None = None,
    ) -> "ContinuumSourceOrchestrator":
        """Monta o use case a partir de um ``SkyMapGeneratorPort`` (ex.: PySM3)."""
        return cls(GenerateGalaxyMapsUseCase(generator=generator, repository=repository))

    @classmethod
    def default(
        cls,
        *,
        unit: str = "uK_CMB",
        field: str = "I",
    ) -> "ContinuumSourceOrchestrator":
        """Monta com o gerador PySM3 padrão (sem persistência).

        Requer o extra de PySM3 (``pysm3``) instalado; a persistência
        (``persist=True``) exige passar um repositório via ``from_generator``.
        """
        try:
            from radioskysims_skymap.infrastructure.skymap.providers.pysm import (
                PySMSkyMapGeneratorAdapter,
            )
        except ImportError as exc:  # pragma: no cover - depende de extra pysm
            msg = (
                "default() requer o gerador PySM3 (extra 'pysm'); use "
                "from_generator(...) com um SkyMapGeneratorPort próprio."
            )
            raise RuntimeError(msg) from exc
        return cls.from_generator(PySMSkyMapGeneratorAdapter(unit=unit, field=field))

    # ──────────────────────────────────────────────────────────
    # Fluxo
    # ──────────────────────────────────────────────────────────
    def galaxy_maps(
        self,
        *,
        source: Galaxy,
        frequency_range: FrequencyRange,
        nside: int,
        persist: bool = False,
        overwrite: bool = False,
    ) -> GalaxyMapSet:
        """Gera os mapas HEALPix da Galáxia para cada bin de frequência.

        Args:
            source: Fonte de contínuo (``Galaxy`` do domínio).
            frequency_range: Banda em MHz (bins do mapa).
            nside: Resolução HEALPix.
            persist: Se ``True``, persiste cada mapa via repositório.
            overwrite: Sobrescreve mapas já persistidos.
        """
        return self._use_case.execute(
            source=source,
            frequency_range=frequency_range,
            nside=nside,
            persist=persist,
            overwrite=overwrite,
        )

    def __repr__(self) -> str:
        return f"<ContinuumSourceOrchestrator use_case={type(self._use_case).__name__}>"
