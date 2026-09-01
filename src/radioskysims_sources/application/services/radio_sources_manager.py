"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/services/radio_sources_manager.py

Descrição:
    Facade de alto nível (UX) para o contexto RadioSources — no mesmo padrão
    do ``SatelliteManager``. Encapsula criação/consulta do catálogo, seleção,
    curvas de fluxo e tracks, com visualização DEFERIDA para
    ``radioskysims-viz``.

    Exemplo (notebook):
        >>> from radioskysims_sources.application.services import RadioSourcesManager
        >>> rs = RadioSourcesManager.get_radio_sources()   # banco → fallback simulado
        >>> rs.view_mollweide()                            # RA/Dec no mapa
        >>> rs.view_flux_density()                         # fluxo × frequência
        >>> rs.select(min_flux_jy=1.0).list_properties()

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from radioskysims_sources.application.services.flux_view import FluxDensity
from radioskysims_sources.domain.radiosources.models import RadioSource, SpectralModel
from radioskysims_sources.domain.radiosources.selection import angular_separation_deg
from radioskysims_sources.domain.radiosources.signals import RadioSourceSignalEvaluator

__all__ = ["RadioSourcesManager"]

# Grade de frequências padrão (banda L do BINGO).
DEFAULT_FREQS_MHZ: np.ndarray = np.linspace(1000.0, 1500.0, 51)

# Endpoint TAP padrão do VizieR (TAP/ADQL).
DEFAULT_TAP_URL: str = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"


class RadioSourcesManager:
    """Gerencia fontes de rádio com uma API única e de alto nível.

    Attributes:
        sources (list[RadioSource]): Catálogo ativo (seleção atual).
    """

    def __init__(self, sources: list[RadioSource]) -> None:
        self.sources = list(sources)

    # ------------------------------------------------------------------
    # Construção / consulta
    # ------------------------------------------------------------------
    @classmethod
    def simulated(
        cls,
        n_sources: int = 40,
        *,
        seed: int | None = 42,
        region: dict[str, float] | None = None,
    ) -> "RadioSourcesManager":
        """Catálogo sintético para demo/desenvolvimento (sem rede nem banco).

        Args:
            n_sources: Número de fontes.
            seed: Semente RNG (``None`` → não determinístico).
            region: Dict com ``ra_min/ra_max/dec_min/dec_max`` (graus).
        """
        if region is None:
            region = {
                "ra_min": 0.0,
                "ra_max": 360.0,
                "dec_min": -60.0,
                "dec_max": 60.0,
            }
        rng = np.random.default_rng(seed)
        ra = rng.uniform(region["ra_min"], region["ra_max"], n_sources)
        dec = rng.uniform(region["dec_min"], region["dec_max"], n_sources)
        s0 = rng.lognormal(mean=np.log(0.3), sigma=1.4, size=n_sources)
        alpha = rng.normal(0.7, 0.4, size=n_sources)
        sources = [
            RadioSource(
                id=f"SIM-{i:04d}",
                ra_deg=float(ra[i]),
                dec_deg=float(dec[i]),
                spectral_model=SpectralModel(
                    nu0_mhz=1400.0, s0_jy=float(s0[i]), alpha=float(alpha[i])
                ),
            )
            for i in range(n_sources)
        ]
        return cls(sources)

    @classmethod
    def from_repository(
        cls,
        repo: Any,
        *,
        ra_deg: float = 180.0,
        dec_deg: float = 0.0,
        radius_deg: float = 180.0,
        min_flux_jy: float | None = None,
    ) -> "RadioSourcesManager":
        """Carrega fontes de um repositório ``RadioSourcesRepositoryPort``."""
        sources = repo.list_sources_in_region(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_deg=radius_deg,
            min_flux_jy=min_flux_jy,
        )
        return cls(sources)

    @classmethod
    def from_db(cls, **kwargs: Any) -> "RadioSourcesManager":
        """Carrega do PostgreSQL (tabela ``radio_sources``).

        Sem banco disponível ou tabela vazia → cai em ``simulated``.
        """
        try:
            from radioskysims_core.db.engine import Database

            from radioskysims_sources.infrastructure.db.radiosources_repository import (
                PostgresRadioSourcesRepository,
            )

            db = Database()
            repo = PostgresRadioSourcesRepository(session_factory=db.session)
            mgr = cls.from_repository(repo, **kwargs)
            if mgr.sources:
                return mgr
        except Exception:  # noqa: BLE001 - fallback seguro para demo
            pass
        return cls.simulated()

    @classmethod
    def get_radio_sources(cls, **kwargs: Any) -> "RadioSourcesManager":
        """Carrega do banco; sem dados/banco → catálogo simulado (demo)."""
        return cls.from_db(**kwargs)

    @classmethod
    def from_tap(
        cls,
        *,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        smin_jy: float = 0.1,
        tap_url: str | None = None,
        repository: Any = None,
        ingest_port: Any = None,
    ) -> "RadioSourcesManager":
        """Ingere um catálogo REAL via TAP/ADQL (VizieR: NVSS + SPECFIND).

        Consulta NVSS na janela (RA, Dec) com corte em fluxo e cruza com
        SPECFIND para estimar o índice espectral, retornando o catálogo de
        :class:`RadioSource` (também persistido se ``repository`` for dado).

        Args:
            ra_min_deg/ra_max_deg/dec_min_deg/dec_max_deg: Janela RA/Dec (graus).
            smin_jy: Fluxo mínimo em 1.4 GHz (Jy).
            tap_url: Endpoint TAP VizieR (padrão: ``DEFAULT_TAP_URL``).
            repository: Repositório para persistir (default: in-memory).
            ingest_port: Adapter de ingestão (default: ``VizierTapIngestAdapter``).
        """
        from radioskysims_sources.application.radiosources_use_cases import (
            BuildNvssSpecfindCatalogInput,
            BuildNvssSpecfindCatalogUseCase,
        )
        from radioskysims_sources.infrastructure.db.radiosources_repository import (
            InMemoryRadioSourcesRepository,
        )
        from radioskysims_sources.infrastructure.external.radiosources_ingest import (
            VizierTapIngestAdapter,
        )

        repo = repository or InMemoryRadioSourcesRepository()
        if ingest_port is None:
            ingest_port = VizierTapIngestAdapter(
                tap_url=tap_url or DEFAULT_TAP_URL
            )
        use_case = BuildNvssSpecfindCatalogUseCase(
            ingest_port=ingest_port, repository=repo
        )
        params = BuildNvssSpecfindCatalogInput(
            ra_min_deg=ra_min_deg,
            ra_max_deg=ra_max_deg,
            dec_min_deg=dec_min_deg,
            dec_max_deg=dec_max_deg,
            smin_jy=smin_jy,
        )
        sources = use_case.execute(params)
        return cls(sources)

    def ingest_nvss_specfind(
        self,
        *,
        ra_min_deg: float,
        ra_max_deg: float,
        dec_min_deg: float,
        dec_max_deg: float,
        smin_jy: float = 0.1,
        **kwargs: Any,
    ) -> "RadioSourcesManager":
        """Ingere NVSS+SPECFIND via TAP real e retorna o novo catálogo."""
        return self.from_tap(
            ra_min_deg=ra_min_deg,
            ra_max_deg=ra_max_deg,
            dec_min_deg=dec_min_deg,
            dec_max_deg=dec_max_deg,
            smin_jy=smin_jy,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Seleção
    # ------------------------------------------------------------------
    def select(
        self,
        *,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        radius_deg: float | None = None,
        min_flux_jy: float | None = None,
        ids: str | list[str] | None = None,
    ) -> "RadioSourcesManager":
        """Retorna novo manager apenas com as fontes que atendem aos critérios."""
        kept = list(self.sources)

        if ids is not None:
            wanted = {ids} if isinstance(ids, str) else set(ids)
            kept = [s for s in kept if s.id in wanted]

        if (
            ra_deg is not None
            and dec_deg is not None
            and radius_deg is not None
        ):
            sep = angular_separation_deg(
                ra_deg,
                dec_deg,
                np.array([s.ra_deg for s in kept], dtype=float),
                np.array([s.dec_deg for s in kept], dtype=float),
            )
            kept = [s for s, d in zip(kept, sep, strict=False) if d <= radius_deg]

        if min_flux_jy is not None:
            kept = [s for s in kept if s.spectral_model.s0_jy >= min_flux_jy]

        return RadioSourcesManager(kept)

    # ------------------------------------------------------------------
    # Dados / fluxo
    # ------------------------------------------------------------------
    def list_properties(self, freq_mhz: float | None = 1400.0) -> pd.DataFrame:
        """Tabela resumo (id, RA/Dec, modelo espectral e fluxo em ``freq_mhz``)."""
        rows: list[dict[str, Any]] = []
        for s in self.sources:
            flux = (
                float(
                    RadioSourceSignalEvaluator.evaluate_flux(s, [freq_mhz])[0]
                )
                if freq_mhz is not None
                else s.spectral_model.s0_jy
            )
            col = f"flux_{freq_mhz:.0f}mhz_jy" if freq_mhz is not None else "flux_jy"
            rows.append(
                {
                    "id": s.id,
                    "ra_deg": s.ra_deg,
                    "dec_deg": s.dec_deg,
                    "nu0_mhz": s.spectral_model.nu0_mhz,
                    "s0_jy": s.spectral_model.s0_jy,
                    "alpha": s.spectral_model.alpha,
                    col: flux,
                }
            )
        return pd.DataFrame(rows)

    def flux_density(self, freqs_mhz: Any = None) -> FluxDensity:
        """Curvas de fluxo (Jy) × frequência (MHz) — uma por fonte.

        ``FluxDensity.view()`` renderiza o gráfico (lazy para o viz).
        """
        freqs = np.asarray(
            DEFAULT_FREQS_MHZ if freqs_mhz is None else freqs_mhz, dtype=float
        )
        curves = RadioSourceSignalEvaluator.evaluate_flux_grid(self.sources, freqs)
        return FluxDensity(freqs, curves)

    def build_tracks(self, observer: Any, time_range: Any):
        """Calcula tracks das fontes no referencial do observador (skyfield)."""
        from radioskysims_sources.infrastructure.external.radiosource_track_provider import (
            SkyfieldRadioSourceTrackProvider,
        )

        return SkyfieldRadioSourceTrackProvider().compute_track(
            self.sources, observer, time_range
        )

    # ------------------------------------------------------------------
    # Visualização (deferida ao radioskysims-viz)
    # ------------------------------------------------------------------
    def view_mollweide(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """Mapa Mollweide RA/Dec das fontes (tamanho/cor por fluxo S0)."""
        from radioskysims_viz.sources import radio_sources_mollweide

        ra = np.array([s.ra_deg for s in self.sources], dtype=float)
        dec = np.array([s.dec_deg for s in self.sources], dtype=float)
        flux = np.array([s.spectral_model.s0_jy for s in self.sources], dtype=float)
        return radio_sources_mollweide(
            ra, dec, fluxes_jy=flux, title=title, **kwargs
        )

    def view_flux_density(self, freqs_mhz: Any = None, *, title: str | None = None, **kwargs: Any) -> Any:
        """Curvas de fluxo × frequência (uma linha por fonte + soma)."""
        return self.flux_density(freqs_mhz).view(title=title, **kwargs)

    # ------------------------------------------------------------------
    # Introspect
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Resumo textual do catálogo (tabela formatada)."""
        return (
            f"RadioSourcesManager — {len(self.sources)} fontes\n"
            + self.list_properties().to_string(index=False)
        )

    def __len__(self) -> int:
        return len(self.sources)

    def __repr__(self) -> str:
        return f"<RadioSourcesManager: {len(self.sources)} fontes>"
