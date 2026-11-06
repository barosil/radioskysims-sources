"""Builders de componentes do pacote (composition root — strategy).
Extraído de ``radioskysims.bootstrap.builders`` na Fase 5.
Copyright BINGO Collaboration
"""

from __future__ import annotations

import importlib
from typing import Any

from radioskysims_core.config.config_loader import load_config
from radioskysims_core.db.engine import Database

from radioskysims_sources.application.cosmology.generate_hi_map import (
    GenerateNeutralHydrogenMapUseCase,
)
from radioskysims_sources.application.cosmology.generate_hi_maps import (
    GenerateNeutralHydrogenMapsUseCase,
)
from radioskysims_sources.application.ports.radiosources import (
    CatalogIngestPort,
    RadioSourcesCatalogPort,
    RadioSourcesFluxPort,
    RadioSourcesRepositoryPort,
)
from radioskysims_sources.application.radiosources_use_cases import (
    BuildNvssSpecfindCatalogUseCase,
    BuildRadioSourcesCatalogUseCase,
    ComputeRadioSourcesForegroundUseCase,
)
from radioskysims_sources.application.services.local_sources_service import LocalSourcesService
from radioskysims_sources.domain.cosmology import CosmologyParameters
from radioskysims_sources.infrastructure.db.satellite.satelliterepository import SatelliteRepository
from radioskysims_sources.infrastructure.db.signal.repository import SignalRepository
from radioskysims_sources.infrastructure.db.solar_flux_repository import SolarFluxRepository
from radioskysims_sources.infrastructure.external.local_source_track_skyfield_provider import (
    SkyfieldLocalSourceTrackProvider,
)
from radioskysims_sources.infrastructure.external.solar_flux_downloader import SolarFluxDownloader


def build_repositories(
    db: Database,
) -> dict[str, object]:
    """Constroi repositórios concretos."""

    sat_repo = SatelliteRepository(db)

    sig_repo = SignalRepository(db)

    return {
        "satellite_repository": sat_repo,
        "signal_repository": sig_repo,
    }


def build_solar_flux_downloader(source: str = "srd"):
    """Seleciona o adapter de fluxo solar externo.

    Parameters
    ----------
    source : str
        ``"srd"`` (padrão) → SRD/SRS do WDC (BOM, 245–15400 MHz);
        ``"nro"`` → NoRP/NRO (XDR/IDL SAVE, absolute measurements).
    """
    kind = (source or "srd").strip().lower()
    if kind in {"srd", "srs"}:
        return SolarFluxDownloader()
    if kind in {"nro", "norp"}:
        from radioskysims_sources.infrastructure.external.nro_solar_flux_downloader import (
            NroSolarFluxDownloader,
        )

        return NroSolarFluxDownloader()
    msg = f"fonte de fluxo solar desconhecida: {source!r} (use 'srd' ou 'nro')"
    raise ValueError(msg)


def build_local_sources_service(
    db: Database,
    *,
    solar_flux_source: str = "srd",
) -> LocalSourcesService:
    """Constroi servico de fontes locais com adapters concretos.

    Parameters
    ----------
    solar_flux_source : str
        Fonte do fluxo solar externo (``"srd"`` padrão ou ``"nro"``).
    """

    return LocalSourcesService(
        solar_flux_downloader=build_solar_flux_downloader(solar_flux_source),
        solar_flux_repository=(SolarFluxRepository(engine=db)),
        local_source_track_provider=(SkyfieldLocalSourceTrackProvider()),
    )


def build_radiosources_use_cases(
    *,
    ingest_port: CatalogIngestPort,
    repository_port: RadioSourcesRepositoryPort,
    catalog_port: RadioSourcesCatalogPort,
    flux_port: RadioSourcesFluxPort,
) -> dict[str, object]:
    """Caminho oficial de criacao dos casos de uso de RadioSources."""

    return {
        "build_catalog_use_case": (
            BuildRadioSourcesCatalogUseCase(
                ingest_port=ingest_port,
                repository=repository_port,
            )
        ),
        "build_nvss_specfind_catalog_use_case": (
            BuildNvssSpecfindCatalogUseCase(
                ingest_port=ingest_port,
                repository=repository_port,
            )
        ),
        "compute_foreground_use_case": (
            ComputeRadioSourcesForegroundUseCase(
                catalog_port=catalog_port,
                flux_port=flux_port,
            )
        ),
    }


def build_track_factory():
    """Constrói o ``TrackFactory`` (adapter skyfield) no composition root."""
    from radioskysims_core.infrastructure.tracks import TrackFactory

    return TrackFactory()


def build_continuum_source_orchestrator(
    *,
    unit: str = "uK_CMB",
    field: str = "I",
) -> Any:
    """Monta o ``ContinuumSourceOrchestrator`` com gerador PySM3 + repo HDF5.

    Mantém a construção dos adapters concretos de skymap na camada
    de composition root (bootstrap) — a application não importa infrastructure.
    """
    from radioskysims_core.paths import get_project_root
    from radioskysims_skymap.infrastructure.skymap.providers.pysm import (
        PySMSkyMapGeneratorAdapter,
    )
    from radioskysims_skymap.infrastructure.skymap.repository.hdf5 import (
        HDF5SkyMapRepository,
    )

    from radioskysims_sources.application.orchestrators.continuum_source_orchestrator import (
        ContinuumSourceOrchestrator,
    )

    repository_root = get_project_root() / "data" / "aux" / "skymaps"
    return ContinuumSourceOrchestrator.from_generator(
        PySMSkyMapGeneratorAdapter(unit=unit, field=field),
        repository=HDF5SkyMapRepository(root=repository_root),
    )


def build_cosmology_components(
    *,
    skymap_repository: Any,
) -> dict[str, object]:
    """
    Constroi os componentes concretos do pipeline cosmologico de HI.

    Wiring:

        CosmologyParameters
                |
                v
        PyCCLCosmologyAdapter
                |
                v
        GenerateNeutralHydrogenMapUseCase
                ^
                |
        HealpyGaussianMatterFieldAdapter
                |
                v
        GenerateNeutralHydrogenMapsUseCase

    PyCCL e Healpy permanecem confinados na camada Infrastructure.
    """

    _gaussian_mod = importlib.import_module(
        "radioskysims_sources.infrastructure.cosmology.healpy_gaussian"
    )
    HealpyGaussianMatterFieldAdapter = _gaussian_mod.HealpyGaussianMatterFieldAdapter

    _lognormal_mod = importlib.import_module(
        "radioskysims_sources.infrastructure.cosmology.healpy_lognormal"
    )
    HealpyLognormalMatterFieldAdapter = _lognormal_mod.HealpyLognormalMatterFieldAdapter

    _pyccl_mod = importlib.import_module("radioskysims_sources.infrastructure.cosmology.pyccl")
    PyCCLCosmologyAdapter = _pyccl_mod.PyCCLCosmologyAdapter

    cfg = load_config("cosmology") or {}

    required_parameters = (
        "omega_c",
        "omega_b",
        "h",
        "sigma8",
        "n_s",
    )

    missing = [parameter for parameter in required_parameters if parameter not in cfg]

    if missing:
        raise ValueError(
            "settings.toml deve definir os parâmetros cosmológicos ausentes: " + ", ".join(missing)
        )

    cosmology_parameters = CosmologyParameters(
        omega_c=float(cfg["omega_c"]),
        omega_b=float(cfg["omega_b"]),
        h=float(cfg["h"]),
        sigma8=float(cfg["sigma8"]),
        n_s=float(cfg["n_s"]),
    )

    angular_power = PyCCLCosmologyAdapter()

    field_model = (
        str(
            cfg.get(
                "field_model",
                "gaussian",
            )
        )
        .strip()
        .lower()
    )

    if field_model == "gaussian":
        matter_field = HealpyGaussianMatterFieldAdapter()

    elif field_model == "lognormal":
        matter_field = HealpyLognormalMatterFieldAdapter()

    else:
        msg = (
            "Unsupported cosmology field_model: "
            f"{field_model!r}. "
            "Expected 'gaussian' or 'lognormal'."
        )
        raise ValueError(msg)

    channel_generator = GenerateNeutralHydrogenMapUseCase(
        angular_power=angular_power,
        gaussian_field=matter_field,
    )

    neutral_hydrogen_map_generator = GenerateNeutralHydrogenMapsUseCase(
        channel_generator=channel_generator,
        repository=skymap_repository,
    )

    return {
        "cosmology_parameters": cosmology_parameters,
        "neutral_hydrogen_map_generator": neutral_hydrogen_map_generator,
    }


def configure_facade_defaults() -> None:
    """Registra as fábricas concretas dos facades UX (TD-001).

    O composition root concentra a única dependência de infrastructure dos
    facades de conveniência (``Sun``, ``RadioSourcesManager``,
    ``SatelliteOrchestrator``), quebrando o ciclo application → infrastructure.
    """
    from radioskysims_sources.application.orchestrators.satellite_orchestrator import (
        set_default_psd_provider_factory,
    )
    from radioskysims_sources.application.services.local_sources_manager import (
        set_default_track_provider_factory,
    )
    from radioskysims_sources.application.services.radio_sources_manager import (
        set_default_factories,
    )
    from radioskysims_sources.infrastructure.db.radiosources_repository import (
        InMemoryRadioSourcesRepository,
        PostgresRadioSourcesRepository,
    )
    from radioskysims_sources.infrastructure.external.radiosource_track_provider import (
        SkyfieldRadioSourceTrackProvider,
    )
    from radioskysims_sources.infrastructure.external.radiosources_ingest import (
        VizierTapIngestAdapter,
    )
    from radioskysims_sources.infrastructure.signal.gnsstools_psd import (
        GnsstoolsAnalyticPsdProvider,
    )

    set_default_track_provider_factory(SkyfieldLocalSourceTrackProvider)
    set_default_psd_provider_factory(GnsstoolsAnalyticPsdProvider)
    set_default_factories(
        db_repository=lambda: PostgresRadioSourcesRepository(session_factory=Database().session),
        memory_repository=InMemoryRadioSourcesRepository,
        ingest_adapter=VizierTapIngestAdapter,
        track_provider=SkyfieldRadioSourceTrackProvider,
    )
