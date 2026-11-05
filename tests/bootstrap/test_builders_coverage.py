from __future__ import annotations

from types import SimpleNamespace

import pytest

import radioskysims_sources.bootstrap.builders as builders


def test_build_repositories_wires_satellite_and_signal_repositories(monkeypatch):
    classes = []

    class Satellite:
        def __init__(self, db):
            classes.append(("satellite", db))

    class Signal:
        def __init__(self, db):
            classes.append(("signal", db))

    monkeypatch.setattr(builders, "SatelliteRepository", Satellite)
    monkeypatch.setattr(builders, "SignalRepository", Signal)
    database = object()

    result = builders.build_repositories(database)

    assert result.keys() == {"satellite_repository", "signal_repository"}
    assert classes == [("satellite", database), ("signal", database)]


def test_build_local_sources_service_wires_adapters(monkeypatch):
    created = []

    class Downloader:
        pass

    class Repository:
        def __init__(self, **kwargs):
            created.append(("repository", kwargs))

    class TrackProvider:
        pass

    monkeypatch.setattr(builders, "SolarFluxDownloader", Downloader)
    monkeypatch.setattr(builders, "SolarFluxRepository", Repository)
    monkeypatch.setattr(builders, "SkyfieldLocalSourceTrackProvider", TrackProvider)
    database = object()

    result = builders.build_local_sources_service(database)

    assert result._track_provider.__class__ is TrackProvider
    assert created == [("repository", {"engine": database})]


def test_build_radiosources_use_cases_returns_all_use_cases():
    ports = {
        "ingest_port": object(),
        "repository_port": object(),
        "catalog_port": object(),
        "flux_port": object(),
    }

    result = builders.build_radiosources_use_cases(**ports)

    assert set(result) == {
        "build_catalog_use_case",
        "build_nvss_specfind_catalog_use_case",
        "compute_foreground_use_case",
    }


def test_build_cosmology_components_rejects_missing_configuration(monkeypatch):
    monkeypatch.setattr(builders, "load_config", lambda _section: {})

    with pytest.raises(ValueError, match="parâmetros cosmológicos ausentes"):
        builders.build_cosmology_components(skymap_repository=object())


def test_build_cosmology_components_supports_gaussian_and_rejects_unknown(monkeypatch):
    class Adapter:
        pass

    class Generator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    modules = {
        "radioskysims_sources.infrastructure.cosmology.healpy_gaussian": SimpleNamespace(
            HealpyGaussianMatterFieldAdapter=Adapter
        ),
        "radioskysims_sources.infrastructure.cosmology.healpy_lognormal": SimpleNamespace(
            HealpyLognormalMatterFieldAdapter=Adapter
        ),
        "radioskysims_sources.infrastructure.cosmology.pyccl": SimpleNamespace(
            PyCCLCosmologyAdapter=Adapter
        ),
    }
    monkeypatch.setattr(builders.importlib, "import_module", modules.__getitem__)
    config = {"omega_c": 0.25, "omega_b": 0.05, "h": 0.67, "sigma8": 0.8, "n_s": 0.96}
    monkeypatch.setattr(builders, "load_config", lambda _section: config)
    monkeypatch.setattr(builders, "GenerateNeutralHydrogenMapUseCase", Generator)
    monkeypatch.setattr(builders, "GenerateNeutralHydrogenMapsUseCase", Generator)

    result = builders.build_cosmology_components(skymap_repository="repo")

    assert result["cosmology_parameters"].h == 0.67
    assert isinstance(result["neutral_hydrogen_map_generator"], Generator)

    config["field_model"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported cosmology field_model"):
        builders.build_cosmology_components(skymap_repository="repo")
