# radioskysims-sources

Fontes de sinal do ecossistema **RadioSkySim**.

## Conteúdo

- `domain.satellite` — satélites GNSS (TLE/RINEX/sinais)
- `domain.localsources` — fontes locais (Sol: fluxo/track)
- `domain.radiosources` — fontes de rádio (catálogo, tracks, continuum)
- `domain.cosmology` — cosmologia (frequência, brightness temperature)
- `domain.catalog` — `CatalogIngestion`, `CatalogQuery`
- `domain.signal` + `domain.signal_source(.py)`/`signal_source_group.py` —
  protocolo unificado `SignalSource` (fronteira sources↔simulation)
- `domain.source` — modelos de fonte
- `application` — use cases, serviços (satellite/catalog/local_sources/source_assembler),
  `generate_continuum`, cosmologia
- `infrastructure` — repositórios DB (satellite/signal/radiosources/solar_flux),
  adapters externos (SpaceTrack, Solar Flux, catálogos), cosmologia (pyccl/healpy)
- `interfaces` — `satellite_panel`, `satellite_controller`, `satellite_cli`

Depende de `radioskysims-core` + `radioskysims-skymap`.

## Testes

```bash
uv run --package radioskysims-sources pytest
```
