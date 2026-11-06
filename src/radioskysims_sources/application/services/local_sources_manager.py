"""\
Módulo: packages/radioskysims-sources/src/radioskysims_sources/application/services/local_sources_manager.py

Descrição:
    Facade de alto nível (UX) para o contexto LocalSources — no mesmo padrão
    do ``SatelliteManager``. Expõe a classe ``Sun`` (fonte local Sol) com
    track (ALT/AZ + RA/Dec) e fluxo solar, com visualização DEFERIDA para
    ``radioskysims-viz``.

    Exemplo (notebook):
        >>> from radioskysims_sources.application.services import Sun
        >>> sun = Sun.simulated()
        >>> sun.view_skychart()        # carta local ALT/AZ
        >>> sun.view_mollweide()       # RA/Dec no mapa Mollweide
        >>> sun.flux_density().view()  # fluxo × frequência

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from radioskysims_core.domain.common import Observer, TimeRange
from radioskysims_sources.application.ports.localsources import LocalSourceTrackProviderPort
from radioskysims_sources.application.services.flux_view import FluxDensity
from radioskysims_sources.domain.localsources.models import LocalSourceTrack, SolarFlux

__all__ = ["LocalSourcesManager", "Sun"]

# Factory do track provider injetada pelo composition root — quebra o ciclo
# ``application → infrastructure`` (TD-001): a application depende só da porta.
_DEFAULT_TRACK_PROVIDER_FACTORY: Callable[[], LocalSourceTrackProviderPort] | None = None


def set_default_track_provider_factory(
    factory: Callable[[], LocalSourceTrackProviderPort] | None,
) -> None:
    """Registra a factory do track provider de fontes locais (bootstrap)."""
    global _DEFAULT_TRACK_PROVIDER_FACTORY
    _DEFAULT_TRACK_PROVIDER_FACTORY = factory


def _track_provider() -> LocalSourceTrackProviderPort:
    """Retorna o track provider injetado ou falha com erro claro."""
    if _DEFAULT_TRACK_PROVIDER_FACTORY is None:
        msg = (
            "Track provider de fontes locais não configurado: chame "
            "set_default_track_provider_factory() no composition root."
        )
        raise RuntimeError(msg)
    return _DEFAULT_TRACK_PROVIDER_FACTORY()

# Coordenadas de referência do BINGO (Parque de Alcanos / Urânia, RN).
DEFAULT_LAT_DEG: float = -6.2391
DEFAULT_LON_DEG: float = -38.9296
DEFAULT_HEIGHT_M: float = 440.0
# Janela padrão (UTC) para o track/skychart do Sol.
DEFAULT_WINDOW_ISO: tuple[str, str] = ("2026-08-30T10:00:00", "2026-08-30T22:00:00")
DEFAULT_WINDOW_RESOLUTION_SEC: float = 600.0

# Fluxo solar padrão (SFU @ 2800 MHz, F10.7) — Sol calmo. 1 SFU = 1e4 Jy.
DEFAULT_SOLAR_FLUX_SFU: float = 150.0
SFU_TO_JY: float = 1.0e4

# Grade de frequências padrão (banda L do BINGO).
DEFAULT_FREQS_MHZ: np.ndarray = np.linspace(1000.0, 1500.0, 51)


def _default_observer(time_range: TimeRange) -> Observer:
    """Observador padrão (BINGO) para a janela temporal dada."""
    return Observer(
        lat=DEFAULT_LAT_DEG,
        lon=DEFAULT_LON_DEG,
        height=DEFAULT_HEIGHT_M,
        timeframe=time_range,
    )


def _default_time_range() -> TimeRange:
    return TimeRange.from_iso(
        DEFAULT_WINDOW_ISO[0],
        DEFAULT_WINDOW_ISO[1],
        DEFAULT_WINDOW_RESOLUTION_SEC,
    )


def _quiet_sun_flux(time_range: TimeRange, freqs_mhz: np.ndarray) -> SolarFlux:
    """SolarFlux constante (Sol calmo) cobrindo o grid completo da janela."""
    times = time_range.to_datetime64().astype("datetime64[ns]")
    freqs = np.asarray(freqs_mhz, dtype=float)
    # Eixo de frequência com >= 2 pontos: interp do xarray degenera com 1 ponto.
    if freqs.size == 1:
        f0 = float(freqs[0])
        freqs_axis = np.array([f0 - 1.0, f0, f0 + 1.0])
    else:
        freqs_axis = freqs
    flux_jy = float(DEFAULT_SOLAR_FLUX_SFU) * SFU_TO_JY
    ds = xr.Dataset(
        {
            "flux_jy": (
                ("time", "frequency_mhz"),
                np.full((times.size, freqs_axis.size), flux_jy),
            )
        },
        coords={"time": times, "frequency_mhz": freqs_axis},
    )
    return SolarFlux(
        start_utc=pd.Timestamp(np.datetime64(times[0], "s")).to_pydatetime(),
        end_utc=pd.Timestamp(np.datetime64(times[-1], "s")).to_pydatetime(),
        dataset=ds,
        source_url="default:quiet-sun",
    )


class LocalSourcesManager:
    """Facade de alto nível do contexto LocalSources (fontes locais: Sol)."""

    def __init__(
        self,
        *,
        observer: Observer | None = None,
        time_range: TimeRange | None = None,
    ) -> None:
        self.observer = observer
        self.time_range = time_range or _default_time_range()
        if self.observer is None:
            self.observer = _default_observer(self.time_range)

    def sun(
        self,
        *,
        simulated: bool = True,
        freqs_mhz: np.ndarray | None = None,
    ) -> "Sun":
        """Retorna a fonte local Sol (simulada por padrão)."""
        if simulated:
            return Sun.simulated(
                observer=self.observer,
                time_range=self.time_range,
                freqs_mhz=freqs_mhz,
            )
        msg = "Somente a fonte simulada está disponível no MVP atual."
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        return (
            f"<LocalSourcesManager: obs=({self.observer.lat.to_value('deg'):.2f}°, "
            f"{self.observer.lon.to_value('deg'):.2f}°)>"
        )


class Sun:
    """Fonte local Sol: track (ALT/AZ + RA/Dec) e fluxo solar.

    Exemplo (notebook):
        >>> sun = Sun.simulated()
        >>> sun.view_skychart()
        >>> sun.view_mollweide()
        >>> sun.flux_density().view()
    """

    def __init__(
        self,
        *,
        observer: Observer,
        time_range: TimeRange,
        flux: SolarFlux | None = None,
        track: LocalSourceTrack | None = None,
    ) -> None:
        self.observer = observer
        self.time_range = time_range
        self.flux = flux or _quiet_sun_flux(time_range, DEFAULT_FREQS_MHZ)
        self.track = track
        if self.track is None:
            self._compute_track()

    @classmethod
    def simulated(
        cls,
        simulation_params: Any = None,  # noqa: ARG003 - compatível com esboço do notebook
        *,
        observer: Observer | None = None,
        time_range: TimeRange | None = None,
        freqs_mhz: np.ndarray | list[float] | None = None,
        flux: SolarFlux | None = None,
    ) -> "Sun":
        """Sol com track via skyfield e fluxo (real ou Sol calmo padrão).

        Args:
            flux: ``SolarFlux`` real (ex.: download SRD/SRS). ``None`` →
                Sol calmo padrão (~150 SFU).
        """
        tr = time_range or _default_time_range()
        obs = observer or _default_observer(tr)
        if flux is not None:
            return cls(observer=obs, time_range=tr, flux=flux)
        freqs = np.asarray(DEFAULT_FREQS_MHZ if freqs_mhz is None else freqs_mhz, dtype=float)
        return cls(observer=obs, time_range=tr, flux=_quiet_sun_flux(tr, freqs))

    # ------------------------------------------------------------------
    # Track
    # ------------------------------------------------------------------
    def _compute_track(self) -> None:
        """Calcula o track (ALT/AZ/RA/Dec) via provider injetado (bootstrap)."""
        self.track = _track_provider().compute_track(
            "Sun", self.observer, self.time_range
        )

    def position_at(self, t_utc: Any) -> dict[str, float]:
        """Métricas geométricas do Sol em um instante UTC.

        Aceita ``datetime`` (aware ou naive) ou ``numpy.datetime64``.
        """
        from datetime import UTC, datetime

        if isinstance(t_utc, np.datetime64):
            t_utc = pd.Timestamp(np.datetime64(t_utc, "s")).tz_localize("UTC").to_pydatetime()
        elif isinstance(t_utc, datetime) and t_utc.tzinfo is None:
            t_utc = t_utc.replace(tzinfo=UTC)
        return self.track.evaluate(t_utc)

    # ------------------------------------------------------------------
    # Dados / fluxo
    # ------------------------------------------------------------------
    def flux_density(self, freqs_mhz: Any = None) -> FluxDensity:
        """Curva de fluxo solar (Jy) × frequência (MHz) — no meio da janela.

        **Padrão: banda do BINGO** (``DEFAULT_FREQS_MHZ``, 1000–1500 MHz). O SRD
        real cobre 245–15400 MHz e bursts fora da banda (ex.: 15400 MHz) têm
        fluxos de ~1e7 Jy que destroem a escala do gráfico — por isso a curva
        padrão fica na banda do telescópio.

        Args:
            freqs_mhz: Grade alternativa de frequências (MHz). Se None, usa a
                banda padrão do BINGO.
        """
        if freqs_mhz is None:
            freqs = np.asarray(DEFAULT_FREQS_MHZ, dtype=float)
        else:
            freqs = np.asarray(freqs_mhz, dtype=float)
        # Mantém apenas frequências cobertas pelo dataset (evita NaN/linha vazia).
        freq_axis = self.flux.dataset["frequency_mhz"].values.astype(float)
        if freq_axis.size:
            in_range = (freqs >= freq_axis.min()) & (freqs <= freq_axis.max())
            freqs = freqs[in_range]
        times = self.flux.dataset["time"].values.astype("datetime64[ns]")
        t_mid = times[len(times) // 2]
        curve = self.flux.evaluate_grid(np.array([t_mid], dtype="datetime64[ns]"), freqs)[0]
        return FluxDensity(freqs, {"Sun": curve})

    # ------------------------------------------------------------------
    # Visualização (deferida ao radioskysims-viz)
    # ------------------------------------------------------------------
    def view_skychart(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """Carta do céu local ALT/AZ do Sol (matplotlib, lazy)."""
        from radioskysims_viz.sources import sun_skychart

        az = self.track.dataset["az_rad"].values.astype(float)
        el = self.track.dataset["el_rad"].values.astype(float)
        return sun_skychart(az, el, title=title, **kwargs)

    def view_mollweide(self, *, title: str | None = None, **kwargs: Any) -> Any:
        """RA/Dec do Sol no mapa Mollweide (matplotlib, lazy)."""
        from radioskysims_viz.sources import sun_mollweide

        ra = self.track.dataset["ra_rad"].values.astype(float)
        dec = self.track.dataset["dec_rad"].values.astype(float)
        return sun_mollweide(ra, dec, title=title, **kwargs)

    # ------------------------------------------------------------------
    # Introspect
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Resumo textual da fonte local (posição + fluxo de referência)."""
        fd = self.flux_density()
        f0 = float(fd.total[0]) if fd.total.size else float("nan")
        nu0 = float(fd.freqs_mhz[0]) if fd.freqs_mhz.size else float("nan")
        return (
            f"Sun — lat={self.observer.lat.to_value('deg'):.2f}°, "
            f"lon={self.observer.lon.to_value('deg'):.2f}° | "
            f"fluxo ≈ {f0:.3e} Jy @ {nu0:.0f} MHz"
        )

    def __repr__(self) -> str:
        return (
            f"<Sun: lat={self.observer.lat.to_value('deg'):.2f}°, "
            f"lon={self.observer.lon.to_value('deg'):.2f}°>"
        )
