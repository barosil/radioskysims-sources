"""
Módulo: src/radioskysims/domain/satellite/models.py

Descrição:
    Modelos de domínio para satélites e elementos orbitais.
    Define as entidades Satellite, OrbitalElement e SatelliteAssembled
    com suas regras de negócio e validações.
"""

# domain/satellite/models.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr
from astropy.time import Time
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .prn import prn_to_sv

INTERVAL_DAYS = 5


def as_utc(value: datetime | None) -> datetime | None:
    """Normaliza um datetime para UTC com timezone (naive → UTC).

    Convenção do projeto (TODO Satellites): todas as datas na aplicação são
    timezone-aware. Datas naive (ex.: lidas do banco/JSON) são interpretadas
    como UTC.
    """
    if value is None or not isinstance(value, datetime):
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class OrbitalElement(BaseModel):
    """
    Value Object de domínio para elementos orbitais.

    Encapsula os parâmetros orbitais de um satélite em um instante específico.
    Independente de infraestrutura, banco de dados ou ORM.
    """

    model_config = ConfigDict(extra="allow")

    epoch: datetime
    norad_id: int | None = None
    tle_line1: str | None = None
    tle_line2: str | None = None

    @field_validator("epoch")
    @classmethod
    def _epoch_utc(cls, value: datetime) -> datetime:
        return as_utc(value)


class Satellite(BaseModel):
    """
    Entidade de domínio representando um satélite artificial.

    Contém identificação, classificação e parâmetros orbitais.
    Valida consistência temporal entre epoch e elementos orbitais.

    Attributes:
        norad_id (int): Identificador NORAD do satélite.
        name (str | None): Nome do satélite.
        constellation (str | None): Constelação (GPS, Galileo, Starlink, etc.).
        prn (int | None): Número PRN (Pseudo-Random Noise) — identificador
            usado pelas efemérides RINEX (ex.: 22 para ``G22``). Deriva do
            nome CelesTrak na persistência; ``None`` quando não aplicável.
        epoch (datetime): Época de referência.
        element (OrbitalElement): Elementos orbitais do satélite.

    Raises:
        ValueError: Se a diferença entre epoch e element.epoch exceder INTERVAL_DAYS.
    """

    norad_id: int
    name: str | None = None
    constellation: str | None = None
    prn: int | None = None
    epoch: datetime
    element: OrbitalElement

    @property
    def sv(self) -> str | None:
        """SV RINEX correspondente (ex.: ``'G22'``) quando o PRN é conhecido.

        Ponte para as efemérides de transmissão: permite casar este satélite
        (via TLE/NORAD) com o mesmo satélite no RINEX NAV.
        """
        if self.prn is None or not self.constellation:
            return None
        try:
            return prn_to_sv(self.prn, self.constellation)
        except ValueError:
            return None

    @field_validator("epoch")
    @classmethod
    def _epoch_utc(cls, value: datetime) -> datetime:
        return as_utc(value)

    @model_validator(mode="after")
    def validate_epoch_interval(self):
        element_jd = Time(self.element.epoch).jd
        epoch_jd = Time(self.epoch).jd
        delta_days = abs(element_jd - epoch_jd)

        if delta_days > INTERVAL_DAYS:
            msg = (
                f"Epoch inconsistente: diferença de {delta_days:.2f} dias "
                f"excede o limite de {INTERVAL_DAYS} dias."
            )
            raise ValueError(msg)

        return self


class SatelliteAssembled:
    """
    Entidade de domínio representando um satélite observável.

    Combina informações de trajetória (tracks) e sinais transmitidos,
    fornecendo interface unificada para simulações.

    Attributes:
        name (str): Nome identificador do satélite.
        tracks (xr.Dataset): Dataset com trajetória ao longo do tempo.
        signals (xr.Dataset): Dataset com espectros de sinal.
    """

    def __init__(
        self,
        *,
        name: str,
        tracks: xr.Dataset,
        signals: xr.Dataset,
    ) -> None:
        """
        Inicializa satélite observável.

        Args:
            name (str): Nome do satélite.
            tracks (xr.Dataset): Trajetória do satélite.
                - dims: time
                - coords: angle, range, az, el, etc.
            signals (xr.Dataset): Sinais do satélite.
                - dims: freq
                - vars: PSD
        """
        self.name: str = name
        self.tracks: xr.Dataset = tracks
        self.signals: xr.Dataset = signals

    # ===========================
    # Introspécção (UX)
    # ===========================
    def __repr__(self) -> str:
        n_track = self.tracks.sizes.get("satellite", 0)
        names = self.name if isinstance(self.name, list) else [self.name]
        short = ", ".join(str(n) for n in names[:5])
        if len(names) > 5:
            short += f", … (+{len(names) - 5})"
        return (
            f"SatelliteAssembled({n_track} satélites [{short}]; "
            f"tracks {dict(self.tracks.sizes)}, signals {dict(self.signals.sizes)})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def norad_ids(self) -> list[int]:
        """NORAD IDs dos satélites (unicidade pela dim de tracks)."""
        return [int(v) for v in np.asarray(self.tracks.satellite.values)]

    @property
    def constellations(self) -> list[str]:
        """Constelações presentes."""
        if "constellation" in self.tracks.coords:
            return sorted({str(v) for v in self.tracks.constellation.values})
        return []

    def summary(self) -> str:
        """Resumo textual legível para inspeção rápida."""
        lines = [repr(self)]
        if "constellation" in self.tracks.coords:
            counts: dict[str, int] = {}
            for c in self.tracks.constellation.values:
                counts[str(c)] = counts.get(str(c), 0) + 1
            lines.append(
                "  por constelação: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )
        return "\n".join(lines)

    # ===========================
    # Seleção (UX de alto nível)
    # ===========================
    def select_Constellation(self, constellation: str | list[str]) -> "SatelliteAssembled":
        """Retorna um novo assembed contendo apenas satélites da(s) constelação(ões).

        Ex.: ``sats_live.select_Constellation("GPS")``.
        """
        wanted = [constellation] if isinstance(constellation, str) else list(constellation)
        wanted = {str(w).upper() for w in wanted}
        if "constellation" not in self.tracks.coords:
            msg = "O dataset de tracks não tem a coordenada 'constellation'."
            raise ValueError(msg)
        vals = [str(v).upper() for v in self.tracks.constellation.values]
        mask = np.isin(vals, list(wanted))
        return self._subset(mask)

    def select_Satellite(
        self,
        noradID: int | list[int] | None = None,
        name: str | list[str] | None = None,
    ) -> "SatelliteAssembled":
        """Retorna um novo assembed com satélites por NORAD ID e/ou nome.

        Ex.: ``sats_live.select_Satellite(noradID=25544)`` ou
        ``sats_live.select_Satellite(name=["NAVSTAR 63", "NAVSTAR 77"])``.
        """
        mask = np.ones(self.tracks.sizes.get("satellite", 0), dtype=bool)
        if noradID is not None:
            ids = [noradID] if isinstance(noradID, int) else list(noradID)
            mask &= np.isin(self.tracks.satellite.values, ids)
        if name is not None:
            wanted = [name] if isinstance(name, str) else list(name)
            names = np.asarray(self.tracks.names.values, dtype=str)
            mask &= np.isin(names, wanted)
        return self._subset(mask)

    def _subset(self, mask: np.ndarray) -> "SatelliteAssembled":
        """Filtra tracks e signals pelo mesmo mask de satélites (dim 'satellite')."""
        tracks = self.tracks.isel(satellite=mask)
        # signals tem uma linha por sinal (norad_id repetido) → filtra por valor
        sig_mask = np.isin(self.signals.satellite.values, np.asarray(tracks.satellite.values))
        signals = self.signals.isel(satellite=sig_mask)
        names = tracks.names.values.tolist() if "names" in tracks.coords else []
        return SatelliteAssembled(name=names, tracks=tracks, signals=signals)

    # ===========================
    # Visualização (delega lazy a radioskysims_viz)
    # ===========================
    def plot_tracks(
        self,
        smooth: bool = True,
        kde: bool = False,
        tracks: bool = True,
        figsize: tuple[float, float] = (9.0, 5.0),
        **kwargs: Any,
    ):
        """Vista Mollweide das órbitas-terra (sub-satellite points) dos satélites.

        - ``smooth=True``: aplica suavização (rolling mean) nas tracks;
        - ``tracks=False``: não plota as tracks individuais (mantém apenas os
          contornos de densidade);
        - ``contours=True``: contornos de densidade KDE 2D sobre TODOS os
          tracks (independente da constelação), com 5 níveis e colorbar
          horizontal inferior (KernelDensity do sklearn, métrica esférica
          haversine);
        - ``kde=True``: plota um heatmap KDE (densidade de passagens) em vez
          das tracks individuais;
        - ``bandwidth``: largura de banda do kernel KDE (graus); ``None`` →
          regra de Scott automática (sklearn.neighbors.KernelDensity);
        - ``coastlines=True``: contornos dos continentes (Natural Earth 110m,
          via cartopy) sobre o mapa.

        Retorna uma ``matplotlib.Figure``. O import de ``radioskysims-viz`` é
        lazy (renderização na camada de visualização).
        """
        from radioskysims_viz.satellite import tracks_mollweide

        return tracks_mollweide(
            self.tracks,
            smooth=smooth,
            kde=kde,
            tracks=tracks,
            figsize=figsize,
            **kwargs,
        )

    def plot_signals(self, satellites: Any = None, **kwargs: Any):
        """Vista holoviews dos espectros (PSD dB vs freq MHz) dos sinais.

        - ``satellites``: opcional; NORAD IDs, nomes ou constelações para
          restringir (delega a ``select_Satellite``/``select_Constellation``).
        """
        from radioskysims_viz.satellite import signals_plot

        target = self
        if isinstance(satellites, str):
            target = self.select_Constellation(satellites)
        elif satellites is not None:
            target = self.select_Satellite(noradID=satellites)

        norad_names = {
            int(n): str(nm)
            for n, nm in zip(
                np.asarray(target.tracks.satellite.values),
                np.asarray(target.tracks.names.values),
                strict=False,
            )
        }
        return signals_plot(target.signals, norad_names=norad_names, **kwargs)

    # ===========================
    # Interface FUNDAMENTAL
    # ===========================
    def angle(self) -> xr.DataArray:
        """Ângulo instantâneo satelite-feixe."""
        return self.tracks["theta"]

    def range(self) -> xr.DataArray:
        """Distância satelite-observador."""
        return self.tracks["range"]

    def psd(self) -> xr.DataArray:
        """PSD(freq) do satélite."""
        return self.signals["PSD"]
