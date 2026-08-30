"""
Módulo: src/radioskysims/domain/signal_source.py

Protocolo unificado e adapters para fontes de sinal no pipeline.

Define a interface comum (SignalSource) que satélites, fontes locais
(Sun) e fontes de rádio (NVSS) implementam, permitindo tratamento
homogêneo pelo pipeline de simulação.

Adapters concretos:
- SatelliteSignalSource: adapta SatelliteAssembled
- LocalSourceSignalSource: adapta LocalSourceTrack + SolarFlux
- RadioSourceSignalSource: adapta RadioSourceTrack + RadioSource

Copyright BINGO Collaboration
Last modified: 2026-07-27
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr


@runtime_checkable
class SignalSource(Protocol):
    """Protocolo unificado para qualquer fonte de sinal.

    Uma SignalSource fornece:
    - tracks (az, el, theta, range) no referencial do observador
    - PSD na grade de frequências da simulação, em:
        * ``W/Hz`` (potência total; ex.: satélites) — o ``TODSimulator``
          aplica o fator de distância ``r`` (``1/r²``) antes de converter
          para temperatura de antena;
        * ``W/m²/Hz`` (densidade de fluxo; ex.: Sun e rádio fontes) — o
          ``TODSimulator`` **não** aplica o fator de distância, pois a
          conversão já está normalizada por ``A_e`` on-axis (= ``λ²``).
    - Metadados de identificação.

    Isto permite que satélites, fontes locais (Sun) e fontes de
    rádio (NVSS) sejam tratados homogeneamente pelo pipeline.
    """

    @property
    def name(self) -> str:
        """Nome identificador da fonte."""
        ...

    @property
    def source_type(self) -> str:
        """Tipo da fonte: 'satellite', 'localsource', 'radiosource'."""
        ...

    @property
    def tracks(self) -> xr.Dataset:
        """Dataset com dims (time,) contendo az, el, theta, range.

        Variáveis esperadas no dataset:
            theta (time): ângulo zenital (rad)
            az (time): azimute (rad)
            el (time): elevação (rad)
            range (time): distância ao observador (m)
        """
        ...

    def psd(self) -> xr.DataArray:
        """Densidade espectral de potência na grade de frequências.

        Unidade indicada no atributo ``units`` do DataArray:
            - ``W/Hz``: potência total (o ``TODSimulator`` aplica o fator
              de distância ``1/r²``);
            - ``W/m²/Hz``: densidade de fluxo (o ``TODSimulator`` **não**
              aplica o fator de distância).

        Returns:
            DataArray com dim (frequency,) e coord `frequency` em Hz.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# Adapter: SatelliteAssembled -> SignalSource
# ═══════════════════════════════════════════════════════════════════


class SatelliteSignalSource:
    """Adapter que envolve SatelliteAssembled como SignalSource.

    SatelliteAssembled já expõe a interface necessária (angle, range,
    psd, tracks), mas com nomes de variáveis ligeiramente diferentes
    nos tracks (``theta``, ``az``, ``el``, ``range``).
    """

    def __init__(self, satellite_assembled: object) -> None:
        self._sat = satellite_assembled

    @property
    def name(self) -> str:
        return str(getattr(self._sat, "name", "unnamed_satellite"))

    @property
    def source_type(self) -> str:
        return "satellite"

    @property
    def tracks(self) -> xr.Dataset:
        """Retorna dataset com variáveis normalizadas (theta, az, el, range)."""
        raw = getattr(self._sat, "tracks", None)
        if raw is None:
            msg = f"Satellite '{self.name}' não possui tracks."
            raise AttributeError(msg)
        return raw

    def psd(self) -> xr.DataArray:
        psd_method = getattr(self._sat, "psd", None)
        if psd_method is None:
            msg = f"Satellite '{self.name}' não possui método psd()."
            raise AttributeError(msg)
        return psd_method()


# ═══════════════════════════════════════════════════════════════════
# Adapter: LocalSourceTrack + SolarFlux -> SignalSource
# ═══════════════════════════════════════════════════════════════════


class LocalSourceSignalSource:
    """Adapter que adapta LocalSourceTrack + SolarFlux em SignalSource.

    O sinal (densidade de fluxo) é convertido de Jy para W/m²/Hz.

    A densidade de fluxo variável no tempo é representada pela média
    numa janela de largura customizada (``mean_window_s``); se omitida,
    usa o tempo aproximado de um trânsito do Sol pelo feixe
    (``beam_fwhm_deg``), mantendo a estrutura esperada pelo TODSimulator.
    """

    def __init__(
        self,
        name: str,
        track: object,
        flux_jy_grid: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
        *,
        source_type: str = "localsource",
        time_seconds: np.ndarray | None = None,
        mean_window_s: float | None = None,
        beam_fwhm_deg: float | None = None,
        declination_deg: float = 0.0,
    ) -> None:
        self._name = name
        self._source_type = source_type
        self._track_obj = track
        self._flux_jy_grid = flux_jy_grid
        self._frequency_hz = frequency_hz
        self._time_seconds = time_seconds
        self._mean_window_s = mean_window_s
        self._beam_fwhm_deg = beam_fwhm_deg
        self._declination_deg = declination_deg

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def tracks(self) -> xr.Dataset:
        """Retorna dataset com variáveis normalizadas (theta, az, el, range).

        Faz o mapeamento de ``theta_rad`` -> ``theta``, ``az_rad`` -> ``az``,
        ``el_rad`` -> ``el``, ``range_m`` -> ``range``.
        """
        ds = getattr(self._track_obj, "dataset", None)
        if ds is None:
            msg = f"LocalSource '{self._name}' não possui dataset de track."
            raise AttributeError(msg)

        # Mapeia nomes de variáveis: _rad/_m -> sem sufixo
        rename_map = {
            "theta_rad": "theta",
            "az_rad": "az",
            "el_rad": "el",
            "range_m": "range",
        }
        to_rename = {k: v for k, v in rename_map.items() if k in ds.data_vars}
        if to_rename:
            ds = ds.rename(to_rename)

        # Garante que as variáveis existam
        required = {"theta", "az", "el", "range"}
        existing = set(ds.data_vars)
        missing = required - existing
        if missing:
            msg = f"LocalSourceTrack '{self._name}' sem variáveis: {sorted(missing)}"
            raise ValueError(msg)

        return ds

    def psd(self) -> xr.DataArray:
        """Retorna densidade de fluxo em W/m²/Hz.

        Para fontes locais (ex.: Sun), o fluxo disponível é em Jy
        (SolarFlux); converte via 1 Jy = 1e-26 W/m²/Hz.

        A média temporal é calculada numa janela de largura customizada
        (``mean_window_s``); se omitida, usa o tempo aproximado de um
        trânsito do Sol pelo feixe (função de ``beam_fwhm_deg``).

        Se não houver grid de fluxo, retorna um array de zeros.
        """
        if self._flux_jy_grid is None or self._frequency_hz is None:
            # Fallback: fluxo nulo (fonte sem sinal no grid da simulação)
            freq_hz = np.array([1.0])  # placeholder, será sobrescrito
            return xr.DataArray(
                np.zeros_like(freq_hz, dtype=float),
                dims="frequency",
                coords={"frequency": freq_hz},
                attrs={"units": "W/m²/Hz"},
            )

        # Conversão Jy -> densidade de fluxo (1 Jy = 1e-26 W/m²/Hz)
        flux_w_m2_hz = np.asarray(self._flux_jy_grid, dtype=float) * 1.0e-26

        flux_values = _flux_density_windowed_mean(
            flux_w_m2_hz,
            time_seconds=self._time_seconds,
            mean_window_s=self._mean_window_s,
            beam_fwhm_deg=self._beam_fwhm_deg,
            declination_deg=self._declination_deg,
        )

        # Guarda física: densidade de fluxo não pode ser negativa. Amostras
        # fora da cobertura temporal dos dados (ex.: extrapolação) ficam NaN
        # → tratadas como 0 (fonte não contribui naquele instante).
        flux_values = np.nan_to_num(flux_values, nan=0.0, posinf=0.0, neginf=0.0)
        flux_values = np.clip(flux_values, 0.0, None)

        return xr.DataArray(
            flux_values,
            dims="frequency",
            coords={"frequency": np.asarray(self._frequency_hz, dtype=float)},
            attrs={"units": "W/m²/Hz"},
        )


# ═══════════════════════════════════════════════════════════════════
# Adapter: RadioSourceTrack + RadioSource -> SignalSource
# ═══════════════════════════════════════════════════════════════════


class RadioSourceSignalSource:
    """Adapter que adapta RadioSourceTrack + RadioSource em SignalSource.

    O sinal (PSD) é calculado a partir do modelo espectral da fonte
    (lei de potência) convertido de Jy para W/Hz.
    """

    def __init__(
        self,
        source_id: str,
        track: object,
        flux_jy_grid: np.ndarray | None = None,
        frequency_hz: np.ndarray | None = None,
    ) -> None:
        self._source_id = source_id
        self._track_obj = track
        self._flux_jy_grid = flux_jy_grid
        self._frequency_hz = frequency_hz

    @property
    def name(self) -> str:
        return self._source_id

    @property
    def source_type(self) -> str:
        return "radiosource"

    @property
    def tracks(self) -> xr.Dataset:
        """Retorna dataset com variáveis normalizadas (theta, az, el, range)."""
        ds = getattr(self._track_obj, "dataset", None)
        if ds is None:
            msg = f"RadioSource '{self._source_id}' não possui dataset de track."
            raise AttributeError(msg)

        rename_map = {
            "theta_rad": "theta",
            "az_rad": "az",
            "el_rad": "el",
            "range_m": "range",
        }
        to_rename = {k: v for k, v in rename_map.items() if k in ds.data_vars}
        if to_rename:
            ds = ds.rename(to_rename)

        required = {"theta", "az", "el", "range"}
        existing = set(ds.data_vars)
        missing = required - existing
        if missing:
            msg = f"RadioSourceTrack '{self._source_id}' sem variáveis: {sorted(missing)}"
            raise ValueError(msg)

        return ds

    def psd(self) -> xr.DataArray:
        """Retorna densidade de fluxo em W/m²/Hz a partir do fluxo em Jy.

        Converte Jy -> W/m²/Hz (1 Jy = 1e-26 W/m²/Hz) para o pipeline.
        """
        if self._flux_jy_grid is None or self._frequency_hz is None:
            freq_hz = np.array([1.0])
            return xr.DataArray(
                np.zeros_like(freq_hz, dtype=float),
                dims="frequency",
                coords={"frequency": freq_hz},
                attrs={"units": "W/m²/Hz"},
            )

        flux_w_m2_hz = np.asarray(self._flux_jy_grid, dtype=float) * 1.0e-26
        return xr.DataArray(
            flux_w_m2_hz,
            dims="frequency",
            coords={"frequency": np.asarray(self._frequency_hz, dtype=float)},
            attrs={"units": "W/m²/Hz"},
        )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _sun_transit_window_s(beam_fwhm_deg: float, declination_deg: float = 0.0) -> float:
    """Tempo aproximado (s) de trânsito do Sol pelo feixe.

    O Sol varre o céu a ~15°/hora; o tempo para cruzar um feixe de
    largura FWHM (em RA, corrigido por cos(dec)) é:

        Δt = FWHM / (15°/h) / cos(dec)   (horas → segundos)

    Retorna ``nan`` se ``beam_fwhm_deg`` não for finito/positivo (o
    chamador deve cair na média total).
    """
    if not np.isfinite(beam_fwhm_deg) or beam_fwhm_deg <= 0:
        return float("nan")
    dec = np.deg2rad(float(declination_deg))
    cos_dec = max(float(np.cos(dec)), 1e-6)
    hours = (float(beam_fwhm_deg) / 15.0) / cos_dec
    return hours * 3600.0


def _flux_density_windowed_mean(
    flux_w_m2_hz: np.ndarray,
    *,
    time_seconds: np.ndarray | None = None,
    mean_window_s: float | None = None,
    beam_fwhm_deg: float | None = None,
    declination_deg: float = 0.0,
) -> np.ndarray:
    """Média temporal da densidade de fluxo numa janela de largura dada.

    - ``flux_w_m2_hz``: shape (time, frequency) ou (frequency,).
    - ``mean_window_s``: largura da janela (s). Se ``None`` e
      ``beam_fwhm_deg`` for fornecido, usa o trânsito do Sol pelo feixe.
    - A janela é centrada no instante de pico temporal do fluxo (trânsito).
    - Sem como definir janela, retorna a média total ao longo do tempo.
    """
    if flux_w_m2_hz.ndim == 1:
        return np.asarray(flux_w_m2_hz, dtype=float)

    if mean_window_s is None:
        if beam_fwhm_deg is None or not np.isfinite(beam_fwhm_deg) or beam_fwhm_deg <= 0:
            return np.nanmean(flux_w_m2_hz, axis=0)
        mean_window_s = _sun_transit_window_s(beam_fwhm_deg, declination_deg)

    if not np.isfinite(mean_window_s) or mean_window_s <= 0:
        return np.nanmean(flux_w_m2_hz, axis=0)

    n_time = flux_w_m2_hz.shape[0]
    if time_seconds is not None and len(time_seconds) > 1:
        dt = float(np.median(np.diff(time_seconds)))
    else:
        dt = 1.0
    if dt <= 0:
        dt = 1.0

    width = max(1, int(round(mean_window_s / dt)))
    # Janela centrada no pico temporal do fluxo (trânsito do Sol).
    # Linhas totalmente NaN (fora da cobertura dos dados) disparam o warning
    # "Mean of empty slice" no nanmean (via warnings, não errstate) —
    # suprimimos e caímos no meio do grid.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_means = np.nanmean(flux_w_m2_hz, axis=1)
    peak = int(np.nanargmax(row_means)) if not np.all(np.isnan(row_means)) else n_time // 2
    lo = max(0, peak - width // 2)
    hi = min(n_time, lo + width)
    if hi - lo < width:  # encostou na borda → expande para janela cheia
        lo = max(0, hi - width)
        hi = min(n_time, lo + width)

    return np.nanmean(flux_w_m2_hz[lo:hi], axis=0)


def compute_theta_boresight(
    tracks: xr.Dataset,
    pointing_ra_deg: float | None = None,
    pointing_dec_deg: float | None = None,
    pointing_alt_deg: float | None = None,
    pointing_az_deg: float | None = None,
    pointing_track_ra: xr.DataArray | None = None,
    pointing_track_dec: xr.DataArray | None = None,
) -> xr.DataArray:
    """Calcula o ângulo de separação entre o pointing e a fonte.

    Usa coordenadas equatoriais (RA/Dec) quando disponíveis nos tracks
    e no pointing, com fallback para coordenadas horizontais (alt-az).

    Aceita duas formas de pointing:

    - **Fixo** (``pointing_ra_deg`` / ``pointing_dec_deg``): coordenada
      única usada para todos os timesteps.
    - **Temporal** (``pointing_track_ra`` / ``pointing_track_dec``):
      DataArrays com dim ``(time,)`` — comparação ponto-a-ponto,
      útil quando o pointing varia no tempo (ex.: montagem alt-az fixa
      que varre RA/Dec devido à rotação da Terra).

    A separação angular é calculada via cosseno esférico:
        cos(γ) = sin(δ₁)·sin(δ₂) + cos(δ₁)·cos(δ₂)·cos(α₂ - α₁)

    O cálculo usa operações xarray/numpy que são lazy-friendly:
    quando os tracks são backed por dask, as operações (sin, cos,
    arccos, clip) propagam-se no grafo lazy.

    Args:
        tracks: Dataset com variáveis de track da fonte (radianos).
            Preferencialmente ``ra`` (rad) e ``dec`` (rad).
            Fallback: ``el`` (rad) e ``az`` (rad).
        pointing_ra_deg: RA fixa do pointing (graus).
        pointing_dec_deg: Dec fixa do pointing (graus).
        pointing_alt_deg: Altitude fixa do pointing (graus, fallback).
        pointing_az_deg: Azimute fixo do pointing (graus, fallback).
        pointing_track_ra: RA temporal do pointing (DataArray, deg).
            Dimensão ``(time,)``. Alternativa a ``pointing_ra_deg``.
        pointing_track_dec: Dec temporal do pointing (DataArray, deg).
            Dimensão ``(time,)``. Alternativa a ``pointing_dec_deg``.

    Returns:
        DataArray com ângulo de separação (rad), dimensão ``time``.

    Raises
    ------
    ValueError
        Se nem RA/Dec nem alt-az forem fornecidos.
    """
    # ── Tenta RA/Dec primeiro (equatorial) ──
    _ra = tracks.get("ra")
    _dec = tracks.get("dec")
    if _ra is not None and _dec is not None:
        _use_track = pointing_track_ra is not None and pointing_track_dec is not None
        _use_fixed = pointing_ra_deg is not None and pointing_dec_deg is not None

        if _use_track:
            # Pointing temporal: comparação ponto-a-ponto (dim time)
            ra0 = np.deg2rad(pointing_track_ra)
            dec0 = np.deg2rad(pointing_track_dec)
            desc = "Angular separation (RA/Dec, temporal pointing)."
            attrs_extra = {}
        elif _use_fixed:
            # Pointing fixo: mesma coordenada para todos os timesteps
            ra0 = np.deg2rad(float(pointing_ra_deg))
            dec0 = np.deg2rad(float(pointing_dec_deg))
            desc = "Angular separation (RA/Dec)."
            attrs_extra = {
                "pointing_ra_deg": float(pointing_ra_deg),
                "pointing_dec_deg": float(pointing_dec_deg),
            }
        else:
            # Sem RA/Dec no pointing → tenta fallback alt-az
            _ra = None

        if _ra is not None:
            cos_gamma = np.sin(_dec) * np.sin(dec0) + np.cos(_dec) * np.cos(dec0) * np.cos(
                _ra - ra0
            )
            cos_gamma = cos_gamma.clip(min=-1.0, max=1.0)
            gamma = np.arccos(cos_gamma)
            gamma.name = "angle_boresight"
            gamma.attrs.update(
                {
                    "units": "rad",
                    "description": desc,
                    **attrs_extra,
                }
            )
            return gamma

    # ── Fallback: alt-az (horizontal) ──
    if pointing_alt_deg is None or pointing_az_deg is None:
        msg = "compute_theta_boresight requer RA/Dec ou alt-az no pointing."
        raise ValueError(msg)

    el = tracks.get("el")
    az = tracks.get("az")
    if el is None or az is None:
        msg = "Tracks não contêm 'ra'/'dec' nem 'el'/'az' para calcular ângulo de separação."
        raise ValueError(
            msg
        )

    el0 = np.deg2rad(float(pointing_alt_deg))
    az0 = np.deg2rad(float(pointing_az_deg))
    cos_gamma = np.sin(el) * np.sin(el0) + np.cos(el) * np.cos(el0) * np.cos(az - az0)
    cos_gamma = cos_gamma.clip(min=-1.0, max=1.0)
    gamma = np.arccos(cos_gamma)
    gamma.name = "angle_boresight"
    gamma.attrs.update(
        {
            "units": "rad",
            "description": "Angular separation (alt-az fallback).",
            "pointing_alt_deg": float(pointing_alt_deg),
            "pointing_az_deg": float(pointing_az_deg),
        }
    )
    return gamma
