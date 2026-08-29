"""
Módulo: src/radioskysims/domain/source/factory.py

Descrição:
    Fábrica para criar instâncias de ``Source`` a partir de dados
    específicos de cada tipo de fonte (satélite, local, rádio).

    Cada método estático recebe os dados já resolvidos (tracks +
    sinais) e retorna um ``Source`` com a estrutura normalizada.

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .models import Source


class SourceFactory:
    """Fábrica para criar instâncias de Source.

    Uso:
        source = SourceFactory.from_satellite(
            name="GPS BIIR-2",
            tracks=geom_dataset,
            psd=psd_dataarray,
            metadata={"norad_id": 24876, "constellation": "GPS"},
        )
    """

    @staticmethod
    def _ensure_track_vars(tracks: xr.Dataset) -> xr.Dataset:
        """Garante que o dataset de tracks tenha as variáveis
        canônicas (theta, az, el, range) com os nomes corretos.

        Aceita nomes alternativos comuns:
            theta_rad -> theta
            az_rad -> az
            el_rad -> el
            range_m -> range
        """
        rename_map = {
            "theta_rad": "theta",
            "az_rad": "az",
            "el_rad": "el",
            "range_m": "range",
            "distance": "range",
        }
        to_rename = {k: v for k, v in rename_map.items() if k in tracks.data_vars}
        if to_rename:
            tracks = tracks.rename(to_rename)

        # Garante variáveis obrigatórias
        required = {"theta", "az", "el", "range"}
        existing = set(tracks.data_vars)
        missing = required - existing
        if missing:
            msg = f"Tracks sem variáveis obrigatórias: {sorted(missing)}"
            raise ValueError(msg)

        return tracks

    @staticmethod
    def _ensure_psd_var(signals: xr.Dataset) -> xr.Dataset:
        """Garante que o dataset de sinais tenha a variável PSD."""
        if "PSD" not in signals.data_vars:
            # Tenta mapear nomes alternativos
            for alt in ("psd", "power_spectral_density", "flux"):
                if alt in signals.data_vars:
                    signals = signals.rename({alt: "PSD"})
                    break
            else:
                msg = "Signals dataset não contém variável 'PSD'."
                raise ValueError(msg)

        if signals["PSD"].attrs.get("units") != "W/Hz":
            signals["PSD"].attrs["units"] = "W/Hz"

        return signals

    # ── Fábrica para satélites ──────────────────────────────────

    @staticmethod
    def from_satellite(
        name: str,
        tracks: xr.Dataset,
        psd: xr.DataArray,
        *,
        norad_id: int | None = None,
        constellation: str | None = None,
        tle_line1: str | None = None,
        tle_line2: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Cria Source a partir de dados de satélite.

        Args:
            name: Nome do satélite.
            tracks: Dataset com geometria (theta, az, el, range, [ra, dec]).
            psd: DataArray com PSD em W/Hz (dimensão frequency).
            norad_id: Identificador NORAD (opcional).
            constellation: Constelação (opcional).
            tle_line1, tle_line2: TLEs (opcional).
            extra_metadata: Metadados adicionais (opcional).

        Returns:
            Source configurado.
        """
        tracks = SourceFactory._ensure_track_vars(tracks)

        signals = xr.Dataset({"PSD": psd})

        metadata: dict[str, Any] = {
            "norad_id": norad_id,
            "constellation": constellation,
        }
        if tle_line1:
            metadata["tle_line1"] = tle_line1
        if tle_line2:
            metadata["tle_line2"] = tle_line2
        if extra_metadata:
            metadata.update(extra_metadata)

        return Source(
            name=name,
            source_type="satellite",
            tracks=tracks,
            signals=signals,
            metadata=metadata,
        )

    # ── Fábrica para fontes locais ──────────────────────────────

    @staticmethod
    def from_local_source(
        name: str,
        tracks: xr.Dataset,
        psd: xr.DataArray,
        *,
        flux_jy_grid: np.ndarray | None = None,
        source_file: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Cria Source a partir de fonte local (ex.: Sun).

        Args:
            name: Nome da fonte (ex.: 'Sun').
            tracks: Dataset com geometria (theta, az, el, range, [ra, dec]).
            psd: DataArray com PSD em W/Hz (dimensão frequency).
            flux_jy_grid: Grid de fluxo em Jy (opcional, para proveniência).
            source_file: Arquivo de origem dos dados (opcional).
            extra_metadata: Metadados adicionais (opcional).

        Returns:
            Source configurado.
        """
        tracks = SourceFactory._ensure_track_vars(tracks)
        signals = xr.Dataset({"PSD": psd})

        metadata: dict[str, Any] = {}
        if flux_jy_grid is not None:
            metadata["flux_jy_grid_shape"] = list(flux_jy_grid.shape)
        if source_file:
            metadata["source_file"] = source_file
        if extra_metadata:
            metadata.update(extra_metadata)

        return Source(
            name=name,
            source_type="localsource",
            tracks=tracks,
            signals=signals,
            metadata=metadata,
        )

    # ── Fábrica para fontes de rádio ────────────────────────────

    @staticmethod
    def from_radio_source(
        source_id: str,
        tracks: xr.Dataset,
        psd: xr.DataArray,
        *,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        flux_jy: float | None = None,
        spectral_index: float | None = None,
        catalog: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Cria Source a partir de fonte de rádio (ex.: NVSS).

        Args:
            source_id: Identificador da fonte no catálogo.
            tracks: Dataset com geometria (theta, az, el, range, [ra, dec]).
            psd: DataArray com PSD em W/Hz (dimensão frequency).
            ra_deg: Ascensão reta (graus, opcional).
            dec_deg: Declinação (graus, opcional).
            flux_jy: Fluxo em Jy (opcional).
            spectral_index: Índice espectral (opcional).
            catalog: Nome do catálogo de origem (opcional).
            extra_metadata: Metadados adicionais (opcional).

        Returns:
            Source configurado.
        """
        tracks = SourceFactory._ensure_track_vars(tracks)
        signals = xr.Dataset({"PSD": psd})

        metadata: dict[str, Any] = {
            "catalog": catalog,
        }
        if ra_deg is not None:
            metadata["ra_deg"] = ra_deg
        if dec_deg is not None:
            metadata["dec_deg"] = dec_deg
        if flux_jy is not None:
            metadata["flux_jy"] = flux_jy
        if spectral_index is not None:
            metadata["spectral_index"] = spectral_index
        if extra_metadata:
            metadata.update(extra_metadata)

        return Source(
            name=source_id,
            source_type="radiosource",
            tracks=tracks,
            signals=signals,
            metadata=metadata,
        )

    # ── Fábrica para grupos de rádio-fontes (colapsadas) ────────

    @staticmethod
    def from_radio_group(
        group_name: str,
        tracks: xr.Dataset,
        psd_sum: xr.DataArray,
        *,
        n_sources: int = 0,
        ra_mean_deg: float | None = None,
        dec_mean_deg: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Source:
        """Cria Source a partir de um grupo de rádio-fontes colapsadas.

        Todas as fontes no grupo compartilham a mesma posição (RA/Dec).
        A PSD é a soma incoerente das PSDs individuais.

        Args:
            group_name: Nome do grupo.
            tracks: Dataset com geometria (theta, az, el, range, [ra, dec]).
            psd_sum: DataArray com PSD total (W/Hz), soma das PSDs.
            n_sources: Número de fontes no grupo.
            ra_mean_deg: RA média (graus).
            dec_mean_deg: Dec média (graus).
            extra_metadata: Metadados adicionais.

        Returns:
            Source configurado como radiosource_group.
        """
        tracks = SourceFactory._ensure_track_vars(tracks)
        signals = xr.Dataset({"PSD": psd_sum})

        metadata: dict[str, Any] = {
            "n_sources_in_group": n_sources,
            "group_name": group_name,
        }
        if ra_mean_deg is not None:
            metadata["ra_mean_deg"] = ra_mean_deg
        if dec_mean_deg is not None:
            metadata["dec_mean_deg"] = dec_mean_deg
        if extra_metadata:
            metadata.update(extra_metadata)

        signals["PSD"].attrs["n_sources"] = n_sources

        return Source(
            name=group_name,
            source_type="radiosource",  # mantém radiosource para compatibilidade
            tracks=tracks,
            signals=signals,
            metadata=metadata,
        )
