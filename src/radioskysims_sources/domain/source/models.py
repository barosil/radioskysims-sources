"""
Módulo: src/radioskysims/domain/source/models.py

Descrição:
    Entidade ``Source`` e tipos associados para representação
    unificada de qualquer fonte de sinal no pipeline de simulação.

    ``Source`` encapsula tracks (posições) e PSD (sinal) de forma
    agnóstica ao tipo de fonte, permitindo que satélites, fontes
    locais (Sun) e fontes de rádio (NVSS) sejam tratados com a
    mesma interface no pipeline.

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["satellite", "localsource", "radiosource"]


class Source(BaseModel):
    """Entidade de domínio que representa uma fonte de sinal.

    Combina informações de trajetória (tracks) e densidade espectral
    de potência (PSD) para qualquer tipo de fonte celestial.

    Attributes:
        name: Nome identificador da fonte.
        source_type: Tipo da fonte ('satellite', 'localsource', 'radiosource').
        tracks: Dataset com trajetória ao longo do tempo.
            Deve conter as variáveis:
                - theta (rad): ângulo zenital
                - az (rad): azimute
                - el (rad): elevação
                - range (m): distância ao observador
                - ra (rad, opcional): ascensão reta
                - dec (rad, opcional): declinação
            Dimensão: (time,).
        signals: Dataset com sinais da fonte.
            Deve conter a variável:
                - PSD (W/Hz): densidade espectral de potência
            Dimensão: (frequency,).
        metadata: Metadados adicionais (TLEs, arquivos de fluxo,
            referências de catálogo, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Nome identificador da fonte.")
    source_type: SourceType = Field(..., description="Tipo da fonte.")
    tracks: xr.Dataset = Field(
        ...,
        description="Dataset com trajetória (theta, az, el, range, [ra, dec]).",
    )
    signals: xr.Dataset = Field(
        ...,
        description="Dataset com sinal (PSD em W/Hz).",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Metadados adicionais (TLEs, arquivos, catálogos).",
    )

    # ── Acesso a tracks ──────────────────────────────────────────

    def angle(self) -> xr.DataArray:
        """Ângulo zenital instantâneo (rad)."""
        return self.tracks["theta"]

    def range(self) -> xr.DataArray:
        """Distância ao observador (m)."""
        return self.tracks["range"]

    def psd(self) -> xr.DataArray:
        """Densidade espectral de potência (W/Hz)."""
        return self.signals["PSD"]

    # ── Coordenadas equatoriais (derivadas) ─────────────────────

    @property
    def ra(self) -> xr.DataArray | None:
        """Ascensão reta (rad), se disponível nos tracks."""
        return self.tracks.get("ra")

    @property
    def dec(self) -> xr.DataArray | None:
        """Declinação (rad), se disponível nos tracks."""
        return self.tracks.get("dec")

    @property
    def ra_mean_deg(self) -> float | None:
        """Ascensão reta média (graus), para ordenação."""
        ra = self.ra
        if ra is None:
            return None
        return float(np.rad2deg(ra.mean().values))

    # ── Conveniência para o SignalSource Protocol ────────────────

    @property
    def source_type_str(self) -> str:
        """Retorna o tipo como string (compatível com SignalSource)."""
        return self.source_type
