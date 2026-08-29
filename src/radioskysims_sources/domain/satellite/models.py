"""
Módulo: src/radioskysims/domain/satellite/models.py

Descrição:
    Modelos de domínio para satélites e elementos orbitais.
    Define as entidades Satellite, OrbitalElement e SatelliteAssembled
    com suas regras de negócio e validações.
"""

# domain/satellite/models.py
from __future__ import annotations

from datetime import datetime

import xarray as xr
from astropy.time import Time
from pydantic import BaseModel, ConfigDict, model_validator

INTERVAL_DAYS = 5


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


class Satellite(BaseModel):
    """
    Entidade de domínio representando um satélite artificial.

    Contém identificação, classificação e parâmetros orbitais.
    Valida consistência temporal entre epoch e elementos orbitais.

    Attributes:
        norad_id (int): Identificador NORAD do satélite.
        name (str | None): Nome do satélite.
        constellation (str | None): Constelação (GPS, Galileo, Starlink, etc.).
        epoch (datetime): Época de referência.
        element (OrbitalElement): Elementos orbitais do satélite.

    Raises:
        ValueError: Se a diferença entre epoch e element.epoch exceder INTERVAL_DAYS.
    """

    norad_id: int
    name: str | None = None
    constellation: str | None = None
    epoch: datetime
    element: OrbitalElement

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
