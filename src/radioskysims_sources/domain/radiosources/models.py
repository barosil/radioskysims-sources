"""\
Módulo: src/radioskysims/domain/radiosources/models.py

Modelos de domínio para o contexto de RadioSources.

Define entidades e value objects relacionados a fontes de rádio
contínuas (extragalácticas/galácticas), incluindo modelo espectral
mínimo viável para extrapolação de fluxo em frequência.

Copyright BINGO Collaboration
Last modified: 2026-02-26
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class SpectralModel(BaseModel):
    """Modelo espectral mínimo via lei de potência.

    Representa um espectro do tipo :math:`S(\\nu) = S_0 (\\nu/\\nu_0)^{-\\alpha}`
    com frequências expressas em **MHz** em todo o projeto.
    """

    nu0_mhz: float = Field(..., description="Frequência de referência em MHz.")
    s0_jy: float = Field(..., gt=0.0, description="Fluxo na frequência de referência (Jy).")
    alpha: float = Field(..., description="Índice espectral (alpha).")


class SpectralModelEvaluator(Protocol):
    """Protocolo para avaliadores de modelo espectral.

    Permite testar implementações alternativas sem acoplar o domínio
    a uma biblioteca numérica específica.
    """

    def __call__(self, nu_mhz: float | list[float]) -> float | list[float]:
        """Avalia o fluxo na(s) frequência(s) fornecida(s) em MHz."""


class RadioSource(BaseModel):
    """Entidade de domínio que representa uma fonte de rádio pontual.

    Mantém identidade, posição no céu e parâmetros espectrais mínimos
    necessários para simulação de foreground discreto.
    """

    id: str = Field(..., description="Identificador da fonte (ex.: nome NVSS ou ID interno).")
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Ascensão reta em graus.")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Declinação em graus.")
    spectral_model: SpectralModel = Field(..., description="Modelo espectral associado à fonte.")


class RadioSourceFluxSample(BaseModel):
    """Value object representando o fluxo de uma fonte em uma frequência."""

    source_id: str = Field(..., description="Identificador da fonte.")
    freq_mhz: float = Field(..., gt=0.0, description="Frequência em MHz.")
    flux_jy: float = Field(..., ge=0.0, description="Fluxo correspondente em Jy.")
