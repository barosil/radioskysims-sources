"""
Módulo: src/radioskysims/domain/source/__init__.py

Descrição:
    Domínio unificado de fontes de sinal para simulação.

    Define a entidade ``Source`` que combina tracks (posições)
    e PSD (sinal) de qualquer tipo de fonte: satélites, fontes
    locais (Sun) e fontes de rádio (NVSS).

    A fábrica ``SourceFactory`` é o ponto de entrada para criar
    instâncias de ``Source`` a partir de dados específicos de
    cada tipo.

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

from .factory import SourceFactory
from .models import Source, SourceType

__all__ = [
    "Source",
    "SourceFactory",
    "SourceType",
]
