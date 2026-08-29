"""\
Módulo: src/radioskysims/domain/localsources/__init__.py

Descrição:
    Entidades e value objects do agregado LocalSources.
    Inclui modelo de fluxo solar interpolável no tempo e frequência.

Copyright BINGO Collaboration
Last modified: 2026-03-03
"""

from __future__ import annotations

from .models import LocalSource, LocalSources, LocalSourceTrack, SolarFlux

__all__ = ["LocalSource", "LocalSourceTrack", "LocalSources", "SolarFlux"]
