"""Helpers de conversão segura usados pelo repositório de satélites (E8).

Extraído de ``satelliterepository.py`` (decomposição TD-011) para permitir
que o mixin de atualizações externas (``_updates.py``) dependa de um módulo
de apoio sem criar ciclo com o módulo principal.

Copyright BINGO Collaboration
"""

from __future__ import annotations

from typing import Any


def _safe_float(value: Any) -> float | None:
    """Converte para float sem quebrar em NaN/None/vazio."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f  # NaN → None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Converte para int sem quebrar em None/NaN/vazio."""
    if value is None:
        return None
    try:
        f = float(value)
        if f != f:  # NaN
            return None
        return int(f)
    except (TypeError, ValueError):
        return None
