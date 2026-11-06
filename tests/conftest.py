"""\
Módulo: tests/conftest.py

Descrição:
    Fixtures globais dos testes do pacote `radioskysims-sources`.

    Registra as fábricas concretas dos facades UX (TD-001) via composition
    root, reproduzindo o comportamento anterior em que os facades criavam os
    adapters reais por padrão.

Copyright BINGO Collaboration
Last modified: 2026-11-05
"""

from __future__ import annotations

import pytest

from radioskysims_sources.bootstrap.builders import configure_facade_defaults


@pytest.fixture(autouse=True)
def _configure_facade_defaults() -> None:
    """Registra as fábricas concretas usadas pelos facades UX (TD-001)."""
    configure_facade_defaults()
