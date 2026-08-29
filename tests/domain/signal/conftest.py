"""
Módulo: tests/domain/signal/conftest.py

Conftest para testes do domínio signal.
Garante resolução correta de imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[4] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
