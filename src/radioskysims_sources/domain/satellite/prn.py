"""Mapeamento NORAD ↔ Nome ↔ PRN para satélites GNSS.

O número **PRN** (Pseudo-Random Noise) é o identificador usado pelas
efemérides de transmissão (RINEX NAV) — ex.: ``G22``, ``E05``, ``C47`` —
e hoje, no RadioSkySim, só está disponível como texto livre dentro do
``name`` do catálogo CelesTrak (ex.: ``"GPS BIIR-5 (PRN 22)"``).

Este módulo torna o PRN um cidadão de primeira classe:

- ``parse_prn_from_name`` extrai o PRN do nome canônico CelesTrak;
- ``prn_to_sv`` / ``sv_to_prn`` fazem a ponte com o esquema de SV do RINEX
  (espelhando ``gnsstools4astro.domain.satellite``);
- ``norad_to_prn`` combina um mapa canônico (NORAD → PRN) com a extração
  do nome — usado na persistência (coluna ``prn`` em ``SatID``).

Isso viabiliza, por exemplo, casar satélites obtidos via **TLE** (identificados
por ``norad_id``) com os mesmos satélites obtidos via **RINEX** (identificados
por ``sv``/``prn``), e comparar posicionamento entre os dois métodos.
"""

from __future__ import annotations

import re

# Prefijo de SV por constelação — espelha gnsstools4astro.domain.satellite.
SV_PREFIX: dict[str, str] = {
    "GPS": "G",
    "GALILEO": "E",
    "GLONASS": "R",
    "BEIDOU": "C",
}

# Inverso: letra do SV → constelação canônica.
_PREFIX_TO_CONSTELLATION: dict[str, str] = {v: k for k, v in SV_PREFIX.items()}


def parse_prn_from_name(
    name: str | None,
    constellation: str | None = None,
) -> int | None:
    """Extrai o PRN do nome canônico CelesTrak de um satélite GNSS.

    Regras calibradas contra o catálogo real (CelesTrak GNSS):

    - **GPS**:      ``GPS BIIR-5 (PRN 22)``      → ``22``
    - **BEIDOU**:   ``BEIDOU-3 M25 (C47)``       → ``47``
    - **SBAS/outros com PRN**: ``GSAT-8 (GAGAN/PRN 127)`` → ``127``
    - **GLONASS**:  ``COSMOS 2433 (720)``        → ``720`` (número Glonass;
      o SV RINEX usa slot 01–24 e requer mapa próprio)
    - **GALILEO**:  ``GSAT0201 (GALILEO 5)``     → ``5`` (número do veículo;
      o PRN ``E01``–``E36`` não é derivável do nome — usar mapa canônico)

    A ordem das regras importa: padrões genéricos com ``(PRN …)`` têm
    prioridade sobre os específicos de constelação.
    """
    if not name:
        return None
    cons = (constellation or "").upper()

    # Genérico: "PRN N)" em qualquer posição — GPS e SBAS/WAAS/EGNOS/GAGAN/QZSS
    # (ex.: "(PRN 22)", "(GAGAN/PRN 127)", "(WAAS/PRN 131)", "(QZSS/PRN 194)").
    m = re.search(r"PRN\s+(\d{1,3})\)", name)
    if m:
        return int(m.group(1))

    # BeiDou: "(C NN)" — o número C é o próprio PRN.
    m = re.search(r"\(C\s?(\d{1,2})\)", name)
    if m:
        return int(m.group(1))

    if cons == "GLONASS":
        m = re.search(r"\((\d{3})\)", name)
        return int(m.group(1)) if m else None

    if cons == "GALILEO":
        m = re.search(r"\(GALILEO\s+(\d+)\)", name)
        return int(m.group(1)) if m else None

    return None


def sv_to_prn(sv: str) -> tuple[str, int] | None:
    """Converte um SV RINEX (ex.: ``G22``, ``E05``) em ``(constelação, PRN)``.

    Returns
    -------
    tuple[str, int] | None
        ``("GPS", 22)`` para ``"G22"``; ``None`` se o SV for desconhecido.
    """
    if not sv or len(sv) < 2:
        return None
    prefix = sv[0].upper()
    cons = _PREFIX_TO_CONSTELLATION.get(prefix)
    if cons is None or not sv[1:].isdigit():
        return None
    return cons, int(sv[1:])


def prn_to_sv(prn: int, constellation: str) -> str:
    """Converte ``(constelação, PRN)`` no SV RINEX correspondente.

    Ex.: ``prn_to_sv(22, "GPS")`` → ``"G22"``; ``prn_to_sv(5, "GALILEO")`` → ``"E05"``.

    Raises
    ------
    ValueError
        Se a constelação não tiver prefixo de SV conhecido.
    """
    prefix = SV_PREFIX.get((constellation or "").upper())
    if prefix is None:
        msg = f"constelação sem prefixo de SV: {constellation!r}"
        raise ValueError(msg)
    return f"{prefix}{prn:02d}"


def norad_to_prn(
    norad_id: int,
    name: str | None = None,
    constellation: str | None = None,
) -> int | None:
    """PRN de um satélite a partir do NORAD ID.

    Consulta primeiro o mapa canônico :data:`NORAD_TO_PRN` e, na ausência,
    tenta derivar do nome CelesTrak.
    """
    override = NORAD_TO_PRN.get(norad_id)
    if override is not None:
        return override
    return parse_prn_from_name(name, constellation)


# Mapa canônico NORAD → PRN para casos em que o PRN não é derivável do nome
# (ex.: GALILEO E01–E36) ou para corrigir exceções. Preenchido
# incrementalmente; vazio por padrão — a extração do nome cobre GPS, BeiDou
# e SBAS.
NORAD_TO_PRN: dict[int, int] = {}
