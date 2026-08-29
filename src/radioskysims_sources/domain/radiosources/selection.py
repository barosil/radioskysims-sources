"""
Módulo: src/radioskysims/domain/radiosources/selection.py

Descrição:
    Seleção de fontes de rádio por proximidade angular ao pointing do
    telescópio. Permite descartar, ANTES de gerar tracks/sinais, fontes
    que jamais entram no feixe — reduzindo objetos pesados (tracks, fluxos
    e grafos dask) e trabalho de propagação.

    A separação angular usa a fórmula padrão de radioastronomia (lei dos
    cossenos esféricos):

        cos γ = sin δ₁·sin δ₂ + cos δ₁·cos δ₂·cos(α₁ − α₂)

    com o pointing podendo ser fixo (RA/Dec únicos) ou temporal (varredura
    de RA/Dec ao longo do tempo, ex.: montagem alt-az fixa tipo BINGO).
    Para pointing temporal, a fonte é mantida se a MENOR separação ao
    caminho apontado estiver dentro do limiar (N×FWHM do feixe).

Copyright BINGO Collaboration
Last modified: 2026-08-15
"""

from __future__ import annotations

import numpy as np

from radioskysims_sources.domain.radiosources.models import RadioSource

DEFAULT_RADIUS_DEG = 5.0


def angular_separation_deg(
    ra1_deg: float | np.ndarray,
    dec1_deg: float | np.ndarray,
    ra2_deg: float | np.ndarray,
    dec2_deg: float | np.ndarray,
) -> np.ndarray:
    """Separação angular (graus) via lei dos cossenos esféricos.

    Vetorizado: as entradas podem ser escalares ou arrays (broadcast).
    Lida naturalmente com o wrap de RA (cos α₁ − α₂).
    """
    ra1 = np.deg2rad(np.asarray(ra1_deg, dtype=float))
    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=float))
    ra2 = np.deg2rad(np.asarray(ra2_deg, dtype=float))
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=float))
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_sep))


def pointing_ra_dec_deg(
    *,
    pointing_track=None,
    pointing_ra_deg: float | None = None,
    pointing_dec_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """RA/Dec (graus) do pointing — temporal (track) ou fixo.

    ``pointing_track`` deve ser um objeto com variáveis ``ra``/``dec`` em
    **graus** (ex.: o xr.Dataset de ``compute_pointing_track``, que já
    retorna RA/Dec em graus com ``units="deg"``).

    Returns:
        (ra_deg, dec_deg) arrays, ou ``None`` se não houver pointing.
    """
    if pointing_track is not None:
        ra = pointing_track.get("ra")
        dec = pointing_track.get("dec")
        if ra is not None and dec is not None:
            # compute_pointing_track já fornece RA/Dec em graus — não aplicar
            # rad2deg aqui (isso inflaria os valores em ~57× e quebraria o
            # filtro angular do feixe).
            return (
                np.asarray(ra.values, dtype=float),
                np.asarray(dec.values, dtype=float),
            )
    if pointing_ra_deg is not None and pointing_dec_deg is not None:
        return (
            np.asarray([float(pointing_ra_deg)], dtype=float),
            np.asarray([float(pointing_dec_deg)], dtype=float),
        )
    return None


def min_separation_to_pointing(
    source_ra_deg: float,
    source_dec_deg: float,
    *,
    pointing_track=None,
    pointing_ra_deg: float | None = None,
    pointing_dec_deg: float | None = None,
) -> float | None:
    """Menor separação angular (graus) entre a fonte e o pointing.

    Para pointing temporal, usa o mínimo sobre todos os instantes: a fonte
    é considerada no feixe se passar pelo caminho apontado em algum momento.

    Returns:
        Separação mínima em graus, ou ``None`` se não houver pointing.
    """
    radec = pointing_ra_dec_deg(
        pointing_track=pointing_track,
        pointing_ra_deg=pointing_ra_deg,
        pointing_dec_deg=pointing_dec_deg,
    )
    if radec is None:
        return None
    ra_p, dec_p = radec
    sep = angular_separation_deg(source_ra_deg, source_dec_deg, ra_p, dec_p)
    return float(np.min(sep))


def _resolve_threshold(
    fwhm_deg: float | None,
    factor: float,
    radius_deg: float | None,
) -> float | None:
    """Limiar angular (graus): ``radius_deg`` se dado, senão ``fwhm*factor``."""
    if radius_deg is not None and radius_deg > 0:
        return float(radius_deg)
    if fwhm_deg is not None and np.isfinite(fwhm_deg) and fwhm_deg > 0:
        return float(fwhm_deg) * float(factor)
    return None


def select_sources_in_beam(
    sources: list[RadioSource],
    *,
    pointing_track=None,
    pointing_ra_deg: float | None = None,
    pointing_dec_deg: float | None = None,
    fwhm_deg: float | None = None,
    factor: float = 3.0,
    radius_deg: float | None = None,
) -> tuple[list[RadioSource], list[float | None]]:
    """Filtra fontes de rádio que estão no feixe do telescópio.

    Mantém fontes cuja menor separação angular ao pointing seja <= limiar.
    O limiar é ``radius_deg`` (se fornecido), senão ``fwhm_deg * factor``.
    Sem pointing ou sem limiar, mantém todas (separação ``None``).

    Args:
        sources: Fontes candidatas (RA/Dec fixas).
        pointing_track: Varredura de pointing (ra/dec em radianos) ou None.
        pointing_ra_deg / pointing_dec_deg: Pointing fixo (graus).
        fwhm_deg: Largura à meia-potência do feixe (graus).
        factor: Multiplicador do FWHM para o limiar (padrão 3×).
        radius_deg: Raio angular explícito (sobrepõe ``fwhm*factor``).

    Returns:
        (mantidas, separações_deg) — separação ``None`` quando não há
        pointing disponível (fonte mantida por falta de filtro).
    """
    if not sources:
        return [], []

    threshold = _resolve_threshold(fwhm_deg, factor, radius_deg)

    kept: list[RadioSource] = []
    separations: list[float | None] = []
    for src in sources:
        sep = min_separation_to_pointing(
            src.ra_deg,
            src.dec_deg,
            pointing_track=pointing_track,
            pointing_ra_deg=pointing_ra_deg,
            pointing_dec_deg=pointing_dec_deg,
        )
        if sep is None or threshold is None or sep <= threshold:
            kept.append(src)
            separations.append(sep)
    return kept, separations


def pointing_region(
    *,
    pointing_track=None,
    pointing_ra_deg: float | None = None,
    pointing_dec_deg: float | None = None,
) -> tuple[float, float, float] | None:
    """Região circular (centro_ra, centro_dec, raio_graus) que cobre o pointing.

    Para pointing fixo, o raio é 0. Para temporal, usa a média circular da
    RA (soma de vetores unitários) e o máximo afastamento como raio de
    cobertura — permite centralizar a consulta de catálogo sem perder fontes
    ao longo da varredura.

    Returns:
        (ra_deg, dec_deg, raio_deg), ou ``None`` se não houver pointing.
    """
    radec = pointing_ra_dec_deg(
        pointing_track=pointing_track,
        pointing_ra_deg=pointing_ra_deg,
        pointing_dec_deg=pointing_dec_deg,
    )
    if radec is None:
        return None
    ra_p, dec_p = radec
    if ra_p.size == 1:
        return float(ra_p[0]), float(dec_p[0]), 0.0

    ra_rad = np.deg2rad(ra_p)
    dec_rad = np.deg2rad(dec_p)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    ux, uy, uz = float(x.mean()), float(y.mean()), float(z.mean())
    norm = np.hypot(ux, uy)
    center_ra = float(np.rad2deg(np.arctan2(uy, ux)) % 360.0)
    center_dec = float(np.rad2deg(np.arctan2(uz, norm)))
    coverage = float(np.max(angular_separation_deg(center_ra, center_dec, ra_p, dec_p)))
    return center_ra, center_dec, coverage
