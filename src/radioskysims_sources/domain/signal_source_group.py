"""
Módulo: src/radioskysims/domain/signal_source_group.py

Descrição:
    Agrupador de fontes de rádio com mesma posição (RA/Dec).
    Soma as PSDs incoerentemente e mantém um track único,
    colapsando a complexidade do grafo dask.

    Uso típico no SourceAssembler:
        grupos = RadioSourceGroupSignalSource.group_by_position(radio_sources)
        # grupos tem 1 SignalSource por posição única

Copyright BINGO Collaboration
Last modified: 2026-07-28
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import xarray as xr

from radioskysims_sources.domain.signal_source import RadioSourceSignalSource


class RadioSourceGroupSignalSource:
    """SignalSource que agrupa múltiplas fontes de rádio com mesma posição.

    Em vez de N fontes de rádio individuais no pipeline (cada uma com seu
    próprio grafo dask para track + beam + T_antenna), as fontes no mesmo
    RA/Dec são agrupadas. Seus sinais (PSD) são somados incoerentemente
    e um track único é mantido, reduzindo a complexidade do grafo dask
    de O(N) para O(grupos).

    Attributes:
        name (str): Nome do grupo.
        source_type (str): Sempre ``"radiosource_group"``.
        _sources (list[RadioSourceSignalSource]): Fontes no grupo.
        _sum_psd (xr.DataArray | None): PSD total (soma), lazy.
    """

    def __init__(
        self,
        name: str,
        sources: list[RadioSourceSignalSource],
    ) -> None:
        if not sources:
            msg = "RadioSourceGroupSignalSource requer ao menos uma fonte."
            raise ValueError(msg)

        self._name = name
        self._sources = sources
        self._sum_psd: xr.DataArray | None = None

    # ── Protocolo SignalSource ─────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "radiosource_group"

    @property
    def tracks(self) -> xr.Dataset:
        """Retorna o track da primeira fonte (posição compartilhada).

        Todas as fontes no grupo têm a mesma posição RA/Dec, então
        o track é idêntico. Usamos o da primeira fonte.
        """
        return self._sources[0].tracks

    def psd(self) -> xr.DataArray:
        """Retorna a soma incoerente das PSDs de todas as fontes no grupo.

        A soma é lazy (dask) se as PSDs individuais forem lazy.
        O resultado é cacheado em ``_sum_psd``.
        """
        if self._sum_psd is not None:
            return self._sum_psd

        psds = [src.psd() for src in self._sources]
        total = psds[0].copy()
        for p in psds[1:]:
            total = total + p

        total.name = "psd"
        # Soma incoerente de densidades de fluxo → mesma unidade dos membros
        total.attrs["units"] = "W/m²/Hz"
        total.attrs["n_sources"] = len(self._sources)
        total.attrs["group_name"] = self._name
        self._sum_psd = total
        return total

    # ── Métodos de fábrica ─────────────────────────────────────

    @staticmethod
    def collapse_by_fwhm(
        sources: list,
        *,
        fwhm_deg: float = 1.0,
        collapse_factor: float = 0.5,
    ) -> list:
        """Agrupa fontes de rádio cuja separação angular é menor que
        uma fração da FWHM do feixe.

        Fontes com separação angular < ``fwhm_deg * collapse_factor``
        são consideradas não-resolvíveis pelo telescópio e agrupadas
        num único ``RadioSourceGroupSignalSource`` com PSD soma.

        Args:
            sources: Lista de objetos compatíveis com SignalSource.
            fwhm_deg: Largura à meia-potência do feixe (graus).
            collapse_factor: Fração da FWHM para agrupamento
                (ex.: 0.5 = FWHM/2).

        Returns:
            Lista de objetos compatíveis com SignalSource, um por grupo.
        """
        if not sources:
            return []

        threshold_deg = fwhm_deg * collapse_factor

        # Extrai posição RA/Dec
        positions: list[tuple[float, float]] = []
        for src in sources:
            try:
                ds = src.tracks  # type: ignore[union-attr]
                ra = float(ds["ra"].mean().values) if "ra" in ds else float("nan")
                dec = float(ds["dec"].mean().values) if "dec" in ds else float("nan")
            except (AttributeError, KeyError, ValueError):
                ra = float("nan")
                dec = float("nan")
            positions.append((ra, dec))

        # Agrupa por separação angular
        buckets: dict[int, list] = {}
        assigned = [False] * len(sources)

        for i in range(len(sources)):
            if assigned[i]:
                continue
            ra_i, dec_i = positions[i]
            if np.isnan(ra_i) or np.isnan(dec_i):
                bucket_id = len(buckets)
                buckets.setdefault(bucket_id, []).append(sources[i])
                assigned[i] = True
                continue

            bucket_id = len(buckets)
            buckets.setdefault(bucket_id, []).append(sources[i])
            assigned[i] = True

            for j in range(i + 1, len(sources)):
                if assigned[j]:
                    continue
                ra_j, dec_j = positions[j]
                if np.isnan(ra_j) or np.isnan(dec_j):
                    continue
                # Separação angular aproximada (pequenos ângulos)
                d_ra = np.deg2rad(ra_j - ra_i) * np.cos(np.deg2rad(0.5 * (dec_i + dec_j)))
                d_dec = np.deg2rad(dec_j - dec_i)
                sep_deg = np.rad2deg(np.sqrt(d_ra**2 + d_dec**2))
                if sep_deg <= threshold_deg:
                    buckets[bucket_id].append(sources[j])
                    assigned[j] = True

        result: list = []
        for bid, group_sources in buckets.items():
            if len(group_sources) == 1:
                result.append(group_sources[0])
            else:
                ra_mean = np.nanmean([positions[sources.index(s)][0] for s in group_sources])
                dec_mean = np.nanmean([positions[sources.index(s)][1] for s in group_sources])
                name = f"radio_group_ra{ra_mean:.1f}_dec{dec_mean:.1f}_n{len(group_sources)}"
                result.append(RadioSourceGroupSignalSource(name, group_sources))

        return result

    @staticmethod
    def group_by_position(
        sources: list,
        *,
        ra_dec_tol_deg: float = 1e-6,
    ) -> list:
        """Agrupa fontes de rádio por posição (RA/Dec).

        Fontes com RA e Dec dentro de ``ra_dec_tol_deg`` são consideradas
        co-localizadas e agrupadas num único ``RadioSourceGroupSignalSource``.

        Fontes isoladas (posição única) são retornadas como estão.

        Args:
            sources: Lista de objetos compatíveis com SignalSource
                (ex.: RadioSourceSignalSource) para agrupar.
            ra_dec_tol_deg: Tolerância para considerar mesma posição (graus).

        Returns:
            Lista de objetos compatíveis com SignalSource, um por posição única.
        """
        if not sources:
            return []

        # Extrai posição RA/Dec de cada fonte a partir dos tracks
        positions: list[tuple[float, float]] = []
        for src in sources:
            try:
                ds = src.tracks  # type: ignore[union-attr]
                ra = float(ds["ra"].mean().values) if "ra" in ds else float("nan")
                dec = float(ds["dec"].mean().values) if "dec" in ds else float("nan")
            except (AttributeError, KeyError, ValueError):
                ra = float("nan")
                dec = float("nan")
            positions.append((ra, dec))

        # Agrupa por buckets de posição
        buckets: dict[tuple[float, float], list] = defaultdict(list)
        assigned = [False] * len(sources)

        for i, (ra, dec) in enumerate(positions):
            if assigned[i]:
                continue
            if np.isnan(ra) or np.isnan(dec):
                buckets[(float("nan"), float("nan"))].append(sources[i])
                assigned[i] = True
                continue

            group_key: tuple[float, float] = (round(ra, 6), round(dec, 6))
            buckets[group_key].append(sources[i])
            assigned[i] = True

            for j in range(i + 1, len(sources)):
                if assigned[j]:
                    continue
                rj, dj = positions[j]
                if np.isnan(rj) or np.isnan(dj):
                    continue
                if abs(rj - ra) <= ra_dec_tol_deg and abs(dj - dec) <= ra_dec_tol_deg:
                    buckets[group_key].append(sources[j])
                    assigned[j] = True

        result: list = []
        for (ra, dec), group_sources in buckets.items():
            if len(group_sources) == 1:
                # Fonte isolada: retorna a original
                result.append(group_sources[0])
            else:
                name = f"radio_group_ra{ra:.1f}_dec{dec:.1f}_n{len(group_sources)}"
                result.append(RadioSourceGroupSignalSource(name, group_sources))

        return result
