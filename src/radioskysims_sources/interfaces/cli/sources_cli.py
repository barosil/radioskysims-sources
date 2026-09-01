# interfaces/cli/sources_cli.py
"""\
CLI simples para o contexto de fontes (RadioSources + LocalSources/Sol).

Subcomandos:
    radio list   — lista fontes de rádio (tabela resumo)
    radio plot   — salva mapas RA/Dec e curvas de fluxo (PNG)
    sun          — info do Sol e (opcionalmente) salva skychart/mapa/fluxo

Exemplos:
    python -m radioskysims_sources.interfaces.cli.sources_cli radio list --source simulated --n 20
    python -m radioskysims_sources.interfaces.cli.sources_cli radio plot --outdir plots
    python -m radioskysims_sources.interfaces.cli.sources_cli sun --plot --outdir plots

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

from radioskysims_sources.application.services.local_sources_manager import Sun
from radioskysims_sources.application.services.radio_sources_manager import (
    RadioSourcesManager,
)


def _load_radio(args: argparse.Namespace) -> RadioSourcesManager:
    if args.source == "db":
        return RadioSourcesManager.get_radio_sources()
    return RadioSourcesManager.simulated(n_sources=args.n)


def _cmd_radio_list(args: argparse.Namespace) -> int:
    mgr = _load_radio(args)
    if args.min_flux is not None:
        mgr = mgr.select(min_flux_jy=args.min_flux)
    print(mgr.summary())
    return 0


def _cmd_radio_plot(args: argparse.Namespace) -> int:
    from radioskysims_viz import save_figure

    mgr = _load_radio(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_figure(mgr.view_mollweide(), str(outdir / "radio_sources_mollweide"))
    save_figure(
        mgr.view_flux_density(), str(outdir / "radio_sources_flux_density")
    )
    print(f"RadioSources: {len(mgr)} fontes → plots em {outdir}/")
    return 0


def _cmd_sun(args: argparse.Namespace) -> int:
    sun = Sun.simulated()
    print(sun)
    print(f"  fluxo ~ {sun.flux_density().total[0]:.3e} Jy @ {sun.flux_density().freqs_mhz[0]:.0f} MHz")
    if args.plot:
        from radioskysims_viz import save_figure

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        save_figure(sun.view_skychart(), str(outdir / "sun_skychart"))
        save_figure(sun.view_mollweide(), str(outdir / "sun_mollweide"))
        save_figure(
            sun.flux_density().view(), str(outdir / "sun_flux_density")
        )
        print(f"  plots salvos em {outdir}/")
    return 0


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="radioskysim-sources",
        description="Fontes de sinal: fontes de rádio e Sol.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    rl = sub.add_parser("radio", help="comandos de fontes de rádio")
    rl_sub = rl.add_subparsers(dest="radio_cmd", required=True)

    rl_list = rl_sub.add_parser("list", help="lista fontes de rádio")
    rl_list.add_argument("--source", default="simulated", choices=["simulated", "db"])
    rl_list.add_argument("--n", type=int, default=40)
    rl_list.add_argument("--min-flux", type=float, default=None)
    rl_list.set_defaults(func=_cmd_radio_list)

    rl_plot = rl_sub.add_parser("plot", help="salva mapas e curvas de fluxo")
    rl_plot.add_argument("--source", default="simulated", choices=["simulated", "db"])
    rl_plot.add_argument("--n", type=int, default=40)
    rl_plot.add_argument("--outdir", default="plots")
    rl_plot.set_defaults(func=_cmd_radio_plot)

    sun = sub.add_parser("sun", help="informações/plots do Sol")
    sun.add_argument("--plot", action="store_true", help="salva plots (PNG)")
    sun.add_argument("--outdir", default="plots")
    sun.set_defaults(func=_cmd_sun)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
