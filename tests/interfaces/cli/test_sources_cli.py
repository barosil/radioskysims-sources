from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

from radioskysims_sources.interfaces.cli import sources_cli


class _Manager:
    def __init__(self, size=2):
        self.size = size
        self.selected = None
        self.saved = []

    def select(self, **kwargs):
        self.selected = kwargs
        return self

    def summary(self):
        return "manager summary"

    def view_mollweide(self):
        return "mollweide"

    def view_flux_density(self):
        return "flux"

    def __len__(self):
        return self.size


def test_load_radio_uses_simulated_or_database(monkeypatch):
    simulated = _Manager()
    database = _Manager()
    monkeypatch.setattr(
        sources_cli.RadioSourcesManager,
        "simulated",
        staticmethod(lambda **_kwargs: simulated),
    )
    monkeypatch.setattr(
        sources_cli.RadioSourcesManager,
        "get_radio_sources",
        staticmethod(lambda: database),
    )

    assert sources_cli._load_radio(SimpleNamespace(source="simulated", n=7)) is simulated
    assert sources_cli._load_radio(SimpleNamespace(source="db", n=7)) is database


def test_radio_list_applies_minimum_flux_and_prints_summary(monkeypatch, capsys):
    manager = _Manager()
    args = SimpleNamespace(source="simulated", n=3, min_flux=1.5)
    monkeypatch.setattr(sources_cli, "_load_radio", lambda _args: manager)

    assert sources_cli._cmd_radio_list(args) == 0
    assert manager.selected == {"min_flux_jy": 1.5}
    assert capsys.readouterr().out == "manager summary\n"


def test_radio_plot_saves_both_figures(monkeypatch, tmp_path, capsys):
    manager = _Manager(size=4)
    monkeypatch.setattr(sources_cli, "_load_radio", lambda _args: manager)
    saved = []
    viz = ModuleType("radioskysims_viz")
    viz.save_figure = lambda figure, path: saved.append((figure, path))
    monkeypatch.setitem(sys.modules, "radioskysims_viz", viz)

    result = sources_cli._cmd_radio_plot(
        SimpleNamespace(source="simulated", n=4, outdir=str(tmp_path))
    )

    assert result == 0
    assert saved == [
        ("mollweide", str(tmp_path / "radio_sources_mollweide")),
        ("flux", str(tmp_path / "radio_sources_flux_density")),
    ]
    assert "4 fontes" in capsys.readouterr().out


def test_sun_without_plot_prints_flux(monkeypatch, capsys):
    class Flux:
        total = [12.3]
        freqs_mhz = [1400.0]

    sun = SimpleNamespace(flux_density=lambda: Flux())
    monkeypatch.setattr(sources_cli.Sun, "simulated", staticmethod(lambda: sun))

    assert sources_cli._cmd_sun(SimpleNamespace(plot=False, outdir="unused")) == 0
    assert "1.230e+01 Jy" in capsys.readouterr().out


def test_sun_plot_saves_three_figures(monkeypatch, tmp_path):
    class Flux:
        total = [1.0]
        freqs_mhz = [100.0]

        def view(self):
            return "flux-view"

    sun = SimpleNamespace(
        flux_density=lambda: Flux(),
        view_skychart=lambda: "sky",
        view_mollweide=lambda: "moll",
    )
    monkeypatch.setattr(sources_cli.Sun, "simulated", staticmethod(lambda: sun))
    saved = []
    viz = ModuleType("radioskysims_viz")
    viz.save_figure = lambda figure, path: saved.append((figure, path))
    monkeypatch.setitem(sys.modules, "radioskysims_viz", viz)

    sources_cli._cmd_sun(SimpleNamespace(plot=True, outdir=str(tmp_path)))

    assert [figure for figure, _ in saved] == ["sky", "moll", "flux-view"]


def test_main_returns_one_and_optionally_prints_traceback(monkeypatch, capsys):
    def fail(_args):
        message = "command failed"
        raise RuntimeError(message)

    monkeypatch.setattr(
        sources_cli,
        "_make_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: SimpleNamespace(func=fail, debug=True)),
    )

    assert sources_cli.main([]) == 1
    captured = capsys.readouterr()
    assert "Erro: command failed" in captured.err
    assert "Traceback" in captured.err
    logging.shutdown()
