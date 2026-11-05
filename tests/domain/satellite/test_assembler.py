from __future__ import annotations

from types import SimpleNamespace

import dask
import numpy as np
import xarray as xr

from radioskysims_sources.domain.satellite.assembler import SatelliteAssembler


class DummySatellite:
    def __init__(self, norad_id, **_kwargs):
        self.norad_id = norad_id


def test_assemble_valid(monkeypatch):
    # monkeypatch do modelo real
    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.assembler.Satellite",
        DummySatellite,
    )

    params = [{"norad_id": 1}, {"norad_id": 2}]

    sats = SatelliteAssembler.assemble(params)

    assert len(sats) == 2
    assert {s.norad_id for s in sats} == {1, 2}


def test_assemble_invalid(monkeypatch):
    def raising_satellite(**_):
        msg = "invalid"
        raise ValueError(msg)

    monkeypatch.setattr(
        "radioskysims_sources.domain.satellite.assembler.Satellite",
        raising_satellite,
    )

    sats = SatelliteAssembler.assemble([{"norad_id": 99}])

    assert sats == []


class DummyTrackFactory:
    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = []

    def build(self, sats):
        self.calls.append(sats)
        return SimpleNamespace(dataset=self.dataset)


class DummySignal:
    def __init__(self, values, frequency, modulation):
        self.psd = dask.delayed(np.asarray(values, dtype=np.float64))
        self.signal = SimpleNamespace(
            frequency=frequency,
            modulation=modulation,
        )


class DummySignalFactory:
    def __init__(self):
        self.fr = None
        self.calls = []

    def frequency_grid_hz(self):
        return np.array([1.0e9, 1.1e9])

    def build(self, sats):
        self.calls.append(sats)
        return {
            1: [
                DummySignal(
                    [1.0, 2.0],
                    frequency=1.05e9,
                    modulation="BPSK",
                )
            ],
            2: [
                DummySignal(
                    [3.0, 4.0],
                    frequency=1.15e9,
                    modulation="QPSK",
                )
            ],
        }


def _tracks_dataset():
    return xr.Dataset(
        coords={
            "satellite": [1, 2],
            "time": [0, 1],
            "names": ("satellite", ["SAT-1", "SAT-2"]),
        }
    )


def test_frequency_coord_converts_hz_to_mhz():
    assembler = SatelliteAssembler(
        signal_factory=DummySignalFactory(),
        track_factory=DummyTrackFactory(_tracks_dataset()),
    )

    freq = assembler._frequency_coord()

    assert freq.dims == ("freq",)
    assert freq.name == "freq"
    np.testing.assert_allclose(freq.values, [1000.0, 1100.0])


def test_build_signals_builds_assembled_dataset():
    tracks = _tracks_dataset()
    signal_factory = DummySignalFactory()
    track_factory = DummyTrackFactory(tracks)

    assembler = SatelliteAssembler(
        signal_factory=signal_factory,
        track_factory=track_factory,
    )

    sats = {"batch": "satellites"}

    result = assembler.build_signals(sats)

    assert track_factory.calls == [sats]
    assert signal_factory.calls == [sats]

    assert result.tracks is tracks
    assert result.name == ["SAT-1", "SAT-2"]

    assert result.signals.sizes["satellite"] == 2
    assert result.signals.sizes["freq"] == 2

    np.testing.assert_allclose(
        result.signals["freq"].values,
        [1000.0, 1100.0],
    )

    np.testing.assert_allclose(
        result.signals["PSD"].compute().values,
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    )

    assert result.signals.attrs["domain"] == "frequency"
    assert result.signals.attrs["units"] == "dB/Hz"


def test_build_signals_recalculate_reuses_existing_tracks():
    tracks = _tracks_dataset()
    signal_factory = DummySignalFactory()
    track_factory = DummyTrackFactory(tracks)

    assembler = SatelliteAssembler(
        signal_factory=signal_factory,
        track_factory=track_factory,
    )

    previous = SimpleNamespace(tracks=tracks)
    new_frequency_range = object()
    sats = {"batch": "satellites"}

    result = assembler.build_signals(
        sats,
        recalculate=True,
        sats_assembled=previous,
        frequency_range=new_frequency_range,
    )

    assert result.tracks is tracks
    assert track_factory.calls == []
    assert signal_factory.fr is new_frequency_range
    assert signal_factory.calls == [sats]


def test_build_psd_blocks_preserves_signal_metadata():
    assembler = SatelliteAssembler(
        signal_factory=DummySignalFactory(),
        track_factory=DummyTrackFactory(_tracks_dataset()),
    )

    signals = DummySignalFactory().build(object())
    sat_ids = np.array([1, 2])
    freq = xr.DataArray([1000.0, 1100.0], dims="freq")

    blocks, fc_values, mod_values = assembler._build_psd_blocks(
        signals,
        sat_ids,
        freq,
    )

    assert len(blocks) == 2
    assert blocks[0].shape == (1, 2)
    assert blocks[1].shape == (1, 2)

    assert fc_values == [[1.05e9], [1.15e9]]
    assert mod_values == [["BPSK"], ["QPSK"]]
