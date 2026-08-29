# tests/test_space_track_client.py
from __future__ import annotations

import httpx
import pandas as pd
import pytest

from radioskysims_sources.infrastructure.external.satellite_downloader import SpaceTrackClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class DummyAsyncResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        # Síncrono: o código de produção chama r.json() sem await.
        return self._data

    def raise_for_status(self):
        return None


class DummySyncResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def clear_registry():
    SpaceTrackClient.clear_registry()
    yield
    SpaceTrackClient.clear_registry()


@pytest.fixture(autouse=True)
def set_spacetrack_env(monkeypatch):
    monkeypatch.setenv("ST_USER", "dummy_user")
    monkeypatch.setenv("ST_PWD", "dummy_pwd")


async def test_async_and_sync_share_registry(monkeypatch):
    # prepare fake async client
    async def fake_post(_self, _url, data=None):
        _ = data
        return DummyAsyncResponse({"ok": True})

    async def fake_get(_self, _url):
        # return a list of dicts with norad_cat_id
        return DummyAsyncResponse([{"NORAD_CAT_ID": 1, "EPOCH": "2020-01-01"}])

    class FakeAsyncClient:
        def __init__(self, timeout=60):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, data=None):
            return await fake_post(self, url, data=data)

        async def get(self, url):
            return await fake_get(self, url)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    # prepare fake sync client
    def fake_sync_post(_self, _url, data=None):
        _ = data
        return DummySyncResponse({"ok": True})

    def fake_sync_get(_self, _url):
        return DummySyncResponse([{"NORAD_CAT_ID": 2, "EPOCH": "2020-01-01"}])

    class FakeSyncClient:
        def __init__(self, timeout=60):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, data=None):
            return fake_sync_post(self, url, data=data)

        def get(self, url):
            return fake_sync_get(self, url)

    monkeypatch.setattr(httpx, "Client", FakeSyncClient)

    client1 = SpaceTrackClient()
    df_async = await client1.fetch_elements([1, 3, 4])  # will fetch 1
    assert isinstance(df_async, pd.DataFrame)
    assert (
        "norad_cat_id" in df_async.columns or "norad_id" in df_async.columns or not df_async.empty
    )

    # Now sync should not fetch 1 again; but will fetch 2
    client2 = SpaceTrackClient()
    df_sync = client2.fetch_elements_sync([1, 2])
    assert isinstance(df_sync, pd.DataFrame)

    # check that registry contains both 1 and 2
    # we can't access private set directly per se, but repeated fetch returns empty
    df_sync_repeat = client2.fetch_elements_sync([1, 2])
    assert df_sync_repeat.empty

    df_async_repeat = await client1.fetch_elements([1, 2])
    assert df_async_repeat.empty


async def test_filter_new_async_and_sync():
    # ensure filter_new works as intended
    SpaceTrackClient.clear_registry()
    c = SpaceTrackClient()
    new = await c.filter_new_async([10, 11, 12])
    assert set(new) == {10, 11, 12}

    # second call in async should return empty
    new2 = await c.filter_new_async([10, 13])
    assert set(new2) == {13}

    # sync filter should see the same registry
    new_sync = SpaceTrackClient.filter_new_sync([10, 11, 14])
    assert set(new_sync) == {14}
