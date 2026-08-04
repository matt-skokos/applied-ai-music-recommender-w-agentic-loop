"""
Covers the search -> artist genres -> skip chain in spotify_client.py.
Network calls are always mocked; no real Spotify credentials needed.
"""

import json

import pytest

import src.spotify_client as spotify_client
from src.recommender import Song
from src.session import LogStatus


def _make_song(**overrides) -> Song:
    song = dict(
        id=1, title="Test", artist="A", genre="pop", mood="happy",
        energy=0.8, tempo_bpm=120, valence=0.8, danceability=0.8, acousticness=0.2,
    )
    song.update(overrides)
    return Song(**song)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_token_cache():
    spotify_client._access_token = None
    spotify_client._token_expires_at = 0.0
    yield
    spotify_client._access_token = None
    spotify_client._token_expires_at = 0.0


def test_get_access_token_fetches_and_caches(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    calls = []
    monkeypatch.setattr(
        spotify_client.requests, "post",
        lambda *a, **kw: calls.append(1) or _FakeResponse({"access_token": "tok123", "expires_in": 3600}),
    )

    first = spotify_client._get_access_token()
    second = spotify_client._get_access_token()

    assert first == second == "tok123"
    assert len(calls) == 1  # second call served from cache, no second POST


def test_get_access_token_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

    with pytest.raises(RuntimeError):
        spotify_client._get_access_token()


def test_search_track_returns_ids(monkeypatch):
    monkeypatch.setattr(spotify_client, "_get_access_token", lambda: "tok")
    payload = {"tracks": {"items": [{"id": "track1", "artists": [{"id": "artist1"}]}]}}
    monkeypatch.setattr(spotify_client.requests, "get", lambda *a, **kw: _FakeResponse(payload))

    hit = spotify_client._search_track("Song", "Artist")

    assert hit == {"track_id": "track1", "artist_id": "artist1"}


def test_search_track_returns_none_when_no_match(monkeypatch):
    monkeypatch.setattr(spotify_client, "_get_access_token", lambda: "tok")
    monkeypatch.setattr(spotify_client.requests, "get", lambda *a, **kw: _FakeResponse({"tracks": {"items": []}}))

    assert spotify_client._search_track("Song", "Artist") is None


def test_artist_genres_returns_list(monkeypatch):
    monkeypatch.setattr(spotify_client, "_get_access_token", lambda: "tok")
    monkeypatch.setattr(
        spotify_client.requests, "get",
        lambda *a, **kw: _FakeResponse({"genres": ["dream pop", "chillwave"]}),
    )

    assert spotify_client._artist_genres("artist1") == ["dream pop", "chillwave"]


def test_fetch_song_genres_ok_when_artist_has_genres(monkeypatch):
    monkeypatch.setattr(spotify_client, "_search_track", lambda title, artist: {"track_id": "t1", "artist_id": "a1"})
    monkeypatch.setattr(spotify_client, "_artist_genres", lambda artist_id: ["lofi", "chillhop"])

    genres, status, detail = spotify_client.fetch_song_genres("Song", "Artist")

    assert genres == ["lofi", "chillhop"]
    assert status == LogStatus.OK
    assert detail["tier"] == "artist"


def test_fetch_song_genres_fallback_when_no_search_match(monkeypatch):
    monkeypatch.setattr(spotify_client, "_search_track", lambda title, artist: None)

    genres, status, detail = spotify_client.fetch_song_genres("Song", "Artist")

    assert genres == []
    assert status == LogStatus.FALLBACK
    assert detail["reason"] == "no_search_match"


def test_fetch_song_genres_fallback_when_artist_has_no_genres(monkeypatch):
    monkeypatch.setattr(spotify_client, "_search_track", lambda title, artist: {"track_id": "t1", "artist_id": "a1"})
    monkeypatch.setattr(spotify_client, "_artist_genres", lambda artist_id: [])

    genres, status, detail = spotify_client.fetch_song_genres("Song", "Artist")

    assert genres == []
    assert status == LogStatus.FALLBACK
    assert detail["reason"] == "artist_has_no_genres"


def test_fetch_song_genres_error_on_exception(monkeypatch):
    def _boom(title, artist):
        raise RuntimeError("network timeout")

    monkeypatch.setattr(spotify_client, "_search_track", _boom)

    genres, status, detail = spotify_client.fetch_song_genres("Song", "Artist")

    assert genres == []
    assert status == LogStatus.ERROR
    assert detail["reason"] == "network timeout"


def test_apply_genre_cache_merges_matching_entries(tmp_path):
    cache_path = tmp_path / "genres.json"
    cache_path.write_text(json.dumps({"1": ["dream pop", "chillwave"], "2": []}))
    song_with_genres = _make_song(id=1)
    song_without_match = _make_song(id=99)  # not present in the cache

    result = spotify_client.apply_genre_cache([song_with_genres, song_without_match], str(cache_path))

    assert result[0].spotify_genres == ["dream pop", "chillwave"]
    assert result[1].spotify_genres == []
    assert song_with_genres.spotify_genres == []  # original left untouched


def test_apply_genre_cache_missing_file_is_a_noop(tmp_path):
    song = _make_song(id=1)
    missing_path = tmp_path / "does_not_exist.json"

    result = spotify_client.apply_genre_cache([song], str(missing_path))

    assert result == [song]
