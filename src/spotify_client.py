"""
Spotify genre enrichment: song/artist as the lookup roots, feeding a
richer multi-genre list into scoring (assets/design_spec.md's facet
enrichment section). Continuous audio features (energy/tempo/valence/etc.)
are not available via Spotify's API for a new developer app, so this only
ever enriches genres.

Chain: search title+artist -> artist genres -> skip. No album-level
fallback tier: Spotify's album `genres` field is documented as always
empty, so it can never actually serve as a fallback.
"""

import base64
import os
import time
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    from session import LogStatus
except ImportError:
    from src.session import LogStatus

load_dotenv()

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"
_REQUEST_TIMEOUT = 10

_access_token: Optional[str] = None
_token_expires_at: float = 0.0


def _get_access_token() -> str:
    """Fetches and caches an app-only access token via the Client Credentials flow, refreshing it once it's close to expiry."""
    global _access_token, _token_expires_at
    if _access_token and time.time() < _token_expires_at:
        return _access_token

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET are not set")

    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = requests.post(
        _TOKEN_URL,
        headers={"Authorization": f"Basic {basic_auth}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    _access_token = payload["access_token"]
    _token_expires_at = time.time() + payload["expires_in"] - 30
    return _access_token


def _search_track(title: str, artist: str) -> Optional[dict]:
    """Looks up a track by title+artist and returns its track/artist ids, or None if nothing matched."""
    response = requests.get(
        f"{_API_BASE}/search",
        headers={"Authorization": f"Bearer {_get_access_token()}"},
        params={"q": f"track:{title} artist:{artist}", "type": "track", "limit": 1},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    items = response.json().get("tracks", {}).get("items", [])
    if not items:
        return None

    artists = items[0].get("artists") or []
    return {
        "track_id": items[0]["id"],
        "artist_id": artists[0]["id"] if artists else None,
    }


def _artist_genres(artist_id: str) -> List[str]:
    """Returns the given artist's Spotify genre tags -- often empty, since Spotify doesn't classify every artist."""
    response = requests.get(
        f"{_API_BASE}/artists/{artist_id}",
        headers={"Authorization": f"Bearer {_get_access_token()}"},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("genres", [])


def fetch_song_genres(title: str, artist: str) -> Tuple[List[str], LogStatus, dict]:
    """Runs the search-then-artist-genres chain for one song, returning its genres plus a status/detail pair the caller can log."""
    try:
        hit = _search_track(title, artist)
        if hit is None or hit["artist_id"] is None:
            return [], LogStatus.FALLBACK, {"tier": "skip", "reason": "no_search_match"}

        genres = _artist_genres(hit["artist_id"])
        if genres:
            return genres, LogStatus.OK, {"tier": "artist", "genres": genres}
        return [], LogStatus.FALLBACK, {"tier": "skip", "reason": "artist_has_no_genres"}
    except Exception as exc:
        return [], LogStatus.ERROR, {"tier": "skip", "reason": str(exc)}
