"""
One-time (re-runnable) catalog enrichment: fetches each song's Spotify
genres via spotify_client.fetch_song_genres() and writes them to
data/spotify_genres_cache.json, so the live app never has to hit the
network per session. Run from within src/: `python3 enrich_songs.py`.
"""

import json
from pathlib import Path

from recommender import load_songs
from spotify_client import fetch_song_genres

SONGS_CSV_PATH = "../data/songs.csv"
CACHE_PATH = Path("../data/spotify_genres_cache.json")


def main() -> None:
    songs = load_songs(SONGS_CSV_PATH)

    print(f"Looking through Spotify data for {len(songs)} songs' genres...\n")

    cache = {}
    hits = fallbacks = errors = 0
    for song in songs:
        genres, status, _detail = fetch_song_genres(song["title"], song["artist"])
        cache[str(song["id"])] = genres

        if status.value == "ok":
            hits += 1
        elif status.value == "fallback":
            fallbacks += 1
        else:
            errors += 1
        print(f"  {song['title']} ({song['artist']}): {genres or '-- no genres found'}")

    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"\nDone -- {hits} enriched, {fallbacks} skipped (no match/no genres), {errors} errors.")
    print(f"Wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
