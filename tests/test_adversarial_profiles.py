"""
Adversarial / edge-case user profiles that expose gaps in the current
scoring logic (src/recommender.py). These are characterization tests: they
assert what the recommender *actually does today*, so each failure mode is
reproducible evidence for the model card's Limitations section rather than
a bug fix.
"""

from pathlib import Path

import pytest

from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    score_song,
    recommend_songs,
    W_GENRE,
    W_MOOD,
)

SONGS_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def test_gap_interaction_history_is_never_scored():
    """
    UserProfile.skips/likes/playlist_adds/play_counts are stored but
    _score_song_for_user() never reads them, so a song the user has
    explicitly skipped 5 times (and never liked) can still be the #1
    recommendation if its content features match the taste vector.
    """
    matched = Song(
        id=1, title="Perfect Match", artist="A", genre="pop", mood="happy",
        energy=0.8, tempo_bpm=120, valence=0.8, danceability=0.8, acousticness=0.2,
    )
    mismatched = Song(
        id=2, title="Total Mismatch", artist="B", genre="rock", mood="sad",
        energy=0.1, tempo_bpm=60, valence=0.1, danceability=0.1, acousticness=0.9,
    )

    user = UserProfile(
        user_id="skips_it_anyway",
        favorite_genres={"pop"},
        favorite_moods={"happy"},
        target_energy=0.8, target_valence=0.8, target_danceability=0.8,
        target_acousticness=0.2, target_tempo=0.5,
        skips={1: 5},
        likes=set(),
    )

    rec = Recommender([matched, mismatched])
    results = rec.recommend(user, k=1)

    assert results[0].id == 1  # recommended #1 despite 5 recorded skips and zero likes


def test_gap_none_value_silently_shadows_favorite_genres():
    """
    score_song() does user_prefs.get("genre", user_prefs.get("favorite_genres")).
    If a caller passes {"genre": None, "favorite_genres": [...]}, the "genre"
    key exists (with value None), so .get returns None instead of falling
    back to favorite_genres -- the user's stated preference is silently
    dropped rather than raising or defaulting.
    """
    base = dict(energy=0.5, tempo=0.5, valence=0.5, danceability=0.5, acousticness=0.5)
    song = dict(genre="pop", mood="happy", energy=0.5, tempo_bpm=120.0,
                valence=0.5, danceability=0.5, acousticness=0.5)

    prefs_shadowed = {**base, "genre": None, "favorite_genres": ["pop"],
                      "mood": None, "favorite_moods": ["happy"]}
    prefs_intended = {**base, "favorite_genres": ["pop"], "favorite_moods": ["happy"]}

    score_shadowed, reasons_shadowed = score_song(prefs_shadowed, song)
    score_intended, reasons_intended = score_song(prefs_intended, song)

    assert score_intended - score_shadowed == pytest.approx(W_GENRE + W_MOOD)
    assert not any("favorite genre" in r for r in reasons_shadowed)
    assert any("favorite genre" in r for r in reasons_intended)


def test_gap_unnormalized_tempo_breaks_score_bounds():
    """
    tempo_bpm gets normalized to 0-1 before comparison, but a user-supplied
    target_tempo does not. A plausible input mistake -- entering a target
    tempo as raw BPM (e.g. 128) instead of the expected 0.00-1.00 scale --
    is silently accepted and produces a wildly negative score, breaking the
    0.0-1.0 bound every other test assumes.
    """
    prefs = dict(genre="house", mood="energetic", energy=0.9, tempo=128,
                 valence=0.5, danceability=0.5, acousticness=0.5)
    song = dict(genre="house", mood="energetic", energy=0.9, tempo_bpm=128.0,
                valence=0.5, danceability=0.5, acousticness=0.5)

    score, _ = score_song(prefs, song)

    assert score < 0.0


def test_gap_genre_and_mood_matching_is_case_sensitive():
    """
    genre_match/mood_match use exact string membership, so a user who types
    "Pop"/"Happy" instead of "pop"/"happy" gets no categorical credit at all,
    even against a song that's an obvious match to a human reader.
    """
    song = dict(genre="pop", mood="happy", energy=0.5, tempo_bpm=120.0,
                valence=0.5, danceability=0.5, acousticness=0.5)
    prefs_lower = dict(genre="pop", mood="happy", energy=0.5, tempo=0.5,
                        valence=0.5, danceability=0.5, acousticness=0.5)
    prefs_capitalized = {**prefs_lower, "genre": "Pop", "mood": "Happy"}

    score_lower, _ = score_song(prefs_lower, song)
    score_capitalized, _ = score_song(prefs_capitalized, song)

    assert score_lower - score_capitalized == pytest.approx(W_GENRE + W_MOOD)


def test_gap_contradictory_mood_and_energy_still_scores_perfectly():
    """
    mood is a free categorical label with no relationship to the continuous
    energy/tempo features, so nothing stops a "chill" song from also being
    fast and high-energy. A user profile with mood="chill" but a
    high-energy, fast-tempo target scores this contradictory song 1.0 and
    confidently explains it as a mood match.
    """
    prefs = dict(genre="rock", mood="chill", energy=0.95, tempo=0.9,
                 valence=0.5, danceability=0.5, acousticness=0.5)
    # tempo_bpm=168 normalizes to (168-60)/120 = 0.9, matching target_tempo
    song = dict(genre="rock", mood="chill", energy=0.95, tempo_bpm=168.0,
                valence=0.5, danceability=0.5, acousticness=0.5)

    score, reasons = score_song(prefs, song)

    assert score == pytest.approx(1.0)
    assert any("favorite mood (chill)" in r for r in reasons)


def test_gap_no_artist_diversity_control():
    """
    Songs are scored independently with no penalty for repeated artists, so
    a taste profile that matches one artist's style closely can return a
    top-3 dominated by that single artist (Wildgrass Trio supplies both
    folk tracks in the catalog).
    """
    songs = load_songs(str(SONGS_CSV_PATH))
    prefs = dict(genre="folk", mood="nostalgic", energy=0.38, tempo=0.25,
                 valence=0.58, danceability=0.47, acousticness=0.85)

    recommendations = recommend_songs(prefs, songs, k=3)
    artists = [song["artist"] for song, _, _ in recommendations]

    assert artists.count("Wildgrass Trio") >= 2


def test_gap_liking_every_genre_erases_genres_discriminative_power():
    """
    genre_match is the single highest-weighted term (W_GENRE=0.30), but it's
    a binary flag. A profile that lists every genre in the catalog as a
    favorite (a "superfan" or careless-onboarding user) makes genre_match
    true for every song, so the model's top-priority feature stops
    differentiating between songs at all.
    """
    songs = load_songs(str(SONGS_CSV_PATH))
    all_genres = list({s["genre"] for s in songs})
    prefs = dict(favorite_genres=all_genres, energy=0.5, tempo=0.5,
                 valence=0.5, danceability=0.5, acousticness=0.5)

    recommendations = recommend_songs(prefs, songs, k=len(songs))

    assert all(
        any("favorite genre" in r for r in explanation.split("; "))
        for _, _, explanation in recommendations
    )
