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

# 10 distinct user preference dicts covering different genre/mood combos from
# songs.csv, a range of energy levels, and a few edge cases: list-based
# favorites (#9), a missing mood key (#9), and no categorical match at all (#10).
USER_PROFILES = [
    {"genre": "pop", "mood": "happy", "energy": 0.85},
    {"genre": "lofi", "mood": "chill", "energy": 0.35, "tempo": 0.15, "valence": 0.55, "danceability": 0.55, "acousticness": 0.8},
    {"genre": "rock", "mood": "intense", "energy": 0.9, "tempo": 0.9},
    {"genre": "ambient", "mood": "chill", "energy": 0.25, "acousticness": 0.9},
    {"genre": "jazz", "mood": "relaxed", "energy": 0.35, "danceability": 0.5},
    {"genre": "house", "mood": "energetic", "energy": 0.88, "danceability": 0.9},
    {"genre": "hip-hop", "mood": "confident", "energy": 0.8, "danceability": 0.85},
    {"genre": "classical", "mood": "dreamy", "energy": 0.3, "acousticness": 0.95},
    {"favorite_genres": ["folk", "r&b"], "energy": 0.4},
    {"energy": 0.6, "tempo": 0.5, "valence": 0.6, "danceability": 0.6, "acousticness": 0.4},
]

def _neutral_song(**overrides) -> dict:
    # tempo_bpm=120 normalizes to 0.5 (MIN_BPM=60, MAX_BPM=180), matching the
    # default 0.5 target below, so every similarity term starts at a known 1.0.
    song = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.5,
        "tempo_bpm": 120.0,
        "valence": 0.5,
        "danceability": 0.5,
        "acousticness": 0.5,
    }
    song.update(overrides)
    return song

def _neutral_prefs(**overrides) -> dict:
    prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.5,
        "tempo": 0.5,
        "valence": 0.5,
        "danceability": 0.5,
        "acousticness": 0.5,
    }
    prefs.update(overrides)
    return prefs

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        user_id="test_user",
        favorite_genres={"pop"},
        favorite_moods={"happy"},
        target_energy=0.8,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        user_id="test_user",
        favorite_genres={"pop"},
        favorite_moods={"happy"},
        target_energy=0.8,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_load_songs_returns_expected_number_of_songs():
    songs = load_songs(str(SONGS_CSV_PATH))

    assert len(songs) == 20
    assert songs[0]["title"] == "Sunrise City"
    assert isinstance(songs[0]["energy"], float)


def test_score_song_perfect_match_scores_one():
    score, reasons = score_song(_neutral_prefs(), _neutral_song())

    assert score == pytest.approx(1.0)
    assert len(reasons) > 0


def test_score_song_full_mismatch_scores_zero():
    prefs = _neutral_prefs(genre="rock", mood="sad", energy=0.0, tempo=0.0, valence=0.0, danceability=0.0, acousticness=0.0)
    song = _neutral_song(energy=1.0, tempo_bpm=180.0, valence=1.0, danceability=1.0, acousticness=1.0)

    score, _ = score_song(prefs, song)

    assert score == pytest.approx(0.0)


def test_genre_match_contributes_exactly_its_weight():
    # mood is held mismatched in both cases so only the genre term can move the score
    prefs = _neutral_prefs(genre="pop", mood="happy")
    song_genre_match = _neutral_song(genre="pop", mood="chill")
    song_genre_mismatch = _neutral_song(genre="rock", mood="chill")

    score_match, _ = score_song(prefs, song_genre_match)
    score_mismatch, _ = score_song(prefs, song_genre_mismatch)

    assert score_match - score_mismatch == pytest.approx(W_GENRE)


def test_mood_match_contributes_exactly_its_weight():
    # genre is held mismatched in both cases so only the mood term can move the score
    prefs = _neutral_prefs(genre="pop", mood="happy")
    song_mood_match = _neutral_song(genre="rock", mood="happy")
    song_mood_mismatch = _neutral_song(genre="rock", mood="chill")

    score_match, _ = score_song(prefs, song_mood_match)
    score_mismatch, _ = score_song(prefs, song_mood_mismatch)

    assert score_match - score_mismatch == pytest.approx(W_MOOD)


def test_genre_weight_outweighs_mood_weight():
    # confirms the requested priority ordering: genre match > mood match
    assert W_GENRE > W_MOOD


def test_tempo_below_min_bpm_clamps_instead_of_exceeding_score_bounds():
    prefs = _neutral_prefs(tempo=1.0)
    song = _neutral_song(tempo_bpm=20.0)  # well below MIN_BPM=60, would go negative unclamped

    score, _ = score_song(prefs, song)

    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("user_prefs", USER_PROFILES)
def test_recommend_songs_handles_diverse_user_profiles(user_prefs):
    songs = load_songs(str(SONGS_CSV_PATH))
    recommendations = recommend_songs(user_prefs, songs, k=5)

    assert len(recommendations) == 5

    scores = [score for _, score, _ in recommendations]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert all(explanation.strip() for _, _, explanation in recommendations)
