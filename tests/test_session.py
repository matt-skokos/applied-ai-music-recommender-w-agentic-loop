"""
Covers the fix for the reasons/facet mismatch flagged in
assets/design_spec.md: interpret_feedback() must nudge the facet that
_score_song_for_user() actually tagged, not a facet re-derived from
reason text.
"""

import pytest

from src.recommender import Facet, Song, UserProfile, _score_song_for_user
from src.session import FeedbackType, LogStatus, LogStep, RecommendationSession, interpret_feedback


def _make_song(**overrides) -> Song:
    song = dict(
        id=1, title="Test", artist="A", genre="pop", mood="happy",
        energy=0.8, tempo_bpm=120, valence=0.8, danceability=0.8, acousticness=0.2,
    )
    song.update(overrides)
    return Song(**song)


def test_interpret_feedback_up_boosts_matched_genre_and_mood():
    song = _make_song()
    user = UserProfile(user_id="u", favorite_genres={"pop"}, favorite_moods={"happy"})
    _, reasons = _score_song_for_user(user, song)
    session = RecommendationSession(base_user=user)

    log = interpret_feedback(session, song, reasons, FeedbackType.UP)

    assert session.facet_weights.genre_boosts["pop"] == pytest.approx(0.15)
    assert session.facet_weights.mood_boosts["happy"] == pytest.approx(0.15)
    assert song.id not in session.excluded_song_ids
    assert log.step == LogStep.INTERPRET
    assert log.status == LogStatus.OK
    assert Facet.GENRE.value in log.detail["facets_touched"]
    assert Facet.MOOD.value in log.detail["facets_touched"]


def test_interpret_feedback_down_excludes_song_and_penalizes_facets():
    song = _make_song()
    user = UserProfile(user_id="u", favorite_genres={"pop"}, favorite_moods={"happy"})
    _, reasons = _score_song_for_user(user, song)
    session = RecommendationSession(base_user=user)

    interpret_feedback(session, song, reasons, FeedbackType.DOWN)

    assert session.facet_weights.genre_boosts["pop"] == pytest.approx(-0.15)
    assert song.id in session.excluded_song_ids


def test_interpret_feedback_records_feedback_event_with_matched_facets():
    song = _make_song()
    user = UserProfile(user_id="u", favorite_genres={"pop"}, favorite_moods={"happy"})
    _, reasons = _score_song_for_user(user, song)
    session = RecommendationSession(base_user=user)

    interpret_feedback(session, song, reasons, FeedbackType.UP)

    assert len(session.feedback_history) == 1
    event = session.feedback_history[0]
    assert event.song_id == song.id
    assert event.feedback == FeedbackType.UP
    assert Facet.GENRE in event.matched_facets


def test_interpret_feedback_ignores_catchall_reason_with_no_facet():
    # every continuous similarity held below the 0.85 reason threshold, and
    # genre/mood both mismatched, so _score() emits only the None-facet catch-all
    song = _make_song(
        genre="rock", mood="sad",
        energy=0.7, tempo_bpm=144.0, valence=0.7, danceability=0.7, acousticness=0.7,
    )
    user = UserProfile(
        user_id="u", favorite_genres={"pop"}, favorite_moods={"happy"},
        target_energy=0.5, target_tempo=0.5, target_valence=0.5,
        target_danceability=0.5, target_acousticness=0.5,
    )
    _, reasons = _score_song_for_user(user, song)
    assert reasons == [("a reasonable overall match to your taste profile", None)]
    session = RecommendationSession(base_user=user)

    log = interpret_feedback(session, song, reasons, FeedbackType.UP)

    assert log.detail["facets_touched"] == []
    assert session.facet_weights.heaviest(5) == []
