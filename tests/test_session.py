"""
Covers the fix for the reasons/facet mismatch flagged in
assets/design_spec.md: interpret_feedback() must nudge the facet that
_score_song_for_user() actually tagged, not a facet re-derived from
reason text.
"""

import pytest

from src.recommender import Facet, Song, UserProfile, _score_song_for_user
from src.session import (
    FacetWeights,
    FeedbackType,
    LogStatus,
    LogStep,
    RecommendationSession,
    _apply_facet_weights,
    interpret_feedback,
    rebuild_pool,
)


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


def test_apply_facet_weights_shifts_and_clamps_continuous_targets():
    user = UserProfile(user_id="u", target_energy=0.5, target_valence=0.9)
    weights = FacetWeights(target_energy_delta=0.2, target_valence_delta=0.5)

    effective = _apply_facet_weights(user, weights)

    assert effective.target_energy == pytest.approx(0.7)
    assert effective.target_valence == pytest.approx(1.0)  # 0.9 + 0.5 clamped to 1.0
    assert user.target_energy == pytest.approx(0.5)  # base_user left untouched


def test_rebuild_pool_filters_excluded_songs():
    kept = _make_song(id=1, genre="pop", mood="happy")
    excluded = _make_song(id=2, genre="pop", mood="happy")
    user = UserProfile(user_id="u", favorite_genres={"pop"}, favorite_moods={"happy"})
    session = RecommendationSession(base_user=user)
    session.excluded_song_ids.add(excluded.id)

    results, log = rebuild_pool(session, [kept, excluded], k=5)

    assert [song.id for song, _, _ in results] == [kept.id]
    assert log.step == LogStep.REBUILD_POOL
    assert log.status == LogStatus.OK
    assert log.detail == {"candidate_count": 1, "excluded_count": 1, "served": 1}


def test_rebuild_pool_continuous_delta_reranks_songs():
    low_energy_song = _make_song(id=1, genre="rock", mood="sad", energy=0.1)
    high_energy_song = _make_song(id=2, genre="rock", mood="sad", energy=0.9)
    user = UserProfile(user_id="u", target_energy=0.1)
    session = RecommendationSession(base_user=user)

    before, _ = rebuild_pool(session, [low_energy_song, high_energy_song], k=1)
    assert before[0][0].id == low_energy_song.id  # target_energy=0.1 favors the low-energy song

    session.facet_weights.target_energy_delta = 0.8  # shifts effective target to 0.9
    after, _ = rebuild_pool(session, [low_energy_song, high_energy_song], k=1)
    assert after[0][0].id == high_energy_song.id


def test_rebuild_pool_adds_genre_and_mood_boosts_to_score():
    song = _make_song(id=1, genre="lofi", mood="chill")
    user = UserProfile(user_id="u")
    session = RecommendationSession(base_user=user)

    baseline, _ = rebuild_pool(session, [song], k=1)
    baseline_score = baseline[0][1]

    session.facet_weights.genre_boosts["lofi"] = 0.3
    session.facet_weights.mood_boosts["chill"] = 0.15
    boosted, _ = rebuild_pool(session, [song], k=1)

    assert boosted[0][1] == pytest.approx(baseline_score + 0.3 + 0.15)


def test_rebuild_pool_preserves_facet_tagged_reasons():
    song = _make_song(id=1, genre="pop", mood="happy")
    user = UserProfile(user_id="u", favorite_genres={"pop"}, favorite_moods={"happy"})
    session = RecommendationSession(base_user=user)

    results, _ = rebuild_pool(session, [song], k=1)

    _, _, reasons = results[0]
    assert (f"matches your favorite genre (pop)", Facet.GENRE) in reasons
