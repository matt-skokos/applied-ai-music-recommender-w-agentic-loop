"""
Covers apply_agent_decision()'s clamping/application logic and
run_agentic_refinement()'s loop control (stability early-stop, hard cap,
graceful stop on a failed decision call). The Gemini call itself is always
mocked -- never hit the real API in the automated suite.
"""

import pytest

import src.agent_loop as agent_loop
from src.recommender import Facet, Song, UserProfile
from src.session import RecommendationSession

_VALID_DECISION = {"facet": "energy", "direction": "up", "magnitude": 0.1, "rationale": "test"}


def _make_song(**overrides) -> Song:
    song = dict(
        id=1, title="Test", artist="A", genre="pop", mood="happy",
        energy=0.5, tempo_bpm=120, valence=0.5, danceability=0.5, acousticness=0.5,
    )
    song.update(overrides)
    return Song(**song)


def _make_session(**overrides) -> RecommendationSession:
    return RecommendationSession(base_user=UserProfile(user_id="u", **overrides))


def test_apply_agent_decision_nudges_continuous_facet_up():
    session = _make_session()
    decision = {"facet": "energy", "direction": "up", "magnitude": 0.2, "rationale": "test"}

    facet = agent_loop.apply_agent_decision(session, decision)

    assert facet == Facet.ENERGY
    assert session.facet_weights.target_energy_delta == pytest.approx(0.2)


def test_apply_agent_decision_down_direction_is_negative():
    session = _make_session()
    decision = {"facet": "valence", "direction": "down", "magnitude": 0.15, "rationale": "test"}

    agent_loop.apply_agent_decision(session, decision)

    assert session.facet_weights.target_valence_delta == pytest.approx(-0.15)


def test_apply_agent_decision_boosts_genre_label():
    session = _make_session()
    decision = {"facet": "genre", "label": "jazz", "direction": "up", "magnitude": 0.1, "rationale": "test"}

    facet = agent_loop.apply_agent_decision(session, decision)

    assert facet == Facet.GENRE
    assert session.facet_weights.genre_boosts["jazz"] == pytest.approx(0.1)


def test_apply_agent_decision_penalizes_mood_label():
    session = _make_session()
    decision = {"facet": "mood", "label": "chill", "direction": "down", "magnitude": 0.1, "rationale": "test"}

    agent_loop.apply_agent_decision(session, decision)

    assert session.facet_weights.mood_boosts["chill"] == pytest.approx(-0.1)


def test_apply_agent_decision_genre_without_label_is_a_noop():
    session = _make_session()
    decision = {"facet": "genre", "label": "", "direction": "up", "magnitude": 0.1, "rationale": "test"}

    result = agent_loop.apply_agent_decision(session, decision)

    assert result is None
    assert session.facet_weights.genre_boosts == {}


def test_apply_agent_decision_clamps_magnitude_above_max():
    session = _make_session()
    decision = {"facet": "energy", "direction": "up", "magnitude": 0.7, "rationale": "test"}

    agent_loop.apply_agent_decision(session, decision)

    assert session.facet_weights.target_energy_delta == pytest.approx(0.3)


def test_apply_agent_decision_clamps_magnitude_below_min():
    session = _make_session()
    decision = {"facet": "energy", "direction": "up", "magnitude": 0.001, "rationale": "test"}

    agent_loop.apply_agent_decision(session, decision)

    assert session.facet_weights.target_energy_delta == pytest.approx(0.05)


def test_run_agentic_refinement_stops_early_once_topk_stabilizes(monkeypatch):
    song_a = _make_song(id=1)
    song_b = _make_song(id=2)
    topk_sequence = [
        [(song_a, 1.0, [])],
        [(song_b, 1.0, [])],
        [(song_b, 1.0, [])],
        [(song_b, 1.0, [])],
        [(song_b, 1.0, [])],
    ]
    calls = {"n": 0}

    def fake_rebuild_pool(session, songs, k=5):
        idx = min(calls["n"], len(topk_sequence) - 1)
        calls["n"] += 1
        return topk_sequence[idx], None

    monkeypatch.setattr(agent_loop, "rebuild_pool", fake_rebuild_pool)
    monkeypatch.setattr(agent_loop, "_get_agent_decision", lambda session, topk: dict(_VALID_DECISION))

    session = _make_session()
    final_topk, log = agent_loop.run_agentic_refinement(
        session, [song_a, song_b], k=1, max_iterations=15, stability_window=3,
    )

    # iterations 1-4 each see a change (or haven't hit the stability window yet) and decide;
    # iteration 5 is the 3rd consecutive unchanged top-k, so it stops before deciding again
    assert len(log) == 4
    assert calls["n"] == 6  # 5 loop iterations + 1 final rebuild_pool for the return value


def test_run_agentic_refinement_runs_to_cap_when_never_stable(monkeypatch):
    song_a = _make_song(id=1)
    song_b = _make_song(id=2)
    calls = {"n": 0}

    def fake_rebuild_pool(session, songs, k=5):
        n = calls["n"]
        calls["n"] += 1
        song = song_a if n % 2 == 0 else song_b
        return [(song, 1.0, [])], None

    monkeypatch.setattr(agent_loop, "rebuild_pool", fake_rebuild_pool)
    monkeypatch.setattr(agent_loop, "_get_agent_decision", lambda session, topk: dict(_VALID_DECISION))

    session = _make_session()
    final_topk, log = agent_loop.run_agentic_refinement(
        session, [song_a, song_b], k=1, max_iterations=5, stability_window=3,
    )

    assert len(log) == 5  # never stabilized, so it ran the full cap
    assert calls["n"] == 6


def test_run_agentic_refinement_stops_gracefully_when_decision_call_fails(monkeypatch):
    song_a = _make_song(id=1)
    monkeypatch.setattr(agent_loop, "rebuild_pool", lambda session, songs, k=5: ([(song_a, 1.0, [])], None))
    decisions = [dict(_VALID_DECISION), dict(_VALID_DECISION), None]

    def fake_decision(session, topk):
        return decisions.pop(0) if decisions else None

    monkeypatch.setattr(agent_loop, "_get_agent_decision", fake_decision)

    session = _make_session()
    final_topk, log = agent_loop.run_agentic_refinement(
        session, [song_a], k=1, max_iterations=15, stability_window=3,
    )

    assert len(log) == 2  # stopped once the 3rd decision call returned None
    assert final_topk == [(song_a, 1.0, [])]
