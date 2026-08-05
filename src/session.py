"""
Per-session state and logging for the agentic feedback loop (see
assets/design_spec.md). Facet weights are per-query only -- nothing here
persists across sessions.
"""

import heapq
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# recommender.py has no cross-module imports of its own yet, so there's no
# precedent in this repo for which import style wins -- main.py/compare_profiles.py
# run flat (cwd=src/), tests import via the src.* package. Support both.
try:
    from recommender import Facet, Reason, Song, UserProfile, _normalize_tempo, _score_song_for_user
except ImportError:
    from src.recommender import Facet, Reason, Song, UserProfile, _normalize_tempo, _score_song_for_user


class FeedbackType(str, Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class FacetWeights:
    # continuous facets: additive deltas applied to the user's base targets
    target_energy_delta: float = 0.0
    target_tempo_delta: float = 0.0
    target_valence_delta: float = 0.0
    target_danceability_delta: float = 0.0
    target_acousticness_delta: float = 0.0

    # categorical facets: per-label weight, +boost / -penalty
    genre_boosts: Dict[str, float] = field(default_factory=dict)
    mood_boosts: Dict[str, float] = field(default_factory=dict)

    def heaviest(self, n: int = 2) -> List[tuple]:
        """Top-n facets by absolute weight -- feeds the transparency message."""
        continuous = {
            "energy": self.target_energy_delta,
            "tempo": self.target_tempo_delta,
            "valence": self.target_valence_delta,
            "danceability": self.target_danceability_delta,
            "acousticness": self.target_acousticness_delta,
        }
        combined = {**continuous, **self.genre_boosts, **self.mood_boosts}
        touched = {k: v for k, v in combined.items() if v != 0.0}
        return sorted(touched.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]


@dataclass
class FeedbackEvent:
    song_id: int
    feedback: FeedbackType
    matched_facets: List[Facet]
    round_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RecommendationSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    base_user: Optional[UserProfile] = None
    facet_weights: FacetWeights = field(default_factory=FacetWeights)
    excluded_song_ids: Set[int] = field(default_factory=set)
    feedback_history: List[FeedbackEvent] = field(default_factory=list)
    round_number: int = 1
    max_rounds: int = 5


class LogStep(str, Enum):
    RETRIEVE_FACETS = "retrieve_facets"
    BUILD_TOPK = "build_topk"
    COLLECT_FEEDBACK = "collect_feedback"
    INTERPRET = "interpret"
    TRANSPARENCY_MESSAGE = "transparency_message"
    REBUILD_POOL = "rebuild_pool"
    RELAX_CONSTRAINT = "relax_constraint"
    OOPS_NO_RESULTS = "oops_no_results"
    PRESENT_FINAL = "present_final"


class LogStatus(str, Enum):
    OK = "ok"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass
class LogEntry:
    session_id: str
    round_number: int
    step: LogStep
    status: LogStatus
    detail: dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


_FACET_STEP = 0.15

_CONTINUOUS_FACETS = (Facet.ENERGY, Facet.TEMPO, Facet.VALENCE, Facet.DANCEABILITY, Facet.ACOUSTICNESS)


def _song_value_for_facet(song: Song, facet: Facet) -> float:
    """Reads a song's own value for a continuous facet, normalizing tempo_bpm to the same 0-1 scale as the rest."""
    if facet == Facet.TEMPO:
        return _normalize_tempo(song.tempo_bpm)
    return getattr(song, facet.value)


def _effective_target_for_facet(base_user: UserProfile, weights: FacetWeights, facet: Facet) -> float:
    """Returns the session's current target for a continuous facet: the base user's value plus its accumulated delta."""
    attr = f"target_{facet.value}"
    return getattr(base_user, attr) + getattr(weights, f"{attr}_delta")


def interpret_feedback(
    session: RecommendationSession,
    song: Song,
    reasons: List[Reason],
    feedback: FeedbackType,
) -> LogEntry:
    """
    Nudges session.facet_weights from this song's feedback: every continuous
    facet's target always moves toward (or away from) the song's own value,
    proportional to how far apart they are -- gating this on whether the
    facet already counted as "close" meant a song whose own features never
    crossed the reason threshold could never be pulled toward, even after
    repeated positive feedback. Genre/mood boosts stay tied to whether
    _score_song_for_user() actually tagged a genre/mood match.
    """
    direction = 1.0 if feedback == FeedbackType.UP else -1.0
    touched: List[Facet] = []

    for facet in _CONTINUOUS_FACETS:
        song_value = _song_value_for_facet(song, facet)
        effective_target = _effective_target_for_facet(session.base_user, session.facet_weights, facet)
        attr = f"target_{facet.value}_delta"
        setattr(
            session.facet_weights, attr,
            getattr(session.facet_weights, attr) + direction * _FACET_STEP * (song_value - effective_target),
        )
        touched.append(facet)

    for _text, facet in reasons:
        if facet == Facet.GENRE:
            session.facet_weights.genre_boosts[song.genre] = (
                session.facet_weights.genre_boosts.get(song.genre, 0.0) + direction * _FACET_STEP
            )
            touched.append(facet)
        elif facet == Facet.MOOD:
            session.facet_weights.mood_boosts[song.mood] = (
                session.facet_weights.mood_boosts.get(song.mood, 0.0) + direction * _FACET_STEP
            )
            touched.append(facet)

    if feedback == FeedbackType.DOWN:
        session.excluded_song_ids.add(song.id)

    session.feedback_history.append(
        FeedbackEvent(
            song_id=song.id,
            feedback=feedback,
            matched_facets=touched,
            round_number=session.round_number,
        )
    )

    return LogEntry(
        session_id=session.session_id,
        round_number=session.round_number,
        step=LogStep.INTERPRET,
        status=LogStatus.OK,
        detail={"song_id": song.id, "feedback": feedback.value, "facets_touched": [f.value for f in touched]},
    )


def _apply_facet_weights(base_user: UserProfile, weights: FacetWeights) -> UserProfile:
    """Returns a copy of base_user with each continuous target shifted by its delta and clamped to [0.0, 1.0]."""
    def _shifted(value: float, delta: float) -> float:
        return max(0.0, min(1.0, value + delta))

    return replace(
        base_user,
        target_energy=_shifted(base_user.target_energy, weights.target_energy_delta),
        target_tempo=_shifted(base_user.target_tempo, weights.target_tempo_delta),
        target_valence=_shifted(base_user.target_valence, weights.target_valence_delta),
        target_danceability=_shifted(base_user.target_danceability, weights.target_danceability_delta),
        target_acousticness=_shifted(base_user.target_acousticness, weights.target_acousticness_delta),
    )


def rebuild_pool(
    session: RecommendationSession,
    songs: List[Song],
    k: int = 5,
) -> Tuple[List[Tuple[Song, float, List[Reason]]], LogEntry]:
    """Re-scores non-excluded songs against the session's shifted targets plus genre/mood boosts, returning the next round's top-k with reasons."""
    candidates = [song for song in songs if song.id not in session.excluded_song_ids]
    effective_user = _apply_facet_weights(session.base_user, session.facet_weights)

    scored = []
    for song in candidates:
        score, reasons = _score_song_for_user(effective_user, song)
        score += session.facet_weights.genre_boosts.get(song.genre, 0.0)
        score += session.facet_weights.mood_boosts.get(song.mood, 0.0)
        scored.append((song, score, reasons))

    top_k = heapq.nlargest(k, scored, key=lambda item: item[1])

    log = LogEntry(
        session_id=session.session_id,
        round_number=session.round_number,
        step=LogStep.REBUILD_POOL,
        status=LogStatus.OK,
        detail={
            "candidate_count": len(candidates),
            "excluded_count": len(songs) - len(candidates),
            "served": len(top_k),
        },
    )
    return top_k, log
