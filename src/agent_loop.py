"""
Autonomous agentic refinement loop (assets/design_spec.md section 8): each
iteration asks Gemini for one structured facet-weight adjustment, applies
it, and repeats until the top-k stabilizes for a few iterations in a row or
a hard cap is hit -- no human input needed per step. Runs before every
generated list, on top of whatever the human's own feedback already did.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from recommender import Facet, Reason, Song
    from session import RecommendationSession, rebuild_pool
except ImportError:
    from src.recommender import Facet, Reason, Song
    from src.session import RecommendationSession, rebuild_pool

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"
_client: Optional["genai.Client"] = None

_MIN_MAGNITUDE = 0.05
_MAX_MAGNITUDE = 0.3

_AGENT_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "facet": {
            "type": "string",
            "enum": ["energy", "tempo", "valence", "danceability", "acousticness", "genre", "mood"],
        },
        "label": {
            "type": "string",
            "description": "Required only when facet is 'genre' or 'mood' -- must be one of the genre/mood labels shown above. Empty string otherwise.",
        },
        "direction": {"type": "string", "enum": ["up", "down"]},
        "magnitude": {
            "type": "number",
            "description": "How strongly to adjust, from 0.05 (subtle) to 0.3 (strong).",
        },
        "rationale": {
            "type": "string",
            "description": "One short sentence (20 words or fewer) explaining the choice.",
        },
        "songs_compared": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional: titles of the songs being compared, if any.",
        },
    },
    "required": ["facet", "direction", "magnitude", "rationale"],
}


def _get_client() -> "genai.Client":
    """Lazily creates and caches the Gemini client, raising if GEMINI_API_KEY isn't set."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _build_decision_prompt(session: RecommendationSession, topk: List[Tuple[Song, float, List[Reason]]]) -> str:
    """Describes the stated taste profile, current top candidates, and weights applied so far, and asks for one adjustment."""
    user = session.base_user
    profile_bits = []
    if user.favorite_genres:
        profile_bits.append(f"favorite genres: {', '.join(sorted(user.favorite_genres))}")
    if user.favorite_moods:
        profile_bits.append(f"favorite moods: {', '.join(sorted(user.favorite_moods))}")
    profile_bits.append(
        f"target energy={user.target_energy:.2f}, valence={user.target_valence:.2f}, "
        f"danceability={user.target_danceability:.2f}, acousticness={user.target_acousticness:.2f}, "
        f"tempo={user.target_tempo:.2f}"
    )
    profile_text = "; ".join(profile_bits)

    candidates_text = "\n".join(
        f'- "{song.title}" by {song.artist} ({song.genre}, {song.mood}) -- score {score:.2f}'
        for song, score, _reasons in topk[:5]
    )

    weight_bits = [f"{name} ({delta:+.2f})" for name, delta in session.facet_weights.heaviest(5)]
    weights_text = ", ".join(weight_bits) if weight_bits else "none yet"

    return (
        "You are an autonomous music-recommendation refinement agent. Your "
        "job is to make ONE small adjustment per turn to better align a "
        "ranked playlist with a listener's stated taste, then stop -- "
        "another turn will follow.\n\n"
        f"Listener's stated taste: {profile_text}\n\n"
        f"Current top candidates:\n{candidates_text}\n\n"
        f"Adjustments already applied this session: {weights_text}\n\n"
        "Pick exactly one facet to nudge (a continuous one -- energy, "
        "tempo, valence, danceability, acousticness -- or a genre/mood "
        "label to boost/penalize) and a direction and magnitude. You may "
        "reference specific songs above by name if you're effectively "
        "comparing them. Keep the rationale to one short sentence."
    )


def _get_agent_decision(session: RecommendationSession, topk: List[Tuple[Song, float, List[Reason]]]) -> Optional[Dict]:
    """Asks Gemini for one structured decision, returning the parsed dict or None if the call fails."""
    prompt = _build_decision_prompt(session, topk)
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_AGENT_DECISION_SCHEMA,
            ),
        )
        return json.loads(response.text)
    except Exception:
        return None


def apply_agent_decision(session: RecommendationSession, decision: Dict) -> Optional[Facet]:
    """Applies one agent decision to session.facet_weights, clamping magnitude into [0.05, 0.3] regardless of what the model actually returned."""
    facet = Facet(decision["facet"])
    direction = 1.0 if decision["direction"] == "up" else -1.0
    magnitude = max(_MIN_MAGNITUDE, min(_MAX_MAGNITUDE, abs(decision["magnitude"])))

    if facet in (Facet.GENRE, Facet.MOOD):
        label = decision.get("label") or ""
        if not label:
            return None
        boosts = session.facet_weights.genre_boosts if facet == Facet.GENRE else session.facet_weights.mood_boosts
        boosts[label] = boosts.get(label, 0.0) + direction * magnitude
    else:
        attr = f"target_{facet.value}_delta"
        setattr(session.facet_weights, attr, getattr(session.facet_weights, attr) + direction * magnitude)

    return facet


def run_agentic_refinement(
    session: RecommendationSession,
    catalog: List[Song],
    k: int = 5,
    max_iterations: int = 5,
    stability_window: int = 3,
    on_step: Optional[Callable[[dict], None]] = None,
) -> Tuple[List[Tuple[Song, float, List[Reason]]], List[Dict]]:
    """
    Runs the autonomous refinement loop: each iteration asks Gemini for one
    structured decision, applies it, and stops once the top-k stabilizes
    for `stability_window` iterations in a row, `max_iterations` is hit, a
    decision call fails, or the candidate pool runs out.
    """
    log: List[Dict] = []
    previous_ids: Optional[Tuple[int, ...]] = None
    stable_count = 0

    for iteration in range(1, max_iterations + 1):
        topk, _log = rebuild_pool(session, catalog, k=k)
        if not topk:
            break

        current_ids = tuple(song.id for song, _score, _reasons in topk)
        if current_ids == previous_ids:
            stable_count += 1
            if stable_count >= stability_window:
                break
        else:
            stable_count = 0
        previous_ids = current_ids

        decision = _get_agent_decision(session, topk)
        if decision is None:
            break

        try:
            apply_agent_decision(session, decision)
        except Exception:
            break

        step = {"iteration": iteration, **decision}
        log.append(step)
        if on_step is not None:
            on_step(step)

    final_topk, _log = rebuild_pool(session, catalog, k=k)
    return final_topk, log
