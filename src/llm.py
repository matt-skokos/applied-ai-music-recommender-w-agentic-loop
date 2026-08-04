"""
Gemini-backed transparency message: one terse, personalized sentence per
round explaining which facets the session's feedback has been leaning
toward (assets/design_spec.md section on transparency messaging).

Scoped to the round-level message only -- per-song explanations
(Recommender.explain_recommendation / score_song) stay templated.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

try:
    from session import LogEntry, LogStatus, LogStep, RecommendationSession
except ImportError:
    from src.session import LogEntry, LogStatus, LogStep, RecommendationSession

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"
_client: Optional["genai.Client"] = None


def _get_client() -> "genai.Client":
    """Lazily creates and caches the Gemini client, raising if GEMINI_API_KEY isn't set."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _build_prompt(session: RecommendationSession) -> Optional[str]:
    """Builds the Gemini prompt from the session's heaviest facet weights, or None if there's no feedback yet to describe."""
    heaviest = session.facet_weights.heaviest(2)
    if not heaviest:
        return None

    signal = ", ".join(f"{name} ({delta:+.2f})" for name, delta in heaviest)
    return (
        "You are a terse, friendly music recommendation assistant. In one "
        "short sentence (20 words or fewer), tell the user what musical "
        f"qualities their thumbs up/down feedback has been steering toward "
        f"this session. Signal so far: {signal}. Describe it naturally -- "
        "no numbers, no weights, no facet jargon."
    )


def generate_transparency_message(session: RecommendationSession) -> Optional[str]:
    """Asks Gemini for a one-sentence transparency message, returning None if there's nothing to report yet or the call fails."""
    prompt = _build_prompt(session)
    if prompt is None:
        return None

    try:
        client = _get_client()
        response = client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        text = (response.text or "").strip()
        return text or None
    except Exception:
        return None


def build_transparency_message(session: RecommendationSession) -> "tuple[Optional[str], LogEntry]":
    """Generates the round's transparency message and its LogEntry, logging OK for 'no feedback yet' but ERROR (message skipped) for a failed Gemini call."""
    has_signal = bool(session.facet_weights.heaviest(1))
    message = generate_transparency_message(session)

    if message is not None:
        status = LogStatus.OK
        detail = {"message": message}
    elif not has_signal:
        status = LogStatus.OK
        detail = {"message": None, "reason": "no_feedback_yet"}
    else:
        status = LogStatus.ERROR
        detail = {"message": None, "reason": "gemini_call_failed"}

    log = LogEntry(
        session_id=session.session_id,
        round_number=session.round_number,
        step=LogStep.TRANSPARENCY_MESSAGE,
        status=status,
        detail=detail,
    )
    return message, log
