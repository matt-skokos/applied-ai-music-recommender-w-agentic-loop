"""
Covers the round-level transparency message: personalized via Gemini when
there's signal to report, and the chosen fallback behavior (log
LogStatus.ERROR, skip the message -- no templated substitute) when the
Gemini call fails. Network calls are always mocked; no real API key needed.
"""

from types import SimpleNamespace

from src.llm import build_transparency_message
from src.session import LogStatus, LogStep, RecommendationSession


def _session_with_signal() -> RecommendationSession:
    session = RecommendationSession()
    session.facet_weights.genre_boosts["lofi"] = 0.3
    return session


def test_build_transparency_message_no_feedback_yet_is_ok_with_no_message():
    session = RecommendationSession()  # no weights touched yet

    message, log = build_transparency_message(session)

    assert message is None
    assert log.step == LogStep.TRANSPARENCY_MESSAGE
    assert log.status == LogStatus.OK
    assert log.detail["reason"] == "no_feedback_yet"


def test_build_transparency_message_success_returns_gemini_text(monkeypatch):
    session = _session_with_signal()
    fake_response = SimpleNamespace(text="Leaning into lofi vibes tonight.")
    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: fake_response))
    monkeypatch.setattr("src.llm._get_client", lambda: fake_client)

    message, log = build_transparency_message(session)

    assert message == "Leaning into lofi vibes tonight."
    assert log.status == LogStatus.OK
    assert log.detail["message"] == message


def test_build_transparency_message_gemini_failure_logs_error_and_skips_message(monkeypatch):
    session = _session_with_signal()

    def _boom():
        raise RuntimeError("no api key configured")

    monkeypatch.setattr("src.llm._get_client", _boom)

    message, log = build_transparency_message(session)

    assert message is None
    assert log.status == LogStatus.ERROR
    assert log.detail["reason"] == "gemini_call_failed"
