"""Unit tests for jarvis/router.py.

Covers:
- Known actionable tags return ACTIONABLE (Requirement 2.1)
- Unknown tags return CONVERSATIONAL (Requirement 2.2)
- Confidence < 0.5 always returns CONVERSATIONAL (Requirement 2.3)
"""

from __future__ import annotations

import pytest

from jarvis.models import Intent
from jarvis.router import RouteDecision, route

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ACTIONABLE_TAGS: frozenset[str] = frozenset(
    {"open_app", "create_reminder", "set_alarm", "send_email", "play_music"}
)


def make_intent(
    tag: str = "open_app",
    confidence: float = 0.9,
    entities: list | None = None,
    raw_input: str = "open chrome",
) -> Intent:
    return Intent(
        tag=tag,
        entities=entities or [],
        confidence=confidence,
        raw_input=raw_input,
    )


# ---------------------------------------------------------------------------
# Requirement 2.1 — known actionable tags return ACTIONABLE
# ---------------------------------------------------------------------------

class TestActionableTags:
    def test_open_app_is_actionable(self):
        intent = make_intent(tag="open_app", confidence=0.95)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_create_reminder_is_actionable(self):
        intent = make_intent(tag="create_reminder", confidence=0.8)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_set_alarm_is_actionable(self):
        intent = make_intent(tag="set_alarm", confidence=0.7)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_send_email_is_actionable(self):
        intent = make_intent(tag="send_email", confidence=0.6)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_play_music_is_actionable(self):
        intent = make_intent(tag="play_music", confidence=0.5)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_exact_confidence_boundary_is_actionable(self):
        """Confidence exactly 0.5 is NOT below 0.5, so tag check applies."""
        intent = make_intent(tag="open_app", confidence=0.5)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_custom_actionable_tag_set(self):
        """route() uses the passed-in set, not any global state."""
        custom_tags = frozenset({"custom_action"})
        intent = make_intent(tag="custom_action", confidence=0.9)
        assert route(intent, custom_tags) == RouteDecision.ACTIONABLE

    def test_empty_actionable_set_never_actionable(self):
        intent = make_intent(tag="open_app", confidence=0.9)
        assert route(intent, frozenset()) == RouteDecision.CONVERSATIONAL


# ---------------------------------------------------------------------------
# Requirement 2.2 — unknown tags return CONVERSATIONAL
# ---------------------------------------------------------------------------

class TestConversationalTags:
    def test_unknown_tag_is_conversational(self):
        intent = make_intent(tag="explain_concept", confidence=0.9)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_empty_tag_is_conversational(self):
        intent = make_intent(tag="", confidence=0.9)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_partial_match_is_not_actionable(self):
        """'open' is not the same as 'open_app'."""
        intent = make_intent(tag="open", confidence=0.9)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_case_sensitive_tag_mismatch(self):
        """Tag matching is case-sensitive."""
        intent = make_intent(tag="Open_App", confidence=0.9)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_general_question_is_conversational(self):
        intent = make_intent(tag="general_question", confidence=0.85)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL


# ---------------------------------------------------------------------------
# Requirement 2.3 — confidence < 0.5 always returns CONVERSATIONAL
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_low_confidence_actionable_tag_is_conversational(self):
        """Even a known actionable tag must be CONVERSATIONAL when confidence < 0.5."""
        intent = make_intent(tag="open_app", confidence=0.49)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_zero_confidence_is_conversational(self):
        intent = make_intent(tag="open_app", confidence=0.0)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_just_below_threshold_is_conversational(self):
        intent = make_intent(tag="send_email", confidence=0.4999)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_low_confidence_unknown_tag_is_conversational(self):
        intent = make_intent(tag="explain_concept", confidence=0.3)
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL

    def test_confidence_just_at_threshold_is_not_low(self):
        """0.5 is the boundary — NOT below 0.5, so tag check applies."""
        intent = make_intent(tag="open_app", confidence=0.5)
        # open_app is in ACTIONABLE_TAGS, so should be ACTIONABLE
        assert route(intent, ACTIONABLE_TAGS) == RouteDecision.ACTIONABLE

    def test_low_confidence_overrides_any_tag(self):
        """Confidence check takes priority over tag membership."""
        for tag in ACTIONABLE_TAGS:
            intent = make_intent(tag=tag, confidence=0.1)
            assert route(intent, ACTIONABLE_TAGS) == RouteDecision.CONVERSATIONAL, (
                f"Expected CONVERSATIONAL for tag={tag!r} with confidence=0.1"
            )


# ---------------------------------------------------------------------------
# Purity / determinism
# ---------------------------------------------------------------------------

class TestPurity:
    def test_same_inputs_same_output(self):
        """Calling route twice with identical inputs must return the same result."""
        intent = make_intent(tag="open_app", confidence=0.9)
        result1 = route(intent, ACTIONABLE_TAGS)
        result2 = route(intent, ACTIONABLE_TAGS)
        assert result1 == result2

    def test_does_not_mutate_actionable_tags(self):
        """route() must not modify the passed-in set."""
        tags = {"open_app", "send_email"}
        original_tags = set(tags)
        intent = make_intent(tag="open_app", confidence=0.9)
        route(intent, tags)
        assert tags == original_tags

    def test_does_not_mutate_intent(self):
        """route() must not modify the intent object."""
        intent = make_intent(tag="open_app", confidence=0.9)
        original_tag = intent.tag
        original_confidence = intent.confidence
        route(intent, ACTIONABLE_TAGS)
        assert intent.tag == original_tag
        assert intent.confidence == original_confidence
