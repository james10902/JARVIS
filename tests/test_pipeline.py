"""Unit and integration tests for jarvis/pipeline.py.

Covers:
- Conversational query → LLM response → formatted output with action=None
- Actionable command → skill dispatch → system action → formatted output with action set
- Low-confidence intent → clarification question, action=None
- Multi-turn conversation → context memory → turns persisted
- Skill not found → failure message in response, action=None

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import json

import pytest

from jarvis.models import (
    ActionResult,
    ConversationContext,
    JarvisConfig,
    JarvisOutput,
    Skill,
    Turn,
)
from jarvis.pipeline import process_input
from jarvis.skill_registry import SkillRegistry


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> JarvisConfig:
    """Build a JarvisConfig with sensible test defaults."""
    defaults = {
        "personality_mode": "formal",
        "max_context_turns": 20,
        "llm_model": "gpt-4o",
        "verbose_mode": False,
        "proactive_suggestions": False,
    }
    defaults.update(kwargs)
    return JarvisConfig(**defaults)


def _empty_ctx(max_turns: int = 20) -> ConversationContext:
    return ConversationContext(turns=[], max_turns=max_turns)


def _make_registry(*skills: Skill) -> SkillRegistry:
    reg = SkillRegistry()
    for skill in skills:
        reg.register(skill)
    return reg


def _open_app_skill() -> Skill:
    """A simple skill that opens an app."""
    return Skill(
        id="open_app",
        description="Opens a named application",
        intent_tags=["open_app"],
        required_params=["app"],
        execute=lambda params: ActionResult.success(f"open -a '{params['app']}'"),
    )


def _nlu_caller_for(tag: str, confidence: float, entities: dict | None = None) -> callable:
    """Return an NLU caller that always returns the given intent JSON."""
    def _caller(messages: list[dict]) -> str:
        return json.dumps({
            "tag": tag,
            "entities": entities or {},
            "confidence": confidence,
        })
    return _caller


def _llm_caller_returning(text: str) -> callable:
    """Return an LLM caller that always returns the given response text."""
    def _caller(messages: list[dict], model: str) -> str:
        return json.dumps({"text": text, "suggestions": []})
    return _caller


# ---------------------------------------------------------------------------
# Requirement 8.3 — Conversational query → LLM response → action=None
# ---------------------------------------------------------------------------

class TestConversationalPath:
    """Conversational intents produce an LLM response with action=None."""

    def test_conversational_response_is_non_empty(self):
        """A conversational query produces a non-empty response."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Quicksort is a divide-and-conquer algorithm.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("Explain quicksort", ctx, registry, config,
                                   nlu_caller=nlu, llm_caller=llm)

        assert output.response != ""
        assert "Quicksort" in output.response or len(output.response) > 0

    def test_conversational_action_is_none(self):
        """Conversational intents must produce action=None (Requirement 8.3)."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Here is an explanation.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("Explain something", ctx, registry, config,
                                   nlu_caller=nlu, llm_caller=llm)

        assert output.action is None

    def test_conversational_response_matches_llm_output(self):
        """The response field reflects the LLM-generated text."""
        expected_text = "This is the LLM response."
        nlu = _nlu_caller_for("general_question", confidence=0.85)
        llm = _llm_caller_returning(expected_text)
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("What is the meaning of life?", ctx, registry, config,
                                   nlu_caller=nlu, llm_caller=llm)

        assert output.response == expected_text

    def test_unknown_tag_routes_to_conversational(self):
        """A tag not in the registry is treated as conversational."""
        nlu = _nlu_caller_for("unknown_tag_xyz", confidence=0.9)
        llm = _llm_caller_returning("I can help with that.")
        registry = _make_registry()  # empty registry → no actionable tags
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("Do something", ctx, registry, config,
                                   nlu_caller=nlu, llm_caller=llm)

        assert output.action is None


# ---------------------------------------------------------------------------
# Requirement 8.2 — Actionable command → dispatch → action set
# ---------------------------------------------------------------------------

class TestActionablePath:
    """Actionable intents dispatch to skills and set the action field."""

    def test_actionable_output_has_action(self):
        """Successful dispatch sets the action field (Requirement 8.2)."""
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={"app": "chrome"})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open chrome", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is not None
        assert output.action != ""

    def test_actionable_action_contains_command(self):
        """The action field contains the command string from the skill."""
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={"app": "chrome"})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open chrome", ctx, registry, config,
                                   nlu_caller=nlu)

        assert "chrome" in output.action

    def test_actionable_response_is_non_empty(self):
        """Actionable dispatch always produces a non-empty response (Requirement 8.1)."""
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={"app": "firefox"})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open firefox", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.response != ""

    def test_actionable_with_exact_confidence_boundary(self):
        """Confidence exactly 0.5 is actionable (not low-confidence)."""
        nlu = _nlu_caller_for("open_app", confidence=0.5, entities={"app": "safari"})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open safari", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is not None


# ---------------------------------------------------------------------------
# Requirement 8.4 — Low-confidence intent → clarification question
# ---------------------------------------------------------------------------

class TestLowConfidencePath:
    """Low-confidence intents produce a clarification question with action=None."""

    def test_low_confidence_returns_clarification(self):
        """Confidence < 0.5 returns the clarification question (Requirement 8.4)."""
        nlu = _nlu_caller_for("open_app", confidence=0.3)
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("do the thing", ctx, registry, config,
                                   nlu_caller=nlu)

        assert "clarify" in output.response.lower()

    def test_low_confidence_action_is_none(self):
        """Low-confidence intents must have action=None (Requirement 8.4)."""
        nlu = _nlu_caller_for("open_app", confidence=0.0)
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("hmm", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is None

    def test_low_confidence_response_is_non_empty(self):
        """Low-confidence response is always non-empty (Requirement 8.1)."""
        nlu = _nlu_caller_for("unknown", confidence=0.1)
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("???", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.response != ""

    def test_just_below_threshold_is_low_confidence(self):
        """Confidence 0.4999 is below 0.5 and triggers clarification."""
        nlu = _nlu_caller_for("open_app", confidence=0.4999)
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open something maybe", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is None
        assert output.response != ""

    def test_zero_confidence_triggers_clarification(self):
        """Zero confidence always triggers clarification."""
        nlu = _nlu_caller_for("open_app", confidence=0.0)
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("...", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is None


# ---------------------------------------------------------------------------
# Requirement 8.5 — Context memory: turns persisted
# ---------------------------------------------------------------------------

class TestContextPersistence:
    """Interactions are persisted as Turns in the ConversationContext."""

    def test_turns_added_after_conversational(self):
        """Two turns (user + jarvis) are added after a conversational interaction."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Here is the explanation.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        _, updated_ctx = process_input("Explain something", ctx, registry, config,
                                        nlu_caller=nlu, llm_caller=llm)

        assert len(updated_ctx.turns) == 2

    def test_user_turn_persisted(self):
        """The user input is persisted as a 'user' turn."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Response text.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        _, updated_ctx = process_input("My question", ctx, registry, config,
                                        nlu_caller=nlu, llm_caller=llm)

        user_turns = [t for t in updated_ctx.turns if t.role == "user"]
        assert len(user_turns) == 1
        assert user_turns[0].content == "My question"

    def test_jarvis_turn_persisted(self):
        """The JARVIS response is persisted as a 'jarvis' turn."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("JARVIS response here.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, updated_ctx = process_input("Question", ctx, registry, config,
                                             nlu_caller=nlu, llm_caller=llm)

        jarvis_turns = [t for t in updated_ctx.turns if t.role == "jarvis"]
        assert len(jarvis_turns) == 1
        assert jarvis_turns[0].content == output.response

    def test_turns_persisted_for_low_confidence(self):
        """Turns are persisted even for low-confidence (clarification) interactions."""
        nlu = _nlu_caller_for("open_app", confidence=0.2)
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        _, updated_ctx = process_input("do the thing", ctx, registry, config,
                                        nlu_caller=nlu)

        assert len(updated_ctx.turns) == 2

    def test_turns_persisted_for_actionable(self):
        """Turns are persisted after a successful skill dispatch."""
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={"app": "chrome"})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        _, updated_ctx = process_input("open chrome", ctx, registry, config,
                                        nlu_caller=nlu)

        assert len(updated_ctx.turns) == 2

    def test_multi_turn_conversation_accumulates_turns(self):
        """Multiple interactions accumulate turns in the context."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Response.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        _, ctx = process_input("First question", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)
        _, ctx = process_input("Second question", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)
        _, ctx = process_input("Third question", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)

        # 3 interactions × 2 turns each = 6 turns
        assert len(ctx.turns) == 6

    def test_multi_turn_rolling_window_enforced(self):
        """Rolling window is enforced across multiple interactions."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Response.")
        registry = _make_registry()
        config = _make_config()
        # max_turns=4 means only 4 turns are kept (2 interactions)
        ctx = _empty_ctx(max_turns=4)

        _, ctx = process_input("First", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)
        _, ctx = process_input("Second", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)
        _, ctx = process_input("Third", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)

        # Rolling window caps at max_turns=4
        assert len(ctx.turns) <= 4

    def test_original_context_not_mutated(self):
        """The original context is not mutated (pure style)."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Response.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()
        original_turn_count = len(ctx.turns)

        _, updated_ctx = process_input("Question", ctx, registry, config,
                                        nlu_caller=nlu, llm_caller=llm)

        # Original context unchanged
        assert len(ctx.turns) == original_turn_count
        # Updated context has new turns
        assert len(updated_ctx.turns) > original_turn_count


# ---------------------------------------------------------------------------
# Requirements 11.2, 11.3 — Error recovery: skill not found, missing params
# ---------------------------------------------------------------------------

class TestErrorRecovery:
    """Error conditions produce user-friendly responses with action=None."""

    def test_skill_not_found_response_is_non_empty(self):
        """Skill not found produces a non-empty response (Requirement 11.2)."""
        nlu = _nlu_caller_for("nonexistent_skill", confidence=0.9)
        registry = _make_registry()  # register a skill so the tag is "actionable"
        # Register a skill with a different tag to make the registry non-empty
        # but the intent tag won't match any skill ID
        dummy_skill = Skill(
            id="nonexistent_skill",
            description="Dummy",
            intent_tags=["nonexistent_skill"],
            required_params=[],
            execute=lambda p: ActionResult.success("ok"),
        )
        registry.register(dummy_skill)
        # Now remove it to simulate skill not found at dispatch time
        # Actually, let's use a skill whose ID differs from the intent tag
        registry2 = _make_registry()
        other_skill = Skill(
            id="other_skill",
            description="Other",
            intent_tags=["nonexistent_skill"],  # tag maps to this skill in registry
            required_params=[],
            execute=lambda p: ActionResult.success("ok"),
        )
        registry2.register(other_skill)
        # The intent tag is "nonexistent_skill" but the skill ID is "other_skill"
        # dispatch will look up by skill_id = intent.tag = "nonexistent_skill" → not found
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("do nonexistent thing", ctx, registry2, config,
                                   nlu_caller=nlu)

        assert output.response != ""

    def test_skill_not_found_action_is_none(self):
        """Skill not found produces action=None.

        When lookup_by_tag returns no match, the pipeline falls back to using
        intent.tag as the skill_id, which is also not registered, so dispatch
        returns Failure and the formatter sets action=None.
        """
        # Intent tag "missing_skill_xyz" has NO registered skill at all —
        # lookup_by_tag returns [], the fallback skill_id is also unregistered,
        # so dispatch returns Failure and action must be None.
        nlu = _nlu_caller_for("missing_skill_xyz", confidence=0.9)
        # Register a completely unrelated skill so the registry is non-empty
        # (ensures the tag really is absent, not that the registry is empty).
        unrelated = Skill(
            id="unrelated",
            description="Unrelated skill",
            intent_tags=["unrelated_tag"],
            required_params=[],
            execute=lambda p: ActionResult.success("ok"),
        )
        registry = _make_registry(unrelated)
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("do missing thing", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is None

    def test_missing_params_response_is_non_empty(self):
        """Missing required params produces a non-empty response (Requirement 11.3)."""
        # Intent has no entities, but skill requires "app"
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open something", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.response != ""

    def test_missing_params_action_is_none(self):
        """Missing required params produces action=None."""
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open something", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.action is None

    def test_missing_params_message_mentions_missing(self):
        """Missing params failure message mentions the missing parameter."""
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open something", ctx, registry, config,
                                   nlu_caller=nlu)

        # The dispatcher returns "Missing params: app" which formatter surfaces
        assert "app" in output.response.lower() or "missing" in output.response.lower()

    def test_skill_execution_error_produces_response(self):
        """Skill execution errors produce a non-empty response (Requirement 11.4)."""
        def _failing_execute(params):
            raise RuntimeError("OS error: permission denied")

        failing_skill = Skill(
            id="risky_skill",
            description="Always fails",
            intent_tags=["risky_skill"],
            required_params=[],
            execute=_failing_execute,
        )
        nlu = _nlu_caller_for("risky_skill", confidence=0.9)
        registry = _make_registry(failing_skill)
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("do risky thing", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.response != ""
        assert output.action is None


# ---------------------------------------------------------------------------
# Requirement 8.1 — Output always has non-empty response
# ---------------------------------------------------------------------------

class TestOutputInvariants:
    """The pipeline always produces a non-empty response."""

    def test_response_non_empty_conversational(self):
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Some response.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("Tell me something", ctx, registry, config,
                                   nlu_caller=nlu, llm_caller=llm)

        assert isinstance(output, JarvisOutput)
        assert output.response != ""

    def test_response_non_empty_actionable(self):
        nlu = _nlu_caller_for("open_app", confidence=0.9, entities={"app": "chrome"})
        registry = _make_registry(_open_app_skill())
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("open chrome", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.response != ""

    def test_response_non_empty_low_confidence(self):
        nlu = _nlu_caller_for("open_app", confidence=0.1)
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        output, _ = process_input("hmm", ctx, registry, config,
                                   nlu_caller=nlu)

        assert output.response != ""

    def test_returns_tuple_of_output_and_context(self):
        """process_input returns a (JarvisOutput, ConversationContext) tuple."""
        nlu = _nlu_caller_for("explain_concept", confidence=0.9)
        llm = _llm_caller_returning("Response.")
        registry = _make_registry()
        config = _make_config()
        ctx = _empty_ctx()

        result = process_input("Question", ctx, registry, config,
                                nlu_caller=nlu, llm_caller=llm)

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, updated_ctx = result
        assert isinstance(output, JarvisOutput)
        assert isinstance(updated_ctx, ConversationContext)
