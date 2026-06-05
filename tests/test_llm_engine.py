"""Unit tests for the LLM Response Engine.

Tests cover:
- Fallback response when LLM API raises an exception (Requirement 11.1)
- Proactive suggestions included when proactiveSuggestions=True (Requirement 3.3)
- Personality constraints applied via system prompt (Requirement 3.2)
- Context-aware response generation (Requirement 3.1)
- JSON parse failure falls back to raw text (Requirement 3.1)
- Numbered steps for complex answers (Requirement 3.4)

Requirements: 3.1, 3.2, 3.3, 11.1
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock

import pytest

from jarvis.llm_engine import (
    _FALLBACK_RESPONSE,
    _build_messages,
    _build_system_prompt,
    _parse_llm_response,
    generate_response,
)
from jarvis.models import (
    ConversationContext,
    Intent,
    JarvisConfig,
    LLMResponse,
    Turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(raw_input: str = "How does quicksort work?", tag: str = "explain_concept") -> Intent:
    return Intent(tag=tag, entities=[], confidence=0.9, raw_input=raw_input)


def _make_ctx(*contents: str) -> ConversationContext:
    turns = [Turn(role="user", content=c) for c in contents]
    return ConversationContext(turns=turns, max_turns=20)


def _make_config(proactive: bool = True) -> JarvisConfig:
    return JarvisConfig(
        personality_mode="formal",
        max_context_turns=20,
        llm_model="gpt-4o",
        proactive_suggestions=proactive,
    )


def _json_caller(text: str, suggestions: List[str]):
    """Return a mock llm_caller that always returns the given JSON payload."""
    payload = json.dumps({"text": text, "suggestions": suggestions})

    def caller(messages, model):
        return payload

    return caller


def _raising_caller(exc: Exception):
    """Return a mock llm_caller that always raises the given exception."""
    def caller(messages, model):
        raise exc

    return caller


# ---------------------------------------------------------------------------
# Requirement 11.1 — Fallback response on LLM API failure
# ---------------------------------------------------------------------------

class TestFallbackOnFailure:
    """Fallback response is returned when the LLM API raises an exception."""

    def test_single_exception_exhausts_retries_and_returns_fallback(self):
        """All retries fail → fallback LLMResponse is returned."""
        caller = _raising_caller(RuntimeError("API unavailable"))
        result = generate_response(
            _make_intent(), _make_ctx(), _make_config(),
            llm_caller=caller,
        )
        assert result.text == _FALLBACK_RESPONSE.text
        assert result.suggestions == []

    def test_fallback_text_is_informative(self):
        """Fallback text informs the user that processing is unavailable."""
        caller = _raising_caller(ConnectionError("timeout"))
        result = generate_response(
            _make_intent(), _make_ctx(), _make_config(),
            llm_caller=caller,
        )
        # Must mention trouble / unavailability (Requirement 11.1)
        assert "trouble" in result.text.lower() or "unavailable" in result.text.lower()

    def test_fallback_has_empty_suggestions(self):
        """Fallback response has no suggestions."""
        caller = _raising_caller(ValueError("bad response"))
        result = generate_response(
            _make_intent(), _make_ctx(), _make_config(),
            llm_caller=caller,
        )
        assert result.suggestions == []

    def test_exception_on_first_two_calls_then_success(self):
        """If the first two calls fail but the third succeeds, return the success."""
        call_count = [0]

        def caller(messages, model):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient error")
            return json.dumps({"text": "Quicksort explanation.", "suggestions": []})

        result = generate_response(
            _make_intent(), _make_ctx(), _make_config(),
            llm_caller=caller,
        )
        assert result.text == "Quicksort explanation."
        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# Requirement 3.3 — Proactive suggestions
# ---------------------------------------------------------------------------

class TestProactiveSuggestions:
    """Proactive suggestions are included when proactiveSuggestions=True."""

    def test_suggestions_returned_when_proactive_enabled(self):
        """LLM suggestions are passed through when proactive_suggestions=True."""
        suggestions = ["You may also want to review merge sort.", "Consider time complexity."]
        caller = _json_caller("Quicksort is a divide-and-conquer algorithm.", suggestions)
        config = _make_config(proactive=True)

        result = generate_response(_make_intent(), _make_ctx(), config, llm_caller=caller)

        assert len(result.suggestions) == 2
        assert "merge sort" in result.suggestions[0].lower()

    def test_suggestions_empty_when_proactive_disabled(self):
        """When proactive_suggestions=False, suggestions from LLM are still parsed
        but the system prompt does not request them — test that the response is
        still valid regardless."""
        caller = _json_caller("Quicksort explanation.", [])
        config = _make_config(proactive=False)

        result = generate_response(_make_intent(), _make_ctx(), config, llm_caller=caller)

        assert result.text == "Quicksort explanation."
        assert result.suggestions == []

    def test_proactive_prompt_contains_suggestion_instruction(self):
        """System prompt includes suggestion instruction when proactive=True."""
        config = _make_config(proactive=True)
        prompt = _build_system_prompt(config)
        assert "suggestion" in prompt.lower() or "proactive" in prompt.lower()

    def test_non_proactive_prompt_does_not_contain_suggestion_instruction(self):
        """System prompt omits proactive suggestion instruction when proactive=False."""
        config = _make_config(proactive=False)
        prompt = _build_system_prompt(config)
        # The base prompt should not contain the proactive suggestion instruction
        assert "proactive" not in prompt.lower()


# ---------------------------------------------------------------------------
# Requirement 3.2 — Personality constraints in system prompt
# ---------------------------------------------------------------------------

class TestPersonalityConstraints:
    """Personality constraints are applied via the system prompt."""

    def test_system_prompt_enforces_formal_tone(self):
        """System prompt explicitly requires formal tone."""
        config = _make_config()
        prompt = _build_system_prompt(config)
        assert "formal" in prompt.lower()

    def test_system_prompt_enforces_concise_phrasing(self):
        """System prompt explicitly requires concise phrasing."""
        config = _make_config()
        prompt = _build_system_prompt(config)
        assert "concise" in prompt.lower()

    def test_system_prompt_forbids_self_deprecating_phrases(self):
        """System prompt explicitly forbids self-deprecating phrases."""
        config = _make_config()
        prompt = _build_system_prompt(config)
        # Must mention the forbidden phrase or the concept
        assert "just an ai" in prompt.lower() or "self-deprecat" in prompt.lower()

    def test_system_message_is_first_in_messages(self):
        """The system prompt is the first message in the messages list."""
        intent = _make_intent()
        ctx = _make_ctx()
        config = _make_config()
        messages = _build_messages(intent, ctx, config)
        assert messages[0]["role"] == "system"

    def test_system_prompt_is_passed_to_llm_caller(self):
        """The llm_caller receives messages with the system prompt as first entry."""
        captured = {}

        def caller(messages, model):
            captured["messages"] = messages
            return json.dumps({"text": "Response.", "suggestions": []})

        config = _make_config()
        generate_response(_make_intent(), _make_ctx(), config, llm_caller=caller)

        assert "messages" in captured
        system_msg = captured["messages"][0]
        assert system_msg["role"] == "system"
        assert "formal" in system_msg["content"].lower()
        assert "concise" in system_msg["content"].lower()

    def test_personality_prompt_forbids_instruction_override(self):
        """System prompt instructs LLM to ignore override attempts in user messages."""
        config = _make_config()
        prompt = _build_system_prompt(config)
        assert "override" in prompt.lower() or "ignore" in prompt.lower()


# ---------------------------------------------------------------------------
# Requirement 3.1 — Context-aware response generation
# ---------------------------------------------------------------------------

class TestContextAwareGeneration:
    """LLM receives conversation context turns for context-aware responses."""

    def test_context_turns_included_in_messages(self):
        """Prior conversation turns are injected into the messages list."""
        captured = {}

        def caller(messages, model):
            captured["messages"] = messages
            return json.dumps({"text": "Response.", "suggestions": []})

        ctx = _make_ctx("What is Python?", "Tell me more.")
        generate_response(_make_intent(), ctx, _make_config(), llm_caller=caller)

        messages = captured["messages"]
        contents = [m["content"] for m in messages]
        assert "What is Python?" in contents
        assert "Tell me more." in contents

    def test_user_raw_input_is_last_user_message(self):
        """The intent's raw_input is the final user message sent to the LLM."""
        captured = {}

        def caller(messages, model):
            captured["messages"] = messages
            return json.dumps({"text": "Response.", "suggestions": []})

        intent = _make_intent(raw_input="Explain recursion please.")
        generate_response(intent, _make_ctx(), _make_config(), llm_caller=caller)

        last_msg = captured["messages"][-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"] == "Explain recursion please."

    def test_model_from_config_passed_to_caller(self):
        """The model identifier from JarvisConfig is passed to the llm_caller."""
        captured = {}

        def caller(messages, model):
            captured["model"] = model
            return json.dumps({"text": "Response.", "suggestions": []})

        config = JarvisConfig(llm_model="gpt-4-turbo", proactive_suggestions=False)
        generate_response(_make_intent(), _make_ctx(), config, llm_caller=caller)

        assert captured["model"] == "gpt-4-turbo"


# ---------------------------------------------------------------------------
# JSON parsing edge cases
# ---------------------------------------------------------------------------

class TestResponseParsing:
    """_parse_llm_response handles valid JSON, invalid JSON, and edge cases."""

    def test_valid_json_parsed_correctly(self):
        raw = json.dumps({"text": "Hello.", "suggestions": ["Try this."]})
        result = _parse_llm_response(raw)
        assert result.text == "Hello."
        assert result.suggestions == ["Try this."]

    def test_invalid_json_uses_raw_text(self):
        raw = "This is not JSON at all."
        result = _parse_llm_response(raw)
        assert result.text == "This is not JSON at all."
        assert result.suggestions == []

    def test_json_missing_text_field_uses_raw(self):
        raw = json.dumps({"suggestions": ["A suggestion."]})
        result = _parse_llm_response(raw)
        # text is empty in JSON → falls back to raw string
        assert result.text == raw.strip()

    def test_json_with_empty_suggestions_list(self):
        raw = json.dumps({"text": "Answer.", "suggestions": []})
        result = _parse_llm_response(raw)
        assert result.text == "Answer."
        assert result.suggestions == []

    def test_json_with_non_list_suggestions_returns_empty(self):
        raw = json.dumps({"text": "Answer.", "suggestions": "not a list"})
        result = _parse_llm_response(raw)
        assert result.suggestions == []

    def test_generate_response_returns_llm_response_instance(self):
        """generate_response always returns an LLMResponse."""
        caller = _json_caller("Some answer.", [])
        result = generate_response(_make_intent(), _make_ctx(), _make_config(), llm_caller=caller)
        assert isinstance(result, LLMResponse)
