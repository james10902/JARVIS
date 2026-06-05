"""Unit tests for jarvis/formatter.py — format_output function.

Covers all four valid input combinations and the invariant that
``response`` is never empty.

Requirements: 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import pytest

from jarvis.formatter import format_output
from jarvis.models import ActionResult, JarvisOutput, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(text: str = "Here is your answer.") -> LLMResponse:
    return LLMResponse(text=text)


def _success(msg: str = "open -a 'Chrome'") -> ActionResult:
    return ActionResult.success(msg)


def _failure(msg: str = "Skill not found: unknown") -> ActionResult:
    return ActionResult.failure(msg)


# ---------------------------------------------------------------------------
# Combination 1: LLM response + ActionResult.Success  (Requirement 7.1)
# ---------------------------------------------------------------------------

class TestLLMAndSuccess:
    """Both LLM response and a successful action result are present."""

    def test_returns_jarvis_output(self):
        result = format_output(_llm(), _success())
        assert isinstance(result, JarvisOutput)

    def test_response_equals_llm_text(self):
        llm = _llm("Opening Chrome for you.")
        result = format_output(llm, _success("open -a 'Chrome'"))
        assert result.response == "Opening Chrome for you."

    def test_action_equals_success_message(self):
        result = format_output(_llm(), _success("open -a 'Chrome'"))
        assert result.action == "open -a 'Chrome'"

    def test_response_is_non_empty(self):
        result = format_output(_llm("Some reply."), _success("cmd"))
        assert result.response != ""

    def test_action_is_non_empty(self):
        result = format_output(_llm(), _success("cmd"))
        assert result.action is not None and result.action != ""


# ---------------------------------------------------------------------------
# Combination 2: LLM response only (no action result)  (Requirement 7.2)
# ---------------------------------------------------------------------------

class TestLLMOnly:
    """Only an LLM response is provided; no action result."""

    def test_returns_jarvis_output(self):
        result = format_output(_llm(), None)
        assert isinstance(result, JarvisOutput)

    def test_response_equals_llm_text(self):
        llm = _llm("Quicksort is a divide-and-conquer algorithm.")
        result = format_output(llm, None)
        assert result.response == "Quicksort is a divide-and-conquer algorithm."

    def test_action_is_none(self):
        result = format_output(_llm(), None)
        assert result.action is None

    def test_response_is_non_empty(self):
        result = format_output(_llm("Non-empty reply."), None)
        assert result.response != ""


# ---------------------------------------------------------------------------
# Combination 3: ActionResult.Failure only  (Requirement 7.3)
# ---------------------------------------------------------------------------

class TestFailureOnly:
    """Only a failure action result is provided; no LLM response."""

    def test_returns_jarvis_output(self):
        result = format_output(None, _failure())
        assert isinstance(result, JarvisOutput)

    def test_response_contains_failure_message(self):
        result = format_output(None, _failure("Skill not found: open_app"))
        assert result.response == "Skill not found: open_app"

    def test_action_is_none(self):
        result = format_output(None, _failure())
        assert result.action is None

    def test_response_is_non_empty(self):
        result = format_output(None, _failure("Something went wrong."))
        assert result.response != ""


# ---------------------------------------------------------------------------
# Combination 4: ActionResult.Success only (no LLM response)
# ---------------------------------------------------------------------------

class TestSuccessOnly:
    """Only a successful action result is provided; no LLM response."""

    def test_returns_jarvis_output(self):
        result = format_output(None, _success())
        assert isinstance(result, JarvisOutput)

    def test_response_equals_success_message(self):
        result = format_output(None, _success("open -a 'Chrome'"))
        assert result.response == "open -a 'Chrome'"

    def test_action_equals_success_message(self):
        result = format_output(None, _success("open -a 'Chrome'"))
        assert result.action == "open -a 'Chrome'"

    def test_response_is_non_empty(self):
        result = format_output(None, _success("cmd"))
        assert result.response != ""

    def test_action_is_non_empty(self):
        result = format_output(None, _success("cmd"))
        assert result.action is not None and result.action != ""


# ---------------------------------------------------------------------------
# Invariant: response is never empty for any valid input  (Requirement 7.4)
# ---------------------------------------------------------------------------

class TestResponseNeverEmpty:
    """Parametrised check that response is non-empty across all valid combos."""

    @pytest.mark.parametrize(
        "llm_resp, act_result",
        [
            (_llm("reply"), _success("cmd")),   # LLM + Success
            (_llm("reply"), None),               # LLM only
            (None, _failure("err")),             # Failure only
            (None, _success("cmd")),             # Success only
            # LLM + Failure: LLM text is used as response
            (_llm("reply"), _failure("err")),
        ],
        ids=[
            "llm_and_success",
            "llm_only",
            "failure_only",
            "success_only",
            "llm_and_failure",
        ],
    )
    def test_response_non_empty(self, llm_resp, act_result):
        result = format_output(llm_resp, act_result)
        assert result.response != "", "response must never be empty"


# ---------------------------------------------------------------------------
# Edge case: both None raises ValueError  (Requirement 7.4 — precondition)
# ---------------------------------------------------------------------------

class TestBothNoneRaisesError:
    def test_raises_value_error(self):
        with pytest.raises(ValueError):
            format_output(None, None)


# ---------------------------------------------------------------------------
# LLM + Failure: action should be None, response from LLM
# ---------------------------------------------------------------------------

class TestLLMAndFailure:
    """When LLM response is present alongside a Failure, use LLM text and no action."""

    def test_action_is_none(self):
        result = format_output(_llm("I couldn't do that."), _failure("err"))
        assert result.action is None

    def test_response_is_llm_text(self):
        result = format_output(_llm("I couldn't do that."), _failure("err"))
        assert result.response == "I couldn't do that."
