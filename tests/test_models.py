"""Unit tests for jarvis/models.py — core data models and JarvisConfig validation.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import pytest

from jarvis.models import (
    ActionRequest,
    ActionResult,
    ActionResultVariant,
    ConversationContext,
    Intent,
    JarvisConfig,
    JarvisOutput,
    LLMResponse,
    Skill,
    Turn,
)


# ---------------------------------------------------------------------------
# JarvisConfig — valid construction
# ---------------------------------------------------------------------------

class TestJarvisConfigValid:
    def test_default_construction(self):
        cfg = JarvisConfig()
        assert cfg.personality_mode == "formal"
        assert cfg.max_context_turns == 20
        assert cfg.llm_model == "gpt-4o"
        assert cfg.verbose_mode is False
        assert cfg.proactive_suggestions is True

    def test_formal_personality(self):
        cfg = JarvisConfig(personality_mode="formal")
        assert cfg.personality_mode == "formal"

    def test_casual_personality(self):
        cfg = JarvisConfig(personality_mode="casual")
        assert cfg.personality_mode == "casual"

    def test_custom_max_context_turns(self):
        cfg = JarvisConfig(max_context_turns=5)
        assert cfg.max_context_turns == 5

    def test_max_context_turns_of_one(self):
        cfg = JarvisConfig(max_context_turns=1)
        assert cfg.max_context_turns == 1

    def test_custom_llm_model(self):
        cfg = JarvisConfig(llm_model="gpt-3.5-turbo")
        assert cfg.llm_model == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# JarvisConfig — validation failures (Requirements 9.2, 9.3, 9.4)
# ---------------------------------------------------------------------------

class TestJarvisConfigValidation:
    def test_invalid_personality_mode_raises(self):
        """Requirement 9.4: personalityMode must be 'formal' or 'casual'."""
        with pytest.raises(ValueError, match="personalityMode"):
            JarvisConfig(personality_mode="aggressive")

    def test_empty_personality_mode_raises(self):
        with pytest.raises(ValueError, match="personalityMode"):
            JarvisConfig(personality_mode="")

    def test_max_context_turns_zero_raises(self):
        """Requirement 9.2: maxContextTurns must be > 0."""
        with pytest.raises(ValueError, match="maxContextTurns"):
            JarvisConfig(max_context_turns=0)

    def test_max_context_turns_negative_raises(self):
        """Requirement 9.2: maxContextTurns must be > 0."""
        with pytest.raises(ValueError, match="maxContextTurns"):
            JarvisConfig(max_context_turns=-1)

    def test_max_context_turns_large_negative_raises(self):
        with pytest.raises(ValueError, match="maxContextTurns"):
            JarvisConfig(max_context_turns=-100)

    def test_empty_llm_model_raises(self):
        """Requirement 9.3: llmModel must be non-empty."""
        with pytest.raises(ValueError, match="llmModel"):
            JarvisConfig(llm_model="")

    def test_whitespace_only_llm_model_raises(self):
        """Requirement 9.3: llmModel must be non-empty (whitespace-only counts as empty)."""
        with pytest.raises(ValueError, match="llmModel"):
            JarvisConfig(llm_model="   ")


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

class TestIntent:
    def test_basic_construction(self):
        intent = Intent(
            tag="open_app",
            entities=[("app", "chrome")],
            confidence=0.9,
            raw_input="open chrome",
        )
        assert intent.tag == "open_app"
        assert intent.entities == [("app", "chrome")]
        assert intent.confidence == 0.9
        assert intent.raw_input == "open chrome"

    def test_empty_entities(self):
        intent = Intent(tag="greet", entities=[], confidence=1.0, raw_input="hello")
        assert intent.entities == []


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------

class TestTurn:
    def test_user_turn(self):
        turn = Turn(role="user", content="Hello JARVIS", ts=1000)
        assert turn.role == "user"
        assert turn.content == "Hello JARVIS"
        assert turn.ts == 1000

    def test_jarvis_turn(self):
        turn = Turn(role="jarvis", content="Good morning.", ts=1001)
        assert turn.role == "jarvis"

    def test_default_timestamp_is_set(self):
        turn = Turn(role="user", content="test")
        assert turn.ts > 0


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------

class TestConversationContext:
    def test_default_construction(self):
        ctx = ConversationContext()
        assert ctx.turns == []
        assert ctx.max_turns == 20

    def test_custom_max_turns(self):
        ctx = ConversationContext(max_turns=5)
        assert ctx.max_turns == 5


# ---------------------------------------------------------------------------
# ActionRequest
# ---------------------------------------------------------------------------

class TestActionRequest:
    def test_basic_construction(self):
        req = ActionRequest(skill_id="open_app", params={"app": "chrome"})
        assert req.skill_id == "open_app"
        assert req.params == {"app": "chrome"}

    def test_empty_params_default(self):
        req = ActionRequest(skill_id="greet")
        assert req.params == {}


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------

class TestActionResult:
    def test_success_constructor(self):
        result = ActionResult.success("Opened Chrome.")
        assert result.is_success
        assert not result.is_failure
        assert result.variant == ActionResultVariant.SUCCESS
        assert result.message == "Opened Chrome."

    def test_failure_constructor(self):
        result = ActionResult.failure("Skill not found: open_app")
        assert result.is_failure
        assert not result.is_success
        assert result.variant == ActionResultVariant.FAILURE
        assert result.message == "Skill not found: open_app"


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_basic_construction(self):
        resp = LLMResponse(text="Here is the answer.", suggestions=["Try this"])
        assert resp.text == "Here is the answer."
        assert resp.suggestions == ["Try this"]

    def test_empty_suggestions_default(self):
        resp = LLMResponse(text="Hello.")
        assert resp.suggestions == []


# ---------------------------------------------------------------------------
# JarvisOutput
# ---------------------------------------------------------------------------

class TestJarvisOutput:
    def test_response_only(self):
        out = JarvisOutput(response="Hello.")
        assert out.response == "Hello."
        assert out.action is None

    def test_response_with_action(self):
        out = JarvisOutput(response="Opening Chrome.", action="open -a 'Google Chrome'")
        assert out.action == "open -a 'Google Chrome'"


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

class TestSkill:
    def test_basic_construction(self):
        def execute(params):
            return ActionResult.success(f"open -a '{params['app']}'")

        skill = Skill(
            id="open_app",
            description="Opens a named application",
            intent_tags=["open_app"],
            required_params=["app"],
            execute=execute,
        )
        assert skill.id == "open_app"
        assert skill.intent_tags == ["open_app"]
        assert skill.required_params == ["app"]

    def test_execute_callable(self):
        skill = Skill(
            id="greet",
            description="Greets the user",
            intent_tags=["greet"],
            required_params=[],
            execute=lambda params: ActionResult.success("Hello!"),
        )
        result = skill.execute({})
        assert result.is_success
        assert result.message == "Hello!"
