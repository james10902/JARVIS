"""Unit tests for jarvis/dispatcher.py.

Covers:
- Skill-not-found returns correct Failure message format
- Missing required params returns Failure listing all missing names
- Exception in skill.execute is caught and returned as Failure
- Successful dispatch returns Success with command string
- Command injection sanitization on entity values

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.1
"""

from __future__ import annotations

import pytest

from jarvis.dispatcher import _sanitize, dispatch
from jarvis.models import ActionRequest, ActionResult, ActionResultVariant, Skill
from jarvis.skill_registry import SkillRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*skills: Skill) -> SkillRegistry:
    """Build a SkillRegistry pre-populated with the given skills."""
    reg = SkillRegistry()
    for skill in skills:
        reg.register(skill)
    return reg


def _echo_skill(skill_id: str, required_params: list[str]) -> Skill:
    """Return a skill that echoes its params as a success message."""
    return Skill(
        id=skill_id,
        description="Echo skill for testing",
        intent_tags=[skill_id],
        required_params=required_params,
        execute=lambda params: ActionResult.success(
            " ".join(f"{k}={v}" for k, v in sorted(params.items()))
        ),
    )


def _raising_skill(skill_id: str) -> Skill:
    """Return a skill whose execute always raises RuntimeError."""
    def _execute(params):  # noqa: ANN001
        raise RuntimeError("boom")

    return Skill(
        id=skill_id,
        description="Always-failing skill for testing",
        intent_tags=[skill_id],
        required_params=[],
        execute=_execute,
    )


# ---------------------------------------------------------------------------
# Requirement 4.2 — Skill not found
# ---------------------------------------------------------------------------

class TestSkillNotFound:
    """dispatch returns Failure with the correct message when skill is absent."""

    def test_failure_variant(self):
        """Result is a Failure when skill_id is not in registry."""
        reg = _make_registry()
        req = ActionRequest(skill_id="unknown_skill")
        result = dispatch(req, reg)
        assert result.is_failure

    def test_failure_message_format(self):
        """Failure message matches 'Skill not found: <id>' exactly."""
        reg = _make_registry()
        req = ActionRequest(skill_id="open_app")
        result = dispatch(req, reg)
        assert result.message == "Skill not found: open_app"

    def test_failure_message_includes_skill_id(self):
        """Failure message contains the requested skill ID."""
        reg = _make_registry()
        skill_id = "my_custom_skill_xyz"
        req = ActionRequest(skill_id=skill_id)
        result = dispatch(req, reg)
        assert skill_id in result.message


# ---------------------------------------------------------------------------
# Requirement 4.3 — Missing required params
# ---------------------------------------------------------------------------

class TestMissingRequiredParams:
    """dispatch returns Failure listing all missing param names."""

    def test_single_missing_param(self):
        """Failure message lists the one missing param."""
        skill = _echo_skill("open_app", required_params=["app"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_app", params={})
        result = dispatch(req, reg)
        assert result.is_failure
        assert "app" in result.message

    def test_multiple_missing_params(self):
        """Failure message lists all missing params when several are absent."""
        skill = _echo_skill("create_event", required_params=["title", "date", "time"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="create_event", params={"title": "Meeting"})
        result = dispatch(req, reg)
        assert result.is_failure
        assert "date" in result.message
        assert "time" in result.message

    def test_message_prefix(self):
        """Failure message starts with 'Missing params:'."""
        skill = _echo_skill("open_app", required_params=["app"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_app", params={})
        result = dispatch(req, reg)
        assert result.message.startswith("Missing params:")

    def test_all_params_present_does_not_fail(self):
        """No missing-params failure when all required params are supplied."""
        skill = _echo_skill("open_app", required_params=["app"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_app", params={"app": "chrome"})
        result = dispatch(req, reg)
        # Should not be a missing-params failure
        assert not (result.is_failure and result.message.startswith("Missing params:"))


# ---------------------------------------------------------------------------
# Requirement 4.4 / 4.5 — Exception handling
# ---------------------------------------------------------------------------

class TestExceptionHandling:
    """Exceptions from skill.execute are caught and returned as Failure."""

    def test_exception_returns_failure(self):
        """RuntimeError in execute becomes ActionResult.Failure."""
        skill = _raising_skill("bad_skill")
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="bad_skill")
        result = dispatch(req, reg)
        assert result.is_failure

    def test_exception_message_in_failure(self):
        """The exception message is included in the Failure message."""
        skill = _raising_skill("bad_skill")
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="bad_skill")
        result = dispatch(req, reg)
        assert "boom" in result.message

    def test_no_exception_propagated(self):
        """dispatch never raises even when skill.execute raises."""
        skill = _raising_skill("bad_skill")
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="bad_skill")
        # Should not raise
        result = dispatch(req, reg)
        assert isinstance(result, ActionResult)

    def test_arbitrary_exception_type_caught(self):
        """Any exception type (not just RuntimeError) is caught."""
        def _execute(params):  # noqa: ANN001
            raise ValueError("bad value")

        skill = Skill(
            id="val_err_skill",
            description="Raises ValueError",
            intent_tags=["val_err_skill"],
            required_params=[],
            execute=_execute,
        )
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="val_err_skill")
        result = dispatch(req, reg)
        assert result.is_failure
        assert "bad value" in result.message


# ---------------------------------------------------------------------------
# Requirement 4.1 — Successful dispatch
# ---------------------------------------------------------------------------

class TestSuccessfulDispatch:
    """Successful dispatch returns ActionResult.Success with command string."""

    def test_success_variant(self):
        """Result is Success when skill executes without error."""
        skill = _echo_skill("open_app", required_params=["app"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_app", params={"app": "chrome"})
        result = dispatch(req, reg)
        assert result.is_success

    def test_success_message_contains_param(self):
        """Success message reflects the executed command."""
        skill = _echo_skill("open_app", required_params=["app"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_app", params={"app": "chrome"})
        result = dispatch(req, reg)
        assert "chrome" in result.message

    def test_extra_params_passed_through(self):
        """Extra params beyond required ones are passed to execute."""
        skill = _echo_skill("open_app", required_params=["app"])
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_app", params={"app": "chrome", "mode": "incognito"})
        result = dispatch(req, reg)
        assert result.is_success
        assert "incognito" in result.message

    def test_skill_with_no_required_params(self):
        """Skill with empty required_params dispatches successfully."""
        skill = Skill(
            id="ping",
            description="Ping skill",
            intent_tags=["ping"],
            required_params=[],
            execute=lambda params: ActionResult.success("pong"),
        )
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="ping")
        result = dispatch(req, reg)
        assert result.is_success
        assert result.message == "pong"


# ---------------------------------------------------------------------------
# Requirement 10.1 — Command injection sanitization
# ---------------------------------------------------------------------------

class TestSanitization:
    """Entity values are sanitized before being passed to skill.execute."""

    # Test _sanitize directly for each metacharacter
    @pytest.mark.parametrize("char", [";", "|", "&", "$", "`", "(", ")", "<", ">", "\n", "\r"])
    def test_sanitize_removes_metachar(self, char: str):
        """_sanitize strips the given shell metacharacter."""
        dirty = f"safe{char}value"
        clean = _sanitize(dirty)
        assert char not in clean
        assert "safe" in clean
        assert "value" in clean

    def test_sanitize_preserves_normal_text(self):
        """_sanitize does not alter alphanumeric text."""
        value = "Google Chrome"
        assert _sanitize(value) == value

    def test_sanitize_preserves_path_separators(self):
        """_sanitize keeps forward slashes and dots (common in file paths)."""
        value = "/home/user/documents/file.txt"
        assert _sanitize(value) == value

    def test_dispatch_sanitizes_before_execute(self):
        """Injected metacharacters are stripped before reaching skill.execute."""
        received: dict[str, str] = {}

        def _capture(params):  # noqa: ANN001
            received.update(params)
            return ActionResult.success("ok")

        skill = Skill(
            id="open_app",
            description="Capture params",
            intent_tags=["open_app"],
            required_params=["app"],
            execute=_capture,
        )
        reg = _make_registry(skill)
        # Inject a semicolon and pipe into the app name
        req = ActionRequest(skill_id="open_app", params={"app": "chrome; rm -rf /"})
        dispatch(req, reg)
        assert ";" not in received.get("app", "")
        assert "|" not in received.get("app", "")

    def test_dispatch_sanitizes_newline_injection(self):
        """Newline characters in param values are stripped."""
        received: dict[str, str] = {}

        def _capture(params):  # noqa: ANN001
            received.update(params)
            return ActionResult.success("ok")

        skill = Skill(
            id="open_file",
            description="Capture params",
            intent_tags=["open_file"],
            required_params=["path"],
            execute=_capture,
        )
        reg = _make_registry(skill)
        req = ActionRequest(skill_id="open_file", params={"path": "/tmp/file\nmalicious"})
        dispatch(req, reg)
        assert "\n" not in received.get("path", "")

    def test_dispatch_sanitizes_all_params(self):
        """Sanitization is applied to every param, not just the first."""
        received: dict[str, str] = {}

        def _capture(params):  # noqa: ANN001
            received.update(params)
            return ActionResult.success("ok")

        skill = Skill(
            id="multi_param",
            description="Multi-param skill",
            intent_tags=["multi_param"],
            required_params=["a", "b"],
            execute=_capture,
        )
        reg = _make_registry(skill)
        req = ActionRequest(
            skill_id="multi_param",
            params={"a": "val;a", "b": "val|b"},
        )
        dispatch(req, reg)
        assert ";" not in received.get("a", "")
        assert "|" not in received.get("b", "")
