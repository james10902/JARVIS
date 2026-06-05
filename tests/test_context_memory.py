"""Unit tests for jarvis/context_memory.py — ContextMemory component.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.4
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from jarvis.context_memory import (
    add_turn,
    get_context,
    load_context,
    save_context,
    validate_turn,
)
from jarvis.models import ConversationContext, Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_turn(role: str = "user", content: str = "hello", ts: int = 0) -> Turn:
    return Turn(role=role, content=content, ts=ts)


def make_ctx(max_turns: int = 5, num_turns: int = 0) -> ConversationContext:
    turns = [make_turn(content=f"msg {i}", ts=i) for i in range(num_turns)]
    return ConversationContext(turns=turns, max_turns=max_turns)


# ---------------------------------------------------------------------------
# validate_turn
# ---------------------------------------------------------------------------

class TestValidateTurn:
    def test_valid_user_turn(self):
        validate_turn(Turn(role="user", content="hello"))  # should not raise

    def test_valid_jarvis_turn(self):
        validate_turn(Turn(role="jarvis", content="Good morning."))  # should not raise

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            validate_turn(Turn(role="system", content="hello"))

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            validate_turn(Turn(role="", content="hello"))

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content"):
            validate_turn(Turn(role="user", content=""))

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValueError, match="content"):
            validate_turn(Turn(role="user", content="   "))


# ---------------------------------------------------------------------------
# add_turn — rolling window
# ---------------------------------------------------------------------------

class TestAddTurnRollingWindow:
    def test_well_under_max_turns(self):
        """Adding a turn when well under the limit just appends."""
        ctx = make_ctx(max_turns=10, num_turns=3)
        result = add_turn(make_turn(content="new"), ctx)
        assert len(result.turns) == 4

    def test_at_exact_boundary(self):
        """Adding a turn when already at max_turns drops the oldest."""
        ctx = make_ctx(max_turns=5, num_turns=5)
        result = add_turn(make_turn(content="new"), ctx)
        assert len(result.turns) == 5

    def test_one_over_boundary(self):
        """Adding a turn when at max_turns keeps exactly max_turns entries."""
        ctx = make_ctx(max_turns=3, num_turns=3)
        result = add_turn(make_turn(content="overflow"), ctx)
        assert len(result.turns) == 3

    def test_oldest_turn_dropped(self):
        """The oldest turn is dropped when the window overflows."""
        ctx = ConversationContext(
            turns=[
                Turn(role="user", content="first", ts=1),
                Turn(role="user", content="second", ts=2),
                Turn(role="user", content="third", ts=3),
            ],
            max_turns=3,
        )
        result = add_turn(Turn(role="user", content="fourth", ts=4), ctx)
        contents = [t.content for t in result.turns]
        assert "first" not in contents
        assert "fourth" in contents

    def test_new_turn_is_last(self):
        """The newly added turn is always the last (most recent) entry."""
        ctx = make_ctx(max_turns=5, num_turns=2)
        new_turn = make_turn(content="newest", ts=999)
        result = add_turn(new_turn, ctx)
        assert result.turns[-1].content == "newest"

    def test_max_turns_unchanged(self):
        """Requirement 6.5: max_turns must not change after add_turn."""
        ctx = make_ctx(max_turns=7, num_turns=7)
        result = add_turn(make_turn(content="x"), ctx)
        assert result.max_turns == 7

    def test_original_context_not_mutated(self):
        """add_turn is a pure function — the original context is unchanged."""
        ctx = make_ctx(max_turns=5, num_turns=3)
        original_len = len(ctx.turns)
        add_turn(make_turn(content="new"), ctx)
        assert len(ctx.turns) == original_len

    def test_empty_context_add_one(self):
        ctx = ConversationContext(turns=[], max_turns=5)
        result = add_turn(make_turn(content="first"), ctx)
        assert len(result.turns) == 1

    def test_max_turns_one(self):
        """With max_turns=1, only the latest turn is ever kept."""
        ctx = ConversationContext(
            turns=[Turn(role="user", content="old", ts=1)],
            max_turns=1,
        )
        result = add_turn(Turn(role="jarvis", content="new", ts=2), ctx)
        assert len(result.turns) == 1
        assert result.turns[0].content == "new"


# ---------------------------------------------------------------------------
# add_turn — validation
# ---------------------------------------------------------------------------

class TestAddTurnValidation:
    def test_invalid_role_rejected(self):
        ctx = make_ctx(max_turns=5)
        with pytest.raises(ValueError, match="role"):
            add_turn(Turn(role="admin", content="hello"), ctx)

    def test_empty_content_rejected(self):
        ctx = make_ctx(max_turns=5)
        with pytest.raises(ValueError, match="content"):
            add_turn(Turn(role="user", content=""), ctx)


# ---------------------------------------------------------------------------
# get_context — chronological ordering
# ---------------------------------------------------------------------------

class TestGetContext:
    def test_returns_turns_in_chronological_order(self):
        """Requirement 6.2: turns are returned oldest-first, newest-last."""
        turns = [
            Turn(role="user", content="first", ts=1),
            Turn(role="jarvis", content="second", ts=2),
            Turn(role="user", content="third", ts=3),
        ]
        ctx = ConversationContext(turns=turns, max_turns=10)
        result = get_context(ctx)
        assert [t.content for t in result] == ["first", "second", "third"]

    def test_empty_context_returns_empty_list(self):
        ctx = ConversationContext(turns=[], max_turns=5)
        assert get_context(ctx) == []

    def test_returns_copy_not_reference(self):
        """Mutating the returned list must not affect the context."""
        ctx = make_ctx(max_turns=5, num_turns=3)
        result = get_context(ctx)
        result.clear()
        assert len(ctx.turns) == 3

    def test_chronological_after_rolling_window(self):
        """After overflow, the remaining turns are still in order."""
        ctx = ConversationContext(turns=[], max_turns=3)
        for i in range(5):
            ctx = add_turn(Turn(role="user", content=f"msg {i}", ts=i), ctx)
        result = get_context(ctx)
        timestamps = [t.ts for t in result]
        assert timestamps == sorted(timestamps)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Persistence — save_context / load_context
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        """Saving and loading a context produces an equivalent object."""
        ctx = ConversationContext(
            turns=[
                Turn(role="user", content="hello", ts=100),
                Turn(role="jarvis", content="hi there", ts=101),
            ],
            max_turns=10,
        )
        file_path = tmp_path / "ctx.json"
        save_context(ctx, path=file_path)
        loaded = load_context(path=file_path)

        assert loaded.max_turns == ctx.max_turns
        assert len(loaded.turns) == len(ctx.turns)
        for orig, restored in zip(ctx.turns, loaded.turns):
            assert restored.role == orig.role
            assert restored.content == orig.content
            assert restored.ts == orig.ts

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_context(path=tmp_path / "missing.json")

    def test_load_invalid_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_context(path=bad_file)

    def test_empty_context_persists(self, tmp_path):
        ctx = ConversationContext(turns=[], max_turns=5)
        file_path = tmp_path / "empty.json"
        save_context(ctx, path=file_path)
        loaded = load_context(path=file_path)
        assert loaded.turns == []
        assert loaded.max_turns == 5


# ---------------------------------------------------------------------------
# Redaction — sensitive values before persistence (Requirement 10.4)
# ---------------------------------------------------------------------------

class TestSensitiveValueRedaction:
    def _save_and_read_raw(self, ctx: ConversationContext, tmp_path: Path) -> dict:
        file_path = tmp_path / "ctx.json"
        save_context(ctx, path=file_path)
        return json.loads(file_path.read_text(encoding="utf-8"))

    def test_password_in_content_is_not_redacted_as_content(self, tmp_path):
        """Turn content itself is not redacted — only dict keys matching patterns."""
        ctx = ConversationContext(
            turns=[Turn(role="user", content="my password is hunter2", ts=1)],
            max_turns=5,
        )
        raw = self._save_and_read_raw(ctx, tmp_path)
        # Content is plain text, not a key-value dict — should be preserved.
        assert raw["turns"][0]["content"] == "my password is hunter2"

    def test_sensitive_key_in_nested_dict_is_redacted(self, tmp_path):
        """Keys matching sensitive patterns in the serialised dict are redacted."""
        from jarvis.context_memory import _redact_sensitive

        data = {
            "password": "s3cr3t",
            "api_key": "sk-abc123",
            "token": "bearer xyz",
            "secret": "topsecret",
            "username": "alice",
        }
        redacted = _redact_sensitive(data)
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["token"] == "[REDACTED]"
        assert redacted["secret"] == "[REDACTED]"
        # Non-sensitive key is preserved.
        assert redacted["username"] == "alice"

    def test_nested_sensitive_keys_are_redacted(self, tmp_path):
        from jarvis.context_memory import _redact_sensitive

        data = {"outer": {"api_key": "abc", "safe": "value"}}
        redacted = _redact_sensitive(data)
        assert redacted["outer"]["api_key"] == "[REDACTED]"
        assert redacted["outer"]["safe"] == "value"

    def test_sensitive_keys_in_list_of_dicts_are_redacted(self, tmp_path):
        from jarvis.context_memory import _redact_sensitive

        data = [{"token": "tok123"}, {"name": "bob"}]
        redacted = _redact_sensitive(data)
        assert redacted[0]["token"] == "[REDACTED]"
        assert redacted[1]["name"] == "bob"

    def test_case_insensitive_key_matching(self, tmp_path):
        from jarvis.context_memory import _redact_sensitive

        data = {"PASSWORD": "abc", "Api_Key": "xyz", "TOKEN": "tok"}
        redacted = _redact_sensitive(data)
        assert redacted["PASSWORD"] == "[REDACTED]"
        assert redacted["Api_Key"] == "[REDACTED]"
        assert redacted["TOKEN"] == "[REDACTED]"

    def test_apikey_without_underscore_is_redacted(self, tmp_path):
        from jarvis.context_memory import _redact_sensitive

        data = {"apikey": "abc123"}
        redacted = _redact_sensitive(data)
        assert redacted["apikey"] == "[REDACTED]"
