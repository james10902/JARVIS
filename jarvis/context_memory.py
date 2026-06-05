"""Context Memory component for the JARVIS AI assistant.

Maintains a rolling window of conversation turns, persists context to a JSON
file backend, and redacts sensitive values before writing to storage.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.4
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import List

from jarvis.models import ConversationContext, Turn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLES = frozenset({"user", "jarvis"})

# Patterns whose values should be redacted before persistence.
# Matches keys containing these substrings (case-insensitive).
_SENSITIVE_KEY_PATTERNS = re.compile(
    r"(password|api_key|apikey|token|secret)",
    re.IGNORECASE,
)

# Default path for the JSON persistence file.
DEFAULT_CONTEXT_FILE = Path("jarvis_context.json")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_turn(turn: Turn) -> None:
    """Validate a Turn before it is added to the context.

    Args:
        turn: The Turn to validate.

    Raises:
        ValueError: If the role is not "user" or "jarvis", or if the content
                    is empty or whitespace-only.
    """
    if turn.role not in VALID_ROLES:
        raise ValueError(
            f"Invalid turn role '{turn.role}'. Must be one of: {sorted(VALID_ROLES)}."
        )
    if not turn.content or not turn.content.strip():
        raise ValueError("Turn content must be non-empty.")


# ---------------------------------------------------------------------------
# Core operations (pure / immutable style)
# ---------------------------------------------------------------------------

def add_turn(turn: Turn, ctx: ConversationContext) -> ConversationContext:
    """Add a Turn to the context, enforcing the rolling window.

    This is a pure function — it returns a *new* ConversationContext and
    leaves the original unchanged.

    The rolling window is enforced by dropping the oldest turn(s) whenever
    the total count would exceed ``ctx.max_turns``.

    Args:
        turn: The Turn to append. Must pass :func:`validate_turn`.
        ctx:  The current ConversationContext.

    Returns:
        A new ConversationContext with the turn appended and the window
        enforced.  ``max_turns`` is identical to the input context's value.

    Raises:
        ValueError: If the turn fails validation.
    """
    validate_turn(turn)

    new_turns: List[Turn] = list(ctx.turns) + [turn]

    # Enforce rolling window: keep only the most recent max_turns entries.
    if len(new_turns) > ctx.max_turns:
        new_turns = new_turns[len(new_turns) - ctx.max_turns:]

    return ConversationContext(turns=new_turns, max_turns=ctx.max_turns)


def get_context(ctx: ConversationContext) -> List[Turn]:
    """Return turns in chronological order (oldest first, newest last).

    Args:
        ctx: The ConversationContext to query.

    Returns:
        A list of Turn objects in chronological order.
    """
    return list(ctx.turns)


# ---------------------------------------------------------------------------
# Sensitive-value redaction
# ---------------------------------------------------------------------------

def _redact_sensitive(data: object) -> object:
    """Recursively redact sensitive values in a JSON-serialisable structure.

    Any dict key whose name matches a sensitive pattern (password, api_key,
    token, secret, …) has its value replaced with ``"[REDACTED]"``.

    Args:
        data: A JSON-serialisable Python object (dict, list, str, int, …).

    Returns:
        A deep copy of *data* with sensitive values replaced.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(key, str) and _SENSITIVE_KEY_PATTERNS.search(key):
                result[key] = "[REDACTED]"
            else:
                result[key] = _redact_sensitive(value)
        return result
    if isinstance(data, list):
        return [_redact_sensitive(item) for item in data]
    return data


def _context_to_dict(ctx: ConversationContext) -> dict:
    """Serialise a ConversationContext to a plain dict."""
    return {
        "max_turns": ctx.max_turns,
        "turns": [
            {"role": t.role, "content": t.content, "ts": t.ts}
            for t in ctx.turns
        ],
    }


def _context_from_dict(data: dict) -> ConversationContext:
    """Deserialise a ConversationContext from a plain dict."""
    turns = [
        Turn(role=t["role"], content=t["content"], ts=t["ts"])
        for t in data.get("turns", [])
    ]
    return ConversationContext(turns=turns, max_turns=data.get("max_turns", 20))


# ---------------------------------------------------------------------------
# Persistence backend
# ---------------------------------------------------------------------------

def save_context(
    ctx: ConversationContext,
    path: Path = DEFAULT_CONTEXT_FILE,
) -> None:
    """Persist the ConversationContext to a JSON file.

    Sensitive values (passwords, API keys, tokens, secrets) are redacted
    before writing.  The original ``ctx`` object is not modified.

    Args:
        ctx:  The ConversationContext to persist.
        path: Destination file path (default: ``jarvis_context.json``).
    """
    raw = _context_to_dict(ctx)
    safe = _redact_sensitive(raw)
    path = Path(path)
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")


def load_context(path: Path = DEFAULT_CONTEXT_FILE) -> ConversationContext:
    """Load a ConversationContext from a JSON file.

    Args:
        path: Source file path (default: ``jarvis_context.json``).

    Returns:
        The deserialised ConversationContext.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or an unexpected schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in context file '{path}': {exc}") from exc
    return _context_from_dict(data)
