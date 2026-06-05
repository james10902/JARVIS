"""Core data models for the JARVIS AI assistant."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """Structured representation of user intent produced by the NLU resolver.

    Attributes:
        tag: Action tag, e.g. "open_app", "create_reminder". Must be non-empty.
        entities: Named entity key-value pairs, e.g. [("app", "chrome")].
        confidence: Confidence score in [0.0, 1.0].
        raw_input: The original, unmodified input string.
    """

    tag: str
    entities: List[tuple[str, str]]
    confidence: float
    raw_input: str


# ---------------------------------------------------------------------------
# Turn & ConversationContext
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation entry.

    Attributes:
        role: Either "user" or "jarvis".
        content: The text content of the turn. Must be non-empty.
        ts: Unix timestamp (integer seconds).
    """

    role: str
    content: str
    ts: int = field(default_factory=lambda: int(time.time()))


@dataclass
class ConversationContext:
    """Rolling window of conversation turns.

    Attributes:
        turns: Ordered list of turns (oldest first, newest last).
        max_turns: Maximum number of turns to retain.
    """

    turns: List[Turn] = field(default_factory=list)
    max_turns: int = 20


# ---------------------------------------------------------------------------
# ActionRequest & ActionResult
# ---------------------------------------------------------------------------

@dataclass
class ActionRequest:
    """Request to execute a registered skill.

    Attributes:
        skill_id: The unique identifier of the target skill.
        params: Key-value parameter map passed to the skill executor.
    """

    skill_id: str
    params: Dict[str, str] = field(default_factory=dict)


class ActionResultVariant(Enum):
    """Discriminator for ActionResult."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class ActionResult:
    """Result of a skill execution.

    Use the class-method constructors :meth:`success` and :meth:`failure`
    rather than instantiating directly.

    Attributes:
        variant: Whether this is a success or failure.
        message: Human-readable confirmation or error description.
    """

    variant: ActionResultVariant
    message: str

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def success(cls, message: str) -> "ActionResult":
        """Create a successful ActionResult."""
        return cls(variant=ActionResultVariant.SUCCESS, message=message)

    @classmethod
    def failure(cls, message: str) -> "ActionResult":
        """Create a failed ActionResult."""
        return cls(variant=ActionResultVariant.FAILURE, message=message)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_success(self) -> bool:
        return self.variant == ActionResultVariant.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.variant == ActionResultVariant.FAILURE


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Response produced by the LLM Response Engine.

    Attributes:
        text: The generated natural language response text.
        suggestions: Optional proactive suggestions JARVIS may offer.
    """

    text: str
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JarvisOutput
# ---------------------------------------------------------------------------

@dataclass
class JarvisOutput:
    """Structured output returned to the user after processing.

    Attributes:
        response: What JARVIS says to the user. Always non-empty.
        action: Optional system command string when an action was dispatched.
    """

    response: str
    action: Optional[str] = None


# ---------------------------------------------------------------------------
# JarvisConfig
# ---------------------------------------------------------------------------

VALID_PERSONALITY_MODES = frozenset({"formal", "casual"})


@dataclass
class JarvisConfig:
    """Configuration controlling JARVIS behaviour.

    Attributes:
        personality_mode: Tone of responses — "formal" or "casual".
        max_context_turns: Rolling memory window size. Must be > 0.
        llm_model: LLM model identifier, e.g. "gpt-4o". Must be non-empty.
        verbose_mode: Whether to emit verbose diagnostic output.
        proactive_suggestions: Whether to include proactive suggestions.

    Raises:
        ValueError: If any validation rule is violated.
    """

    personality_mode: str = "formal"
    max_context_turns: int = 20
    llm_model: str = "gpt-4o"
    verbose_mode: bool = False
    proactive_suggestions: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration fields.

        Raises:
            ValueError: On the first validation failure encountered.
        """
        if self.personality_mode not in VALID_PERSONALITY_MODES:
            raise ValueError(
                f"Invalid personalityMode '{self.personality_mode}'. "
                f"Must be one of: {sorted(VALID_PERSONALITY_MODES)}."
            )
        if self.max_context_turns <= 0:
            raise ValueError(
                f"maxContextTurns must be greater than 0, got {self.max_context_turns}."
            )
        if not self.llm_model or not self.llm_model.strip():
            raise ValueError("llmModel must be a non-empty string.")


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """A registered capability that JARVIS can execute.

    Attributes:
        id: Unique skill identifier.
        description: Human-readable description of what the skill does.
        intent_tags: List of intent tags that map to this skill.
        required_params: Parameter names that must be present in ActionRequest.
        execute: Callable that receives the params dict and returns an ActionResult.
    """

    id: str
    description: str
    intent_tags: List[str]
    required_params: List[str]
    execute: Callable[[Dict[str, str]], ActionResult]
