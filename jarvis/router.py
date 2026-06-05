"""Command Router for the JARVIS AI assistant.

Classifies an Intent as either ACTIONABLE (dispatch to a skill) or
CONVERSATIONAL (send to the LLM engine).  The router is a pure function
with no side effects — same inputs always produce the same output.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Set, Union

from jarvis.models import Intent


class RouteDecision(Enum):
    """Possible routing outcomes for an Intent."""

    ACTIONABLE = "actionable"
    CONVERSATIONAL = "conversational"


def route(
    intent: Intent,
    actionable_tags: Union[Set[str], FrozenSet[str]],
) -> RouteDecision:
    """Classify an intent as actionable or conversational.

    The confidence check takes priority: if ``intent.confidence < 0.5`` the
    intent is always routed to the LLM engine for clarification, regardless
    of the tag.

    Args:
        intent: The structured intent produced by the NLU resolver.
        actionable_tags: The set of intent tags that map to registered skills.

    Returns:
        ``RouteDecision.ACTIONABLE`` when the intent tag is in
        ``actionable_tags`` *and* confidence is >= 0.5.
        ``RouteDecision.CONVERSATIONAL`` in all other cases.
    """
    # Requirement 2.3 — low-confidence intents always go to LLM
    if intent.confidence < 0.5:
        return RouteDecision.CONVERSATIONAL

    # Requirement 2.1 — known actionable tag → dispatch
    if intent.tag in actionable_tags:
        return RouteDecision.ACTIONABLE

    # Requirement 2.2 — unknown tag → conversational
    return RouteDecision.CONVERSATIONAL
