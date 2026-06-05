"""End-to-end processing pipeline for the JARVIS AI assistant.

Wires together all components: NLU → Router → LLM/Dispatcher → Formatter.
Persists interactions to context memory after producing output.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 10.2, 11.2, 11.3, 11.4
"""

from __future__ import annotations

from typing import Callable, Optional

from jarvis.context_memory import add_turn
from jarvis.dispatcher import dispatch
from jarvis.formatter import format_output
from jarvis.llm_engine import generate_response
from jarvis.models import (
    ActionRequest,
    ConversationContext,
    JarvisConfig,
    JarvisOutput,
    Turn,
)
from jarvis.nlu import resolve_intent
from jarvis.router import RouteDecision, route
from jarvis.skill_registry import SkillRegistry


def process_input(
    input_str: str,
    ctx: ConversationContext,
    registry: SkillRegistry,
    config: JarvisConfig,
    nlu_caller: Optional[Callable[[list[dict]], str]] = None,
    llm_caller: Optional[Callable[[list[dict], str], str]] = None,
) -> tuple[JarvisOutput, ConversationContext]:
    """Process user input through the full JARVIS pipeline.

    This is the main entry point for the JARVIS system. It orchestrates:
    1. Intent resolution (NLU)
    2. Routing (conversational vs actionable)
    3. Response generation (LLM) or action dispatch (skills)
    4. Output formatting
    5. Context persistence

    The function returns a tuple of (JarvisOutput, updated ConversationContext)
    following a pure functional style — the input context is not mutated.

    Low-confidence handling (Requirement 8.4):
        When ``intent.confidence < 0.5``, returns a clarification question
        with ``action=None`` and persists the turn.

    Conversational intents (Requirement 8.3):
        Routed to the LLM engine; output has ``action=None``.

    Actionable intents (Requirement 8.2):
        Routed to the dispatcher; successful dispatch sets ``action`` to the
        command string from ``ActionResult.Success``.

    Error recovery (Requirements 11.2, 11.3, 11.4):
        - Skill not found: dispatcher returns a Failure with a user-friendly
          message; formatter surfaces it in the response.
        - Missing params: dispatcher returns a Failure listing missing names;
          formatter surfaces it in the response.
        - System-level errors: dispatcher catches exceptions and returns
          Failure; formatter surfaces it in the response.

    Args:
        input_str: The raw user input string. Must be non-empty.
        ctx: The current ConversationContext.
        registry: The SkillRegistry containing all registered skills.
        config: The JarvisConfig controlling personality and features.
        nlu_caller: Optional injectable LLM caller for NLU (testing).
        llm_caller: Optional injectable LLM caller for response generation (testing).

    Returns:
        A tuple of (JarvisOutput, updated ConversationContext). The output
        always has a non-empty ``response`` field. The context includes the
        new user and JARVIS turns.

    Raises:
        ValueError: If ``input_str`` is empty (propagated from resolve_intent).

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 10.2, 11.2, 11.3, 11.4
    """
    # Step 1: Resolve intent (Requirement 8.1)
    intent = resolve_intent(input_str, ctx, llm_caller=nlu_caller)

    # Step 2: Low-confidence check (Requirement 8.4)
    if intent.confidence < 0.5:
        output = JarvisOutput(
            response="Could you clarify what you'd like me to do?",
            action=None,
        )
        # Persist the interaction before returning
        ctx = _persist_interaction(input_str, output, ctx)
        return output, ctx

    # Step 3: Derive actionable tag set from registry (design constraint)
    actionable_tags = {
        tag for skill in registry._skills.values() for tag in skill.intent_tags
    }

    # Step 4: Route the intent (Requirement 8.1)
    decision = route(intent, actionable_tags)

    # Step 5: Generate response or dispatch action
    if decision == RouteDecision.CONVERSATIONAL:
        # Conversational path (Requirement 8.3)
        llm_resp = generate_response(intent, ctx, config, llm_caller=llm_caller)
        output = format_output(llm_resp, None)
    else:
        # Actionable path (Requirement 8.2)
        # Resolve skill ID from intent tag — tag ≠ skill ID
        params = dict(intent.entities)
        matched_skills = registry.lookup_by_tag(intent.tag)
        if matched_skills:
            skill_id = matched_skills[0].id
        else:
            skill_id = intent.tag   # fallback: let dispatcher return "not found"
        req = ActionRequest(skill_id=skill_id, params=params)
        act_result = dispatch(req, registry)
        # Format the action result (Requirements 11.2, 11.3, 11.4)
        output = format_output(None, act_result)

    # Step 6: Persist the interaction (Requirement 8.5)
    ctx = _persist_interaction(input_str, output, ctx)

    return output, ctx


def _persist_interaction(
    user_input: str,
    output: JarvisOutput,
    ctx: ConversationContext,
) -> ConversationContext:
    """Persist both the user input and JARVIS response as Turns.

    Args:
        user_input: The raw user input string.
        output: The JarvisOutput produced by the pipeline.
        ctx: The current ConversationContext.

    Returns:
        The updated ConversationContext with both turns added.

    Requirements: 8.5
    """
    # Add user turn
    user_turn = Turn(role="user", content=user_input)
    ctx = add_turn(user_turn, ctx)

    # Add JARVIS turn
    jarvis_turn = Turn(role="jarvis", content=output.response)
    ctx = add_turn(jarvis_turn, ctx)

    return ctx
