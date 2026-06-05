"""NLU & Intent Resolver for the JARVIS AI assistant.

Uses a single, fast Groq call to classify intent.
The creator profile is intentionally excluded here — the NLU prompt must
stay short and JSON-only to keep latency low.

Requirements: 1.1, 1.2, 1.3, 1.4, 10.3
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional

from jarvis.models import ConversationContext, Intent

# ---------------------------------------------------------------------------
# Prompt — short and laser-focused on JSON output only
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an intent classifier. Return ONLY a valid JSON object:
{
  "tag": "<snake_case intent tag>",
  "entities": {"<key>": "<value>"},
  "confidence": <0.0-1.0>
}
Common tags: open_app, open_file, find_file, explain_concept, general_question,
sleep_computer, shutdown_computer, restart_computer, create_reminder, greeting.
No markdown. No explanation. JSON only.\
"""


def _build_messages(input_str: str, ctx: ConversationContext) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    # Only last 4 turns for pronoun disambiguation — keeps prompt tiny
    for turn in ctx.turns[-4:]:
        role = "user" if turn.role == "user" else "assistant"
        messages.append({"role": role, "content": turn.content})
    messages.append({"role": "user", "content": input_str})
    return messages


def _default_llm_caller(messages: list[dict]) -> str:
    from jarvis.groq_client import chat
    # Use the smaller, faster model for classification only
    return chat(messages, model="llama-3.1-8b-instant")


def resolve_intent(
    input_str: str,
    ctx: ConversationContext,
    llm_caller: Optional[Callable[[list[dict]], str]] = None,
) -> Intent:
    if not input_str or not input_str.strip():
        raise ValueError("Input string must be non-empty.")

    caller = llm_caller if llm_caller is not None else _default_llm_caller
    messages = _build_messages(input_str, ctx)

    try:
        raw_response = caller(messages)
        data = json.loads(raw_response)

        tag: str = str(data.get("tag", "")).strip() or "unknown"

        raw_entities = data.get("entities", {})
        entities: List[tuple[str, str]] = (
            [(str(k), str(v)) for k, v in raw_entities.items()]
            if isinstance(raw_entities, dict) else []
        )

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

    except Exception as _nlu_exc:
        import traceback
        print(f"[NLU ERROR] {type(_nlu_exc).__name__}: {_nlu_exc}")
        traceback.print_exc()
        return Intent(tag="unknown", entities=[], confidence=0.0, raw_input=input_str)

    return Intent(tag=tag, entities=entities, confidence=confidence, raw_input=input_str)
