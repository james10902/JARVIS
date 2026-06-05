"""Response Formatter for the JARVIS AI assistant.

Assembles the final JarvisOutput by combining an optional LLM response
with an optional ActionResult.  This is a pure function with no side effects.
"""

from __future__ import annotations

from typing import Optional

from jarvis.models import ActionResult, ActionResultVariant, JarvisOutput, LLMResponse


def format_output(
    llm_resp: Optional[LLMResponse],
    act_result: Optional[ActionResult],
) -> JarvisOutput:
    """Assemble a JarvisOutput from an optional LLM response and action result.

    At least one of *llm_resp* or *act_result* must be provided (not both None).

    Combination rules
    -----------------
    - LLM response **and** ``ActionResult.Success``:
        ``response`` = LLM text, ``action`` = success message (command string)
    - LLM response **only** (no action result, or action result is Failure):
        ``response`` = LLM text, ``action`` = None
    - ``ActionResult.Failure`` **only** (no LLM response):
        ``response`` = failure message, ``action`` = None
    - ``ActionResult.Success`` **only** (no LLM response):
        ``response`` = success message, ``action`` = success message

    Parameters
    ----------
    llm_resp:
        The LLM-generated response, or ``None`` if the pipeline did not
        produce a conversational reply.
    act_result:
        The result of skill execution, or ``None`` if no action was dispatched.

    Returns
    -------
    JarvisOutput
        A structured output whose ``response`` field is always non-empty.

    Raises
    ------
    ValueError
        If both *llm_resp* and *act_result* are ``None``.
    """
    if llm_resp is None and act_result is None:
        raise ValueError(
            "format_output requires at least one of llm_resp or act_result to be provided."
        )

    # Case 1: LLM response present
    if llm_resp is not None:
        response_text = llm_resp.text

        # Pair with a successful action result → include action field
        if act_result is not None and act_result.variant == ActionResultVariant.SUCCESS:
            return JarvisOutput(response=response_text, action=act_result.message)

        # LLM only (no action, or action failed — failure info is not surfaced
        # here because the LLM response already covers the user-facing reply)
        return JarvisOutput(response=response_text, action=None)

    # Case 2: No LLM response — act_result must be present (guaranteed above)
    assert act_result is not None  # type narrowing for static analysers

    if act_result.variant == ActionResultVariant.SUCCESS:
        # Success only: echo the command string as both response and action
        return JarvisOutput(response=act_result.message, action=act_result.message)

    # Failure only: surface the failure message as the response
    return JarvisOutput(response=act_result.message, action=None)
