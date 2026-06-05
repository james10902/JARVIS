"""Action Dispatcher for the JARVIS AI assistant.

Resolves and executes skills from the registry, returning structured
ActionResult values. Never propagates exceptions to the caller.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.1
"""

from __future__ import annotations

import re

from jarvis.models import ActionRequest, ActionResult
from jarvis.skill_registry import SkillRegistry

# Shell metacharacters that could enable command injection.
# Strip or escape: ; | & $ ` ( ) < > newline carriage-return
_SHELL_METACHAR_RE = re.compile(r"[;|&$`()<>\r\n]")


def _sanitize(value: str) -> str:
    """Remove shell metacharacters from an entity value.

    Strips the characters ``;``, ``|``, ``&``, ``$``, `` ` ``, ``(``,
    ``)``, ``<``, ``>``, ``\\n``, and ``\\r`` to prevent command injection
    when the value is later embedded in a shell command.

    Args:
        value: The raw entity value to sanitize.

    Returns:
        The sanitized string with all shell metacharacters removed.

    Requirements: 10.1
    """
    return _SHELL_METACHAR_RE.sub("", value)


def dispatch(req: ActionRequest, registry: SkillRegistry) -> ActionResult:
    """Resolve and execute a skill, returning a structured ActionResult.

    Steps:
    1. Look up the skill by ``req.skill_id``; return a Failure if absent.
    2. Validate that all ``required_params`` are present in ``req.params``;
       return a Failure listing every missing name if any are absent.
    3. Sanitize all param values to prevent command injection.
    4. Execute the skill, catching any exception and returning it as a Failure.

    This function never raises — it always returns an :class:`ActionResult`.

    Args:
        req: The :class:`~jarvis.models.ActionRequest` describing which skill
            to run and with what parameters.
        registry: The :class:`~jarvis.skill_registry.SkillRegistry` to look
            up the skill from.

    Returns:
        An :class:`~jarvis.models.ActionResult` — either Success or Failure.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.1
    """
    # Step 1: Skill lookup (Requirement 4.1, 4.2)
    skill = registry.lookup_by_id(req.skill_id)
    if skill is None:
        return ActionResult.failure(f"Skill not found: {req.skill_id}")

    # Step 2: Required-param validation (Requirement 4.3)
    missing = [p for p in skill.required_params if p not in req.params]
    if missing:
        return ActionResult.failure(f"Missing params: {', '.join(missing)}")

    # Step 3: Sanitize entity values to prevent command injection (Requirement 10.1)
    sanitized_params = {k: _sanitize(v) for k, v in req.params.items()}

    # Step 4: Execute with exception guard (Requirement 4.4, 4.5)
    try:
        return skill.execute(sanitized_params)
    except Exception as exc:  # noqa: BLE001
        return ActionResult.failure(str(exc))
