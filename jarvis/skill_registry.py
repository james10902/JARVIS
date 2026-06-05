"""Skill Registry for the JARVIS AI assistant.

Stores registered skills and provides lookup by ID and intent tag.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

from typing import Dict, List, Optional

from jarvis.models import Skill


class SkillRegistry:
    """Mutable registry of JARVIS skills.

    Skills are stored by their unique ID. Registering a skill with an
    existing ID replaces the previous entry (Requirement 5.3).

    Attributes:
        _skills: Internal mapping from skill ID to Skill instance.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a skill, replacing any existing skill with the same ID.

        Validates the skill's ``required_params`` before storing:
        - No empty-string parameter names.
        - No duplicate parameter names.

        Args:
            skill: The :class:`~jarvis.models.Skill` to register.

        Raises:
            ValueError: If ``required_params`` contains an empty string or
                duplicate parameter names.

        Requirements: 5.1, 5.3, 5.4
        """
        self._validate_required_params(skill.required_params)
        self._skills[skill.id] = skill

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_by_id(self, skill_id: str) -> Optional[Skill]:
        """Return the skill with the given ID, or ``None`` if not found.

        Args:
            skill_id: The unique identifier to look up.

        Returns:
            The matching :class:`~jarvis.models.Skill`, or ``None``.

        Requirements: 5.1, 5.5
        """
        return self._skills.get(skill_id)

    def lookup_by_tag(self, tag: str) -> List[Skill]:
        """Return all skills whose ``intent_tags`` includes *tag*.

        Args:
            tag: The intent tag to search for.

        Returns:
            A list of matching :class:`~jarvis.models.Skill` instances
            (may be empty if no skills declare the given tag).

        Requirements: 5.2
        """
        return [skill for skill in self._skills.values() if tag in skill.intent_tags]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required_params(required_params: List[str]) -> None:
        """Validate a ``required_params`` list.

        Args:
            required_params: The list to validate.

        Raises:
            ValueError: If any entry is an empty string or if there are
                duplicate entries.

        Requirements: 5.4
        """
        for param in required_params:
            if param == "":
                raise ValueError(
                    "required_params must not contain empty strings."
                )

        if len(required_params) != len(set(required_params)):
            raise ValueError(
                "required_params must not contain duplicate parameter names."
            )
