"""Unit tests for jarvis/skill_registry.py — SkillRegistry.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import pytest

from jarvis.models import ActionResult, Skill
from jarvis.skill_registry import SkillRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    skill_id: str = "test_skill",
    description: str = "A test skill",
    intent_tags: list[str] | None = None,
    required_params: list[str] | None = None,
) -> Skill:
    """Factory for creating minimal Skill instances in tests."""
    return Skill(
        id=skill_id,
        description=description,
        intent_tags=intent_tags if intent_tags is not None else ["test_tag"],
        required_params=required_params if required_params is not None else [],
        execute=lambda params: ActionResult.success("ok"),
    )


# ---------------------------------------------------------------------------
# Registration — basic behaviour
# ---------------------------------------------------------------------------

class TestRegisterBasic:
    def test_register_single_skill(self):
        """Requirement 5.1: registered skill is retrievable by ID."""
        registry = SkillRegistry()
        skill = _make_skill("open_app")
        registry.register(skill)
        assert registry.lookup_by_id("open_app") is skill

    def test_register_multiple_distinct_skills(self):
        """Multiple skills with different IDs can coexist."""
        registry = SkillRegistry()
        s1 = _make_skill("skill_a")
        s2 = _make_skill("skill_b")
        registry.register(s1)
        registry.register(s2)
        assert registry.lookup_by_id("skill_a") is s1
        assert registry.lookup_by_id("skill_b") is s2


# ---------------------------------------------------------------------------
# ID uniqueness — Requirement 5.3
# ---------------------------------------------------------------------------

class TestIdUniqueness:
    def test_second_registration_replaces_first(self):
        """Requirement 5.3: registering a duplicate ID replaces the existing skill."""
        registry = SkillRegistry()
        original = _make_skill("open_app", description="original")
        replacement = _make_skill("open_app", description="replacement")

        registry.register(original)
        registry.register(replacement)

        result = registry.lookup_by_id("open_app")
        assert result is replacement
        assert result.description == "replacement"

    def test_no_error_on_duplicate_id(self):
        """Requirement 5.3: duplicate registration must not raise an error."""
        registry = SkillRegistry()
        registry.register(_make_skill("dup"))
        # Should not raise
        registry.register(_make_skill("dup"))

    def test_only_one_entry_after_duplicate_registration(self):
        """After replacing, only one skill exists for that ID."""
        registry = SkillRegistry()
        registry.register(_make_skill("x"))
        registry.register(_make_skill("x"))
        # lookup_by_tag should return exactly one skill for the shared tag
        results = registry.lookup_by_tag("test_tag")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# requiredParams validation — Requirement 5.4
# ---------------------------------------------------------------------------

class TestRequiredParamsValidation:
    def test_valid_required_params_accepted(self):
        """Well-formed required_params should not raise."""
        registry = SkillRegistry()
        skill = _make_skill(required_params=["app", "mode"])
        registry.register(skill)  # no exception

    def test_empty_string_param_raises(self):
        """Requirement 5.4: empty string in required_params must raise ValueError."""
        registry = SkillRegistry()
        skill = _make_skill(required_params=["app", ""])
        with pytest.raises(ValueError, match="empty"):
            registry.register(skill)

    def test_only_empty_string_param_raises(self):
        """A single empty-string param must also raise."""
        registry = SkillRegistry()
        skill = _make_skill(required_params=[""])
        with pytest.raises(ValueError, match="empty"):
            registry.register(skill)

    def test_duplicate_params_raises(self):
        """Requirement 5.4: duplicate param names must raise ValueError."""
        registry = SkillRegistry()
        skill = _make_skill(required_params=["app", "app"])
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(skill)

    def test_multiple_duplicates_raises(self):
        """More than two duplicates must also raise."""
        registry = SkillRegistry()
        skill = _make_skill(required_params=["a", "b", "a", "b"])
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(skill)

    def test_empty_required_params_list_accepted(self):
        """An empty required_params list is valid."""
        registry = SkillRegistry()
        skill = _make_skill(required_params=[])
        registry.register(skill)  # no exception

    def test_invalid_params_do_not_persist_skill(self):
        """A skill that fails validation must not be stored."""
        registry = SkillRegistry()
        skill = _make_skill("bad_skill", required_params=["x", "x"])
        with pytest.raises(ValueError):
            registry.register(skill)
        assert registry.lookup_by_id("bad_skill") is None


# ---------------------------------------------------------------------------
# lookup_by_id — Requirements 5.1, 5.5
# ---------------------------------------------------------------------------

class TestLookupById:
    def test_returns_none_for_unknown_id(self):
        """Requirement 5.5: lookup for an unregistered ID returns None."""
        registry = SkillRegistry()
        assert registry.lookup_by_id("nonexistent") is None

    def test_returns_none_on_empty_registry(self):
        """Lookup on an empty registry always returns None."""
        registry = SkillRegistry()
        assert registry.lookup_by_id("anything") is None

    def test_returns_correct_skill_by_id(self):
        """Requirement 5.1: lookup returns the exact registered skill."""
        registry = SkillRegistry()
        skill = _make_skill("my_skill")
        registry.register(skill)
        assert registry.lookup_by_id("my_skill") is skill

    def test_does_not_return_wrong_skill(self):
        """Lookup by one ID must not return a skill registered under a different ID."""
        registry = SkillRegistry()
        registry.register(_make_skill("alpha"))
        assert registry.lookup_by_id("beta") is None


# ---------------------------------------------------------------------------
# lookup_by_tag — Requirement 5.2
# ---------------------------------------------------------------------------

class TestLookupByTag:
    def test_returns_skill_with_matching_tag(self):
        """Requirement 5.2: skill is retrievable by its declared intent tag."""
        registry = SkillRegistry()
        skill = _make_skill(intent_tags=["open_app"])
        registry.register(skill)
        results = registry.lookup_by_tag("open_app")
        assert skill in results

    def test_returns_empty_list_for_unknown_tag(self):
        """No skills registered for a tag → empty list returned."""
        registry = SkillRegistry()
        registry.register(_make_skill(intent_tags=["other_tag"]))
        assert registry.lookup_by_tag("unknown_tag") == []

    def test_returns_empty_list_on_empty_registry(self):
        """Lookup by tag on an empty registry returns an empty list."""
        registry = SkillRegistry()
        assert registry.lookup_by_tag("any_tag") == []

    def test_returns_multiple_skills_for_shared_tag(self):
        """Multiple skills sharing a tag are all returned."""
        registry = SkillRegistry()
        s1 = _make_skill("skill_1", intent_tags=["shared"])
        s2 = _make_skill("skill_2", intent_tags=["shared"])
        registry.register(s1)
        registry.register(s2)
        results = registry.lookup_by_tag("shared")
        assert s1 in results
        assert s2 in results
        assert len(results) == 2

    def test_skill_with_multiple_tags_found_by_each(self):
        """Requirement 5.2: a skill with multiple tags is found by each of them."""
        registry = SkillRegistry()
        skill = _make_skill(intent_tags=["tag_a", "tag_b", "tag_c"])
        registry.register(skill)
        assert skill in registry.lookup_by_tag("tag_a")
        assert skill in registry.lookup_by_tag("tag_b")
        assert skill in registry.lookup_by_tag("tag_c")

    def test_does_not_return_skill_for_non_declared_tag(self):
        """A skill must not appear in results for a tag it did not declare."""
        registry = SkillRegistry()
        skill = _make_skill(intent_tags=["real_tag"])
        registry.register(skill)
        assert skill not in registry.lookup_by_tag("other_tag")

    def test_replaced_skill_found_by_new_tags(self):
        """After replacement, lookup_by_tag reflects the new skill's tags."""
        registry = SkillRegistry()
        original = _make_skill("s", intent_tags=["old_tag"])
        replacement = _make_skill("s", intent_tags=["new_tag"])
        registry.register(original)
        registry.register(replacement)
        assert replacement in registry.lookup_by_tag("new_tag")
        assert original not in registry.lookup_by_tag("old_tag")
