# Implementation Plan: JARVIS AI Assistant

## Overview

Implement the JARVIS AI assistant in Python using a modular pipeline architecture. The implementation follows the design's component breakdown: NLU & Intent Resolver → Command Router → LLM Engine / Action Dispatcher → Response Formatter, with Context Memory and Skill Registry as supporting components. All components are implemented as pure functions or lightweight classes to match the formal specifications in the design.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create the `jarvis/` package directory with `__init__.py`
  - Create `jarvis/models.py` defining all core dataclasses: `Intent`, `Turn`, `ConversationContext`, `ActionRequest`, `ActionResult`, `LLMResponse`, `JarvisOutput`, `JarvisConfig`, `Skill`
  - Implement `JarvisConfig` validation (reject invalid `personalityMode`, `maxContextTurns <= 0`, empty `llmModel`)
  - Set up `pyproject.toml` or `requirements.txt` with dependencies (openai, pytest, hypothesis)
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 2. Implement Context Memory
  - [x] 2.1 Implement `ContextMemory` class in `jarvis/context_memory.py`
    - Implement `add_turn(turn, ctx)` enforcing rolling window (drop oldest when `len > maxTurns`)
    - Implement `get_context(ctx)` returning turns in chronological order
    - Implement `validate_turn` to reject turns with invalid role or empty content
    - Implement file/SQLite persistence backend (`save_context`, `load_context`) with sensitive value redaction
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.4_

  - [ ]* 2.2 Write property test for `add_turn` rolling window bound
    - **Property 3: addTurn preserves rolling window bound**
    - **Validates: Requirements 6.1, 6.5**
    - Use `hypothesis` to generate arbitrary `Turn` and `ConversationContext` inputs
    - Assert `len(result.turns) <= ctx.max_turns` after every `add_turn` call

  - [ ]* 2.3 Write unit tests for `ContextMemory`
    - Test rolling window at exact boundary (maxTurns), one over, and well under
    - Test chronological ordering of returned turns
    - Test validation rejection for empty content and invalid role
    - Test redaction of passwords and API keys before persistence
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.4_

- [x] 3. Implement Skill Registry
  - [x] 3.1 Implement `SkillRegistry` class in `jarvis/skill_registry.py`
    - Implement `register(skill)` with unique-ID enforcement (replace on duplicate) and `requiredParams` validation (no duplicates, no empty strings)
    - Implement `lookup_by_id(skill_id)` returning `Optional[Skill]`
    - Implement `lookup_by_tag(tag)` returning `List[Skill]`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 3.2 Write property test for skill lookup consistency after registration
    - **Property 6: Skill lookup by ID is consistent with registration**
    - **Validates: Requirements 5.1**
    - Use `hypothesis` to generate arbitrary `Skill` objects and assert `lookup_by_id(skill.id)` returns the registered skill

  - [ ]* 3.3 Write property test for skill lookup by tag consistency
    - **Property 9: Skill lookup by tag is consistent with registration**
    - **Validates: Requirements 5.2**
    - Use `hypothesis` to generate arbitrary `Skill` objects and assert each declared intent tag returns the skill in `lookup_by_tag`

  - [ ]* 3.4 Write unit tests for `SkillRegistry`
    - Test ID uniqueness enforcement (second registration replaces first)
    - Test `requiredParams` validation rejects duplicates and empty strings
    - Test `lookup_by_id` returns `None` for unknown IDs
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 4. Implement Command Router
  - [x] 4.1 Implement `route(intent, actionable_tags)` in `jarvis/router.py`
    - Return `RouteDecision.ACTIONABLE` when `intent.tag` is in the actionable tag registry
    - Return `RouteDecision.CONVERSATIONAL` otherwise
    - Return `RouteDecision.CONVERSATIONAL` when `intent.confidence < 0.5` regardless of tag
    - Implement as a pure function with no side effects
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 4.2 Write property test for route determinism
    - **Property 4: route is deterministic and pure**
    - **Validates: Requirements 2.4**
    - Use `hypothesis` to generate arbitrary `Intent` objects and assert calling `route` twice on the same intent returns the same `RouteDecision`

  - [ ]* 4.3 Write unit tests for `route`
    - Test known actionable tags return `ACTIONABLE`
    - Test unknown tags return `CONVERSATIONAL`
    - Test confidence < 0.5 always returns `CONVERSATIONAL`
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 5. Implement Action Dispatcher
  - [x] 5.1 Implement `dispatch(req, registry)` in `jarvis/dispatcher.py`
    - Look up skill by `skillId`; return `ActionResult.Failure("Skill not found: <id>")` if absent
    - Validate all `requiredParams` are present; return `ActionResult.Failure("Missing params: <list>")` if any are missing
    - Sanitize all entity values (app names, file paths) to prevent command injection before passing to skill executor
    - Wrap `skill.execute` in a try/except; return `ActionResult.Failure(str(e))` on any exception
    - Never propagate exceptions to the caller
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.1_

  - [ ]* 5.2 Write property test for dispatch always returning ActionResult
    - **Property 5: dispatch always returns an ActionResult (never throws)**
    - **Validates: Requirements 4.5**
    - Use `hypothesis` to generate arbitrary `ActionRequest` and `SkillRegistry` inputs (including skills that raise exceptions) and assert `dispatch` always returns an `ActionResult`

  - [ ]* 5.3 Write unit tests for `dispatch`
    - Test skill-not-found returns correct `Failure` message format
    - Test missing required params returns `Failure` listing all missing names
    - Test exception in `skill.execute` is caught and returned as `Failure`
    - Test successful dispatch returns `Success` with command string
    - Test command injection sanitization on entity values
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.1_

- [x] 6. Checkpoint — Ensure all tests pass
  - Run `pytest` and confirm all tests for context memory, skill registry, router, and dispatcher pass. Ask the user if questions arise.

- [x] 7. Implement NLU & Intent Resolver
  - [x] 7.1 Implement `resolve_intent(input_str, ctx)` in `jarvis/nlu.py`
    - Reject empty input strings with a descriptive error
    - Load `ConversationContext` to disambiguate pronouns and references before entity extraction
    - Extract named entities (app names, dates, file paths) as key-value pairs
    - Assign a confidence score in [0.0, 1.0]
    - Set `rawInput` to the original input string
    - Wrap user input in a structured prompt template to prevent LLM instruction override attacks
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 10.3_

  - [ ]* 7.2 Write property test for rawInput preservation
    - **Property 10: Intent rawInput preservation**
    - **Validates: Requirements 1.1**
    - Use `hypothesis` to generate arbitrary non-empty strings and assert `resolve_intent(s, ctx).raw_input == s`

  - [ ]* 7.3 Write unit tests for `resolve_intent`
    - Test empty input raises an error
    - Test known inputs produce expected tag and entity extraction
    - Test confidence score is always in [0.0, 1.0]
    - Test `rawInput` equals the original input string
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 8. Implement LLM Response Engine
  - [x] 8.1 Implement `generate_response(intent, ctx, config)` in `jarvis/llm_engine.py`
    - Apply JARVIS personality constraints: formal tone, concise phrasing, no self-deprecating phrases
    - Include at least one proactive suggestion when `config.proactive_suggestions` is enabled and contextually relevant
    - Structure complex answers as numbered steps
    - Implement exponential backoff retry on LLM API failure or timeout
    - Return a graceful fallback `LLMResponse` if all retries are exhausted
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 11.1_

  - [ ]* 8.2 Write unit tests for `generate_response`
    - Test fallback response is returned when LLM API raises an exception
    - Test proactive suggestions are included when `proactiveSuggestions=True`
    - Test personality constraints are applied (mock LLM call)
    - _Requirements: 3.1, 3.2, 3.3, 11.1_

- [x] 9. Implement Response Formatter
  - [x] 9.1 Implement `format_output(llm_resp, act_result)` in `jarvis/formatter.py`
    - Produce `JarvisOutput` with non-empty `response` and non-empty `action` when both LLM response and `ActionResult.Success` are available
    - Produce `JarvisOutput` with non-empty `response` and `action=None` when only LLM response is available
    - Produce `JarvisOutput` with failure description in `response` and `action=None` when only `ActionResult.Failure` is available
    - Ensure `response` is always non-empty
    - Implement as a pure function with no side effects
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 9.2 Write unit tests for `format_output`
    - Test all four input combinations: (LLM+Success), (LLM only), (Failure only), (Success only)
    - Test `response` is never empty for any valid input combination
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 10. Implement the end-to-end processing pipeline
  - [x] 10.1 Implement `process_input(input_str, ctx, registry, config)` in `jarvis/pipeline.py`
    - Wire together: `resolve_intent` → `route` → `generate_response` or `dispatch` → `format_output`
    - Return clarification question in `response` with `action=None` when `intent.confidence < 0.5`
    - Set `action` to the command string from `ActionResult.Success` for actionable intents
    - Set `action=None` for conversational intents
    - Persist the interaction as a `Turn` in `ConversationContext` after producing output
    - Prompt user for explicit confirmation before executing skills that require elevated permissions
    - Inform user and suggest closest skill when a skill is not found; ask user to supply missing parameters when params are absent
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 10.2, 11.2, 11.3, 11.4_

  - [ ]* 10.2 Write property test for processInput always producing a non-empty response
    - **Property 1: processInput always produces a non-empty response**
    - **Validates: Requirements 8.1**
    - Use `hypothesis` to generate arbitrary non-empty input strings and assert `process_input(...).response != ""`

  - [ ]* 10.3 Write property test for low-confidence intents never producing actions
    - **Property 2: Low-confidence intents never produce actions**
    - **Validates: Requirements 1.3, 8.4**
    - Use `hypothesis` to generate arbitrary `Intent` objects with `confidence < 0.5` and assert `process_input(...).action is None`

  - [ ]* 10.4 Write property test for actionable intents producing action output
    - **Property 7: Actionable intents produce action output**
    - **Validates: Requirements 8.2**
    - Use `hypothesis` to generate actionable intents with successful dispatch and assert `process_input(...).action is not None`

  - [ ]* 10.5 Write property test for conversational intents producing no action output
    - **Property 8: Conversational intents produce no action output**
    - **Validates: Requirements 8.3**
    - Use `hypothesis` to generate conversational intents and assert `process_input(...).action is None`

  - [ ]* 10.6 Write integration tests for the full pipeline
    - Test conversational query → LLM response → formatted output with `action=None`
    - Test actionable command → skill dispatch → system action → formatted output with `action` set
    - Test multi-turn conversation → context memory → context-aware response
    - Test skill registration → intent routing → dispatch end-to-end
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 11. Final checkpoint — Ensure all tests pass
  - Run `pytest` across the full test suite and confirm all tests pass. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical boundaries
- Property tests use the `hypothesis` library and validate universal correctness properties from the design
- Unit tests validate specific examples and edge cases
- The implementation language is Python; all code examples and test patterns should use Python idioms
