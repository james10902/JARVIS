# Requirements Document

## Introduction

JARVIS is a personal AI assistant that combines natural language understanding with structured command dispatch and context-aware conversation. The system accepts raw user input, resolves intent, routes it to either a conversational LLM engine or an action dispatcher, and returns a structured output containing a response and an optional system action. The architecture is modular: new capabilities are registered as skills without modifying the core pipeline. This document derives formal requirements from the approved design document.

---

## Glossary

- **System**: The JARVIS AI assistant as a whole
- **NLU_Resolver**: The NLU & Intent Resolver component that parses raw input into a structured Intent
- **Intent**: A structured representation of user intent containing a tag, entities, confidence score, and raw input
- **Router**: The Command Router that classifies an Intent as Conversational or Actionable
- **LLM_Engine**: The LLM Response Engine that generates natural language responses
- **Dispatcher**: The Action Dispatcher that resolves and executes skills
- **SkillRegistry**: The registry that stores all registered skills and their metadata
- **Skill**: A registered capability with an ID, description, intent tags, required parameters, and an execute function
- **ContextMemory**: The Context Memory component that maintains a rolling window of conversation turns
- **Formatter**: The Response Formatter that assembles the final JarvisOutput
- **JarvisOutput**: The structured output containing a response string and an optional action string
- **Turn**: A single conversation entry with a role ("user" or "jarvis"), content, and timestamp
- **ConversationContext**: A collection of turns with a configured maximum window size
- **ActionRequest**: A request to execute a skill, containing a skillId and parameter map
- **ActionResult**: The result of a skill execution, either Success or Failure with a human-readable message
- **JarvisConfig**: The configuration structure controlling personality mode, context window size, LLM model, and feature flags

---

## Requirements

### Requirement 1: Intent Resolution

**User Story:** As a user, I want JARVIS to understand my natural language input, so that I can interact with it conversationally without learning specific command syntax.

#### Acceptance Criteria

1. WHEN a user provides a non-empty input string, THE NLU_Resolver SHALL parse it into an Intent with a non-empty tag, a confidence score in [0.0, 1.0], and the rawInput field set to the original input string
2. WHEN the NLU_Resolver processes input, THE NLU_Resolver SHALL load the current ConversationContext to disambiguate pronouns and references before resolving entities
3. WHEN the NLU_Resolver extracts entities, THE NLU_Resolver SHALL produce a list of key-value pairs representing named entities such as application names, dates, and file paths
4. IF the user provides an empty input string, THEN THE NLU_Resolver SHALL return an error without producing an Intent

---

### Requirement 2: Intent Routing

**User Story:** As a user, I want JARVIS to automatically decide whether my request needs a conversational reply or a system action, so that I do not have to specify the interaction mode manually.

#### Acceptance Criteria

1. WHEN an Intent tag is present in the actionable tag registry, THE Router SHALL return the Actionable route decision
2. WHEN an Intent tag is not present in the actionable tag registry, THE Router SHALL return the Conversational route decision
3. IF an Intent has a confidence score below 0.5, THEN THE Router SHALL route the Intent to the LLM_Engine for clarification regardless of the intent tag
4. THE Router SHALL be a pure function with no side effects, returning the same RouteDecision for the same Intent on every invocation

---

### Requirement 3: Conversational Response Generation

**User Story:** As a user, I want JARVIS to respond to conversational queries with concise, confident, and contextually relevant answers, so that I receive useful information without unnecessary verbosity.

#### Acceptance Criteria

1. WHEN a Conversational Intent is received, THE LLM_Engine SHALL generate a natural language response using the current ConversationContext
2. WHEN generating a response, THE LLM_Engine SHALL apply the JARVIS personality constraints: formal tone, concise phrasing, and no self-deprecating phrases
3. WHERE the JarvisConfig has proactiveSuggestions enabled, THE LLM_Engine SHALL include at least one proactive suggestion in the LLMResponse when contextually relevant
4. WHEN a complex answer is required, THE LLM_Engine SHALL structure the response as numbered steps

---

### Requirement 4: Action Dispatch

**User Story:** As a user, I want JARVIS to execute system-level actions on my behalf, so that I can control my computer through natural language commands.

#### Acceptance Criteria

1. WHEN an Actionable Intent is received, THE Dispatcher SHALL look up the corresponding Skill in the SkillRegistry by skillId and execute it with the resolved parameters
2. IF the skillId in an ActionRequest is not found in the SkillRegistry, THEN THE Dispatcher SHALL return ActionResult.Failure with a message in the format "Skill not found: \<skillId\>"
3. IF one or more required parameters are absent from the ActionRequest, THEN THE Dispatcher SHALL return ActionResult.Failure with a message listing all missing parameter names
4. IF a system-level exception occurs during skill execution, THEN THE Dispatcher SHALL catch the exception and return ActionResult.Failure with a human-readable error message
5. THE Dispatcher SHALL always return an ActionResult and SHALL NOT propagate exceptions to the caller

---

### Requirement 5: Skill Registry

**User Story:** As a developer, I want to register new skills without modifying the core pipeline, so that JARVIS capabilities can be extended modularly.

#### Acceptance Criteria

1. WHEN a Skill is registered, THE SkillRegistry SHALL store the Skill and make it retrievable by its unique ID
2. WHEN a Skill is registered, THE SkillRegistry SHALL make the Skill retrievable by each of its declared intent tags
3. THE SkillRegistry SHALL enforce unique skill IDs; registering a Skill with an ID that already exists SHALL replace the existing Skill
4. WHEN a Skill is registered, THE SkillRegistry SHALL validate that the Skill's requiredParams list is well-formed (no duplicate or empty parameter names)
5. WHEN a skill lookup by ID is performed for an ID not in the registry, THE SkillRegistry SHALL return no result

---

### Requirement 6: Context Memory

**User Story:** As a user, I want JARVIS to remember the context of our conversation, so that I can refer to prior topics without repeating myself.

#### Acceptance Criteria

1. WHEN a Turn is added to the ConversationContext, THE ContextMemory SHALL append the Turn and enforce the rolling window by dropping the oldest Turn when the total count exceeds maxTurns
2. WHEN the ConversationContext is queried, THE ContextMemory SHALL return turns in chronological order with the most recent turn last
3. THE ContextMemory SHALL persist the ConversationContext to a durable storage backend so that context is available across sessions
4. WHEN a Turn is added, THE ContextMemory SHALL validate that the Turn's role is one of "user" or "jarvis" and that the content is non-empty
5. THE ContextMemory SHALL ensure that the maxTurns value is unchanged after any addTurn operation

---

### Requirement 7: Response Formatting

**User Story:** As a user, I want JARVIS to present its output in a consistent, structured format, so that I can clearly distinguish what JARVIS says from what it does.

#### Acceptance Criteria

1. WHEN both an LLM response and a successful ActionResult are available, THE Formatter SHALL produce a JarvisOutput with a non-empty response field and a non-empty action field
2. WHEN only an LLM response is available, THE Formatter SHALL produce a JarvisOutput with a non-empty response field and no action field
3. WHEN an ActionResult.Failure is the only result, THE Formatter SHALL produce a JarvisOutput with a non-empty response field describing the failure and no action field
4. THE Formatter SHALL always produce a JarvisOutput whose response field is non-empty
5. THE Formatter SHALL be a pure function with no side effects

---

### Requirement 8: End-to-End Processing Pipeline

**User Story:** As a user, I want every input I provide to produce a coherent response, so that JARVIS never silently fails or returns an empty reply.

#### Acceptance Criteria

1. WHEN a non-empty input string is provided, THE System SHALL produce a JarvisOutput with a non-empty response field
2. WHEN an actionable Intent is successfully dispatched, THE System SHALL set the JarvisOutput action field to the command string returned by the Skill
3. WHEN a conversational Intent is processed, THE System SHALL set the JarvisOutput action field to None
4. IF the resolved Intent has a confidence score below 0.5, THEN THE System SHALL return a clarification question in the response field and set the action field to None
5. WHEN a JarvisOutput is produced, THE System SHALL persist the interaction as a Turn in the ConversationContext

---

### Requirement 9: Configuration

**User Story:** As a system operator, I want to configure JARVIS behaviour through a structured configuration object, so that I can tune personality, memory, and model settings without modifying code.

#### Acceptance Criteria

1. THE System SHALL accept a JarvisConfig that specifies personalityMode, maxContextTurns, llmModel, verboseMode, and proactiveSuggestions
2. IF the JarvisConfig specifies a maxContextTurns value of 0 or less, THEN THE System SHALL reject the configuration with a descriptive error
3. IF the JarvisConfig specifies an empty llmModel string, THEN THE System SHALL reject the configuration with a descriptive error
4. IF the JarvisConfig specifies a personalityMode that is not "formal" or "casual", THEN THE System SHALL reject the configuration with a descriptive error

---

### Requirement 10: Security and Safety

**User Story:** As a system operator, I want JARVIS to handle user input and system commands safely, so that the system is not vulnerable to injection attacks or privilege escalation.

#### Acceptance Criteria

1. WHEN constructing a system-level command, THE Dispatcher SHALL sanitize all entity values (such as application names and file paths) to prevent command injection
2. WHEN a Skill requires elevated system permissions, THE System SHALL prompt the user for explicit confirmation before executing the Skill
3. WHEN user input is passed to the LLM_Engine, THE System SHALL wrap it in a structured prompt template that prevents instruction override attacks
4. WHEN persisting ConversationContext, THE ContextMemory SHALL redact sensitive values such as passwords and API keys before writing to storage

---

### Requirement 11: Error Handling and Recovery

**User Story:** As a user, I want JARVIS to handle errors gracefully and guide me toward a resolution, so that failures do not leave me without a useful response.

#### Acceptance Criteria

1. IF the LLM_Engine fails or times out, THEN THE System SHALL return a fallback response informing the user that processing is temporarily unavailable and SHALL retry with exponential backoff
2. IF a Skill is not found for an actionable Intent, THEN THE System SHALL inform the user that the action is not supported and SHALL suggest the closest available Skill or ask the user to rephrase
3. IF required parameters are missing for a Skill, THEN THE System SHALL ask the user to supply the specific missing information
4. IF a system-level execution error occurs, THEN THE System SHALL report the error to the user in plain language without exposing technical stack traces
