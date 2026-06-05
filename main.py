"""JARVIS AI Assistant — fully voice-driven entry point.

Speak to JARVIS. It listens, understands, and responds with a real voice.

Speech-to-text : faster-whisper (local, free)
Text-to-speech : ElevenLabs (set ELEVENLABS_API_KEY in .env for best voice)
                 Falls back to Windows built-in TTS if key not set.
LLM            : Groq (llama-3.3-70b-versatile) with key rotation

Required in .env:
    GROQ_API_KEY          — primary Groq key
    GROQ_API_KEY_2        — (optional) fallback Groq key
    ELEVENLABS_API_KEY    — ElevenLabs key for high-quality voice
    ELEVENLABS_VOICE_ID   — ElevenLabs voice ID (default: Adam)
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
load_dotenv()

from jarvis.models import ConversationContext, JarvisConfig
from jarvis.pipeline import process_input
from jarvis.skill_registry import SkillRegistry
from jarvis.skills import register_all, _time_greeting
from jarvis.voice import listen, speak

# ---------------------------------------------------------------------------
# Wake trigger detection
# ---------------------------------------------------------------------------

_WAKE_PHRASES = {
    "jarvis",
    "hey jarvis",
    "hello jarvis",
    "wake up",
    "wake up daddy is home",
    "daddy is home",
    "wake up jarvis",
    "yo jarvis",
    "jarvis wake up",
}


def _is_wake_trigger(text: str) -> bool:
    return text.strip().lower() in _WAKE_PHRASES


def _wake_response() -> str:
    greeting = _time_greeting()
    return (
        f"{greeting}. Welcome, Sir! "
        "So what are we working on this time — any interesting projects?"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate Groq keys
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GROQ_API_KEY_2"):
        print("Error: No Groq API keys found in .env file.")
        sys.exit(1)

    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not elevenlabs_key or elevenlabs_key == "your_elevenlabs_api_key_here":
        print("Note: ELEVENLABS_API_KEY not set — using Windows built-in TTS voice.")
        print("      Add your ElevenLabs key to .env for a much better voice.\n")

    config = JarvisConfig(
        personality_mode="formal",
        max_context_turns=20,
        llm_model="llama-3.3-70b-versatile",
        verbose_mode=False,
        proactive_suggestions=True,
    )

    ctx = ConversationContext(max_turns=config.max_context_turns)
    registry = SkillRegistry()
    register_all(registry)

    print("=" * 55)
    print("  JARVIS is online. Speak to interact.")
    print("  Say 'exit' or 'quit' to shut down.")
    print("=" * 55)
    print()

    while True:
        # --- Listen ---
        user_input = listen()

        if user_input is None:
            # Nothing heard — keep listening
            continue

        # Guard against single-word noise transcriptions (common with ambient mic pickup)
        user_input_stripped = user_input.strip()
        if len(user_input_stripped.split()) < 2 and user_input_stripped.lower() not in _WAKE_PHRASES:
            print(f"[Ignored noise: '{user_input_stripped}']")
            continue

        user_input_clean = user_input_stripped.lower()

        # --- Exit commands ---
        if user_input_clean in {"exit", "quit", "shut down", "goodbye jarvis"}:
            farewell = "Shutting down. Goodbye, Sir."
            print(f"JARVIS: {farewell}\n")
            speak(farewell)
            break

        # --- Wake trigger ---
        if _is_wake_trigger(user_input_clean):
            response = _wake_response()
            print(f"JARVIS: {response}\n")
            speak(response)
            continue

        # --- Process through pipeline ---
        try:
            output, ctx = process_input(user_input, ctx, registry, config)
        except ValueError as exc:
            msg = str(exc)
            print(f"JARVIS: {msg}\n")
            speak(msg)
            continue
        except RuntimeError as exc:
            msg = str(exc)
            print(f"\n[FATAL] {msg}\n")
            # If it's a key error, no point continuing — exit cleanly
            if "invalid" in msg.lower() or "api key" in msg.lower() or "exhausted" in msg.lower():
                speak("My API keys are invalid, Sir. Please update the dot env file with new Groq keys and restart.")
                break
            speak("I encountered an unexpected error. Please try again.")
            continue
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            msg = "I encountered an unexpected error. Please try again."
            print(f"JARVIS: {msg}\n")
            speak(msg)
            continue

        print(f"JARVIS: {output.response}")
        speak(output.response)

        if output.action:
            print(f"  → {output.action}")
        print()


if __name__ == "__main__":
    main()
