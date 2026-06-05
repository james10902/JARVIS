# JARVIS AI Assistant

A personal AI assistant inspired by Iron Man's JARVIS — calm, confident, and intelligent. Combines natural language understanding with structured command dispatch, context-aware conversation, and full voice I/O.

---

## Features

- **Voice-first** — listens via microphone (faster-whisper), speaks via ElevenLabs or Edge TTS fallback
- **Always-on VAD** — background voice activity detection auto-interrupts JARVIS mid-speech when you speak
- **Conversational AI** — powered by Groq (llama-3.3-70b-versatile) with dual API key rotation
- **Action dispatch** — routes commands to registered skills (open apps, find files, sleep/shutdown/restart)
- **Context memory** — rolling conversation window (default 20 turns) for pronoun resolution and continuity
- **Web UI** — Flask + Socket.IO interface with real-time status (LISTENING / THINKING / SPEAKING)
- **Modular skill registry** — add new capabilities without touching the core pipeline
- **Formal JARVIS personality** — no self-deprecating phrases, concise responses, numbered steps for complex answers

---

## Architecture

```
Voice / Web UI
     │
     ▼
NLU & Intent Resolver  ←──  Context Memory
     │
     ▼
Command Router
     │
  ┌──┴──┐
  │     │
LLM  Dispatcher ──► Skill Registry ──► System Layer
Engine  │
  │     │
  └──┬──┘
     ▼
Response Formatter
     │
     ▼
Voice / Web UI
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)
- Optional: [ElevenLabs API key](https://elevenlabs.io) for high-quality voice

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy or edit `.env` in the project root:

```env
GROQ_API_KEY=your_groq_key_here
GROQ_API_KEY_2=your_second_groq_key_here   # optional fallback

ELEVENLABS_API_KEY=your_elevenlabs_key_here  # optional, falls back to Edge TTS
ELEVENLABS_VOICE_ID=29vD33N1CtxCmqQRPOHJ    # optional, default is Adam
```

### Run (Voice Mode — CLI)

```bash
python main.py
```

Speak to JARVIS. Say `exit` or `quit` to shut down.

### Run (Web UI)

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Voice Pipeline

| Component | Technology |
|---|---|
| Speech-to-text | faster-whisper (tiny, CPU, int8) |
| Text-to-speech (primary) | ElevenLabs `eleven_turbo_v2` |
| Text-to-speech (fallback) | Microsoft Edge TTS (`en-GB-RyanNeural`) |
| Text-to-speech (last resort) | pyttsx3 (Windows David/Mark) |
| Voice activity detection | sounddevice + RMS threshold |

---

## Built-in Skills

| Skill | Trigger phrases | Parameters |
|---|---|---|
| Open application | "open chrome", "launch notepad" | `app` |
| Find / manage file | "find file report.pdf", "search for notes" | `name` |
| Sleep computer | "sleep", "suspend" | — |
| Shutdown computer | "shutdown", "power off" | — |
| Restart computer | "restart", "reboot" | — |

### Adding a Custom Skill

```python
from jarvis.models import ActionResult, Skill
from jarvis.skill_registry import SkillRegistry

registry.register(Skill(
    id="my_skill",
    description="Does something useful",
    intent_tags=["my_intent", "do_thing"],
    required_params=["target"],
    execute=lambda params: ActionResult.success(f"Done: {params['target']}"),
))
```

---

## Project Structure

```
JARVIS/
├── app.py                  # Flask + Socket.IO web UI
├── main.py                 # Voice-only CLI entry point
├── requirements.txt
├── .env                    # API keys (not committed)
│
├── jarvis/
│   ├── models.py           # Core dataclasses (Intent, Turn, JarvisOutput, …)
│   ├── pipeline.py         # End-to-end processing pipeline
│   ├── nlu.py              # NLU & Intent Resolver (LLM-powered)
│   ├── router.py           # Command Router (conversational vs actionable)
│   ├── llm_engine.py       # LLM Response Engine (Groq, with retry)
│   ├── dispatcher.py       # Action Dispatcher (skill execution)
│   ├── skill_registry.py   # Skill Registry
│   ├── skills.py           # Built-in skills (Windows system integration)
│   ├── context_memory.py   # Context Memory (rolling window + persistence)
│   ├── formatter.py        # Response Formatter
│   ├── groq_client.py      # Groq API client (key rotation)
│   └── voice.py            # Voice I/O (listen + speak + VAD)
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── templates/
│   └── index.html
│
├── tests/                  # pytest + hypothesis test suite
│
└── specs/jarvis-ai-assistant/
    ├── requirements.md
    ├── design.md
    └── tasks.md
```

---

## Running Tests

```bash
pytest
```

---

## Wake Phrases

Say any of the following to trigger the wake greeting:

- `jarvis`
- `hey jarvis`
- `wake up`
- `wake up daddy is home`
- `yo jarvis`

---

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | LLM inference (llama-3.3-70b-versatile) |
| `faster-whisper` | Local speech-to-text |
| `elevenlabs` | High-quality TTS |
| `edge-tts` | Free neural TTS fallback |
| `sounddevice` | Audio recording and playback |
| `soundfile` | Audio file I/O |
| `numpy` | Audio signal processing |
| `flask` + `flask-socketio` | Web UI |
| `python-dotenv` | Environment variable loading |
| `pyttsx3` | Last-resort TTS fallback |
| `pytest` + `hypothesis` | Testing |
