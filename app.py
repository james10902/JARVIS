"""JARVIS Flask UI — always-on VAD, auto-interrupt, parallel processing."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from jarvis.models import ConversationContext, JarvisConfig
from jarvis.pipeline import process_input
from jarvis.skill_registry import SkillRegistry
from jarvis.skills import register_all
from jarvis.voice import (
    listen, speak, stop_speaking,
    prewarm, start_vad_monitor, set_interrupt_callback,
)

# ── Pre-warm Whisper immediately (background thread) ─────
threading.Thread(target=prewarm, daemon=True).start()

# ── App setup ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'jarvis-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ── JARVIS state ──────────────────────────────────────────
config = JarvisConfig(
    personality_mode="formal",
    max_context_turns=20,
    llm_model="llama-3.3-70b-versatile",
    verbose_mode=False,
    proactive_suggestions=True,
)

_ctx_lock = threading.Lock()
_ctx      = ConversationContext(max_turns=config.max_context_turns)
_registry = SkillRegistry()
register_all(_registry)

_state_lock    = threading.Lock()
_current_state = "standby"
_listen_lock   = threading.Lock()   # prevent overlapping listen sessions

_WAKE_PHRASES = {
    # Standard
    "jarvis", "hey jarvis", "hello jarvis", "wake up",
    "wake up jarvis", "yo jarvis", "jarvis wake up",
    "wake up daddy is home", "daddy is home",
    # Common Whisper mishearings
    "travis", "hey travis", "javas", "hey javas",
    "jarvis wake", "jarvis please", "ok jarvis",
    "davis", "hey davis", "java", "java's",
}

def _time_greeting() -> str:
    h = datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"

def _wake_response() -> str:
    return (f"{_time_greeting()}. Welcome, Sir! "
            "So what are we working on this time — any interesting projects?")

def _set_state(state: str, label: str):
    global _current_state
    with _state_lock:
        _current_state = state
    socketio.emit('status', {'state': state, 'label': label.upper()})

# ── Auto-interrupt callback (called by VAD when voice detected during speech) ──
def _on_voice_interrupt():
    """VAD detected user speaking while JARVIS was talking — start listening."""
    with _state_lock:
        state = _current_state
    if state == 'speaking':
        socketio.emit('interrupted', {})
        _start_listen_thread()

set_interrupt_callback(_on_voice_interrupt)

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Receive raw audio from mobile browser, transcribe with Whisper, run pipeline.

    The phone records audio via MediaRecorder and POSTs it as multipart/form-data.
    We transcribe on the server (no Whisper needed on the phone) and process it
    through the full JARVIS pipeline, returning the response as JSON.
    """
    global _ctx

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    import tempfile, os, numpy as np
    audio_file = request.files['audio']

    # Save to temp file for Whisper (phones send webm/ogg/mp4)
    suffix = '.webm'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        from faster_whisper import WhisperModel
        from jarvis.voice import _get_whisper
        model = _get_whisper()
        segments, _ = model.transcribe(
            tmp_path,
            language='en',
            beam_size=1,
            vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 200},
        )
        user_text = ' '.join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        return jsonify({'error': f'Transcription failed: {exc}'}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not user_text:
        return jsonify({'error': 'No speech detected'}), 200

    # Wake phrase check
    if user_text.strip().lower() in _WAKE_PHRASES:
        response_text = _wake_response()
        return jsonify({'user_text': user_text, 'response': response_text, 'action': None})

    # Run through pipeline
    try:
        with _ctx_lock:
            output, _ctx = process_input(user_text, _ctx, _registry, config)
        return jsonify({
            'user_text': user_text,
            'response':  output.response,
            'action':    output.action,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

# ── SocketIO events ───────────────────────────────────────
@socketio.on('start_listen')
def handle_start_listen():
    with _state_lock:
        state = _current_state
    if state == 'speaking':
        stop_speaking()
    if state in ('listening', 'thinking'):
        return
    _start_listen_thread()

@socketio.on('interrupt')
def handle_interrupt():
    stop_speaking()
    _set_state('standby', 'INTERRUPTED')
    _start_listen_thread()

def _start_listen_thread():
    t = threading.Thread(target=_listen_and_respond, daemon=True)
    t.start()

# ── Core pipeline ─────────────────────────────────────────
def _listen_and_respond():
    global _ctx

    # Only one listen session at a time
    if not _listen_lock.acquire(blocking=False):
        return

    try:
        _set_state('listening', 'LISTENING')

        def on_partial(text: str):
            socketio.emit('partial_transcript', {'text': text})

        user_text = listen(on_partial=on_partial)

        if not user_text:
            _set_state('standby', 'STANDBY')
            return

        # Guard against noise transcriptions (single words from ambient pickup)
        if len(user_text.strip().split()) < 2 and user_text.strip().lower() not in _WAKE_PHRASES:
            print(f"[Ignored noise: '{user_text.strip()}']")
            _set_state('standby', 'STANDBY')
            return

        socketio.emit('transcript', {'text': user_text})

        # Exit
        if user_text.strip().lower() in {"exit", "quit", "shut down", "goodbye jarvis"}:
            farewell = "Shutting down. Goodbye, Sir."
            _set_state('speaking', 'SPEAKING')
            socketio.emit('response', {'text': farewell, 'action': None, 'user_text': user_text})
            speak(farewell)
            _set_state('standby', 'OFFLINE')
            return

        # Wake trigger
        if user_text.strip().lower() in _WAKE_PHRASES:
            response = _wake_response()
            _set_state('speaking', 'SPEAKING')
            socketio.emit('response', {'text': response, 'action': None, 'user_text': user_text})
            speak(response)
            _set_state('standby', 'STANDBY')
            return

        # ── Pipeline + parallel TTS prefetch ─────────────────────────────
        _set_state('thinking', 'THINKING')

        with _ctx_lock:
            output, _ctx = process_input(user_text, _ctx, _registry, config)

        # Emit response to UI immediately
        _set_state('speaking', 'SPEAKING')
        socketio.emit('response', {
            'text':      output.response,
            'action':    output.action,
            'user_text': user_text,
        })

        # Speak — routes through the dedicated audio thread queue
        speak(output.response)

        _set_state('standby', 'STANDBY')

    except Exception as exc:
        import traceback; traceback.print_exc()
        socketio.emit('error_msg', {'text': str(exc)})
        _set_state('standby', 'ERROR')
    finally:
        _listen_lock.release()

# ── Entry point ───────────────────────────────────────────
if __name__ == '__main__':
    # Start always-on VAD monitor
    start_vad_monitor()

    print("\n" + "="*50)
    print("  JARVIS UI  →  http://localhost:5000")
    print("="*50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
