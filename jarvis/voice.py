"""Voice I/O for JARVIS — optimised for low latency.

Key improvements over v1:
- Silence detection cut to 450ms (was 800ms) → faster end-of-speech detection
- Transcription runs in-memory via BytesIO (no temp file disk I/O)
- Persistent Edge TTS asyncio loop (no per-call loop creation overhead)
- Whisper base.en model (better accuracy than tiny, still fast on CPU)
- Always-on VAD auto-interrupts JARVIS mid-speech when user speaks
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Pre-warm Whisper at import (eliminates first-use delay) ──────────────────
_whisper_model = None
_whisper_lock  = threading.Lock()


def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            # base.en: better accuracy than tiny, still fast on CPU with int8
            _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _whisper_model


def prewarm():
    """Call once at startup to load Whisper into memory."""
    _get_whisper()


# ── Playback state ────────────────────────────────────────────────────────────
_stop_playback   = threading.Event()
_is_speaking     = threading.Event()   # set while JARVIS is playing audio
_interrupt_cb: Optional[Callable] = None


def set_interrupt_callback(cb: Callable):
    """Register a callback that fires when the user speaks over JARVIS."""
    global _interrupt_cb
    _interrupt_cb = cb


def stop_speaking():
    """Stop playback immediately."""
    _stop_playback.set()
    _is_speaking.clear()
    try:
        sd.stop()
    except Exception:
        pass


# ── Always-on VAD background thread ──────────────────────────────────────────
_VAD_THRESHOLD   = 0.015
_VAD_HOLD_FRAMES = 3
_vad_thread: Optional[threading.Thread] = None
_vad_running = False


def start_vad_monitor():
    """Start the always-on background VAD monitor."""
    global _vad_thread, _vad_running
    if _vad_running:
        return
    _vad_running = True
    _vad_thread = threading.Thread(target=_vad_loop, daemon=True)
    _vad_thread.start()


def _vad_loop():
    """Continuously monitor mic. Voice during speech → auto-interrupt."""
    loud_count = 0
    BLOCK = 1600  # 100ms at 16 kHz

    def _cb(indata, frames, time_info, status):
        nonlocal loud_count
        if not _is_speaking.is_set():
            loud_count = 0
            return
        rms = float(np.sqrt(np.mean(indata ** 2)))
        if rms > _VAD_THRESHOLD:
            loud_count += 1
            if loud_count >= _VAD_HOLD_FRAMES:
                loud_count = 0
                stop_speaking()
                if _interrupt_cb:
                    threading.Thread(target=_interrupt_cb, daemon=True).start()
        else:
            loud_count = max(0, loud_count - 1)

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=BLOCK, callback=_cb):
        while _vad_running:
            time.sleep(0.1)


# ── Speech-to-text ────────────────────────────────────────────────────────────
def listen(
    sample_rate: int = 16000,
    silence_threshold: float = 0.018,  # raised from 0.010 — filters ambient noise
    silence_duration: float = 0.55,    # slightly longer to avoid cutting off speech
    max_duration: float = 12.0,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Record until silence, transcribe with Whisper base.en."""
    frames: list[np.ndarray] = []
    silent_frames    = 0
    speaking_started = False
    block_size       = int(sample_rate * 0.1)
    silence_needed   = int(silence_duration / 0.1)
    max_blocks       = int(max_duration / 0.1)
    stop_event       = threading.Event()

    def _callback(indata, frame_count, time_info, status):
        nonlocal silent_frames, speaking_started
        chunk = indata.copy()
        frames.append(chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > silence_threshold:
            speaking_started = True
            silent_frames = 0
        elif speaking_started:
            silent_frames += 1
            if silent_frames >= silence_needed:
                stop_event.set()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                        blocksize=block_size, callback=_callback):
        count = 0
        while not stop_event.is_set() and count < max_blocks:
            sd.sleep(100)
            count += 1
            if speaking_started and count % 8 == 0 and on_partial and len(frames) > 4:
                _emit_partial(frames[:], sample_rate, on_partial)

    if not frames or not speaking_started:
        return None

    audio = np.concatenate(frames).flatten()
    return _transcribe(audio, sample_rate)


def _emit_partial(frames, sample_rate, on_partial):
    try:
        audio = np.concatenate(frames).flatten()
        text  = _transcribe(audio, sample_rate)
        if text:
            on_partial(text)
    except Exception:
        pass


def _transcribe(audio: np.ndarray, sample_rate: int) -> Optional[str]:
    """Transcribe in-memory via BytesIO — no temp file, no disk I/O."""
    try:
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        model = _get_whisper()
        segments, _ = model.transcribe(
            buf,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
        )
        return " ".join(s.text.strip() for s in segments).strip() or None
    except Exception:
        # faster-whisper may not support BytesIO on all builds — fall back to temp file
        return _transcribe_file(audio, sample_rate)


def _transcribe_file(audio: np.ndarray, sample_rate: int) -> Optional[str]:
    """Fallback: temp-file transcription."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, sample_rate)
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(
            tmp_path,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
        )
        return " ".join(s.text.strip() for s in segments).strip() or None
    finally:
        os.unlink(tmp_path)


# ── Text-to-speech ────────────────────────────────────────────────────────────

# Persistent Edge TTS event loop — reused across calls to avoid ~50ms setup overhead
_edge_tts_loop: Optional[object] = None
_edge_tts_lock = threading.Lock()


def _get_edge_loop():
    global _edge_tts_loop
    import asyncio
    with _edge_tts_lock:
        if _edge_tts_loop is None or _edge_tts_loop.is_closed():
            _edge_tts_loop = asyncio.new_event_loop()
    return _edge_tts_loop


def speak(text: str) -> None:
    """Speak text. Marks _is_speaking so VAD can auto-interrupt."""
    _stop_playback.clear()
    _is_speaking.set()

    api_key  = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "29vD33N1CtxCmqQRPOHJ").strip()

    try:
        if api_key and api_key != "your_elevenlabs_api_key_here":
            try:
                _speak_elevenlabs(text, api_key, voice_id)
                return
            except Exception as e:
                print(f"[ElevenLabs unavailable: {type(e).__name__}. Using fallback TTS.]")
        _speak_fallback(text)
    finally:
        _is_speaking.clear()


def _speak_elevenlabs(text: str, api_key: str, voice_id: str) -> None:
    from elevenlabs.client import ElevenLabs

    client    = ElevenLabs(api_key=api_key)
    audio_gen = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_turbo_v2",
        output_format="pcm_22050",
    )
    audio_bytes = b"".join(audio_gen)

    if _stop_playback.is_set():
        return

    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Use time-based wait — sd.get_stream() can return the VAD input stream
    # instead of the output stream when both are open, causing early exit
    sd.play(audio_array, samplerate=22050)
    duration = len(audio_array) / 22050
    import time
    deadline = time.time() + duration + 0.5
    while time.time() < deadline:
        if _stop_playback.is_set():
            sd.stop()
            return
        time.sleep(0.03)


def _speak_fallback(text: str) -> None:
    """Edge TTS (neural, free) with reused event loop. Falls back to pyttsx3."""
    try:
        import edge_tts
        voice = os.environ.get("EDGE_TTS_VOICE", "en-GB-RyanNeural")

        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            await communicate.save(tmp_path)
            return tmp_path

        loop     = _get_edge_loop()
        tmp_path = loop.run_until_complete(_generate())

        if _stop_playback.is_set():
            os.unlink(tmp_path)
            return

        data, samplerate = sf.read(tmp_path)
        os.unlink(tmp_path)

        if data.ndim > 1:
            data = data[:, 0]

        audio_array = data.astype(np.float32)
        sd.play(audio_array, samplerate=samplerate)
        import time
        duration = len(audio_array) / samplerate
        deadline = time.time() + duration + 0.5
        while time.time() < deadline:
            if _stop_playback.is_set():
                sd.stop()
                return
            time.sleep(0.03)

    except Exception as exc:
        print(f"[Edge TTS failed: {exc}] — using pyttsx3")
        _speak_pyttsx3(text)


def _speak_pyttsx3(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            if "david" in v.name.lower() or "mark" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:
        print(f"[pyttsx3 failed: {exc}]")
