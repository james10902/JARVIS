"""Voice I/O for JARVIS — robust audio engine.

Architecture:
- A single persistent AUDIO THREAD owns all sounddevice playback.
  Flask/SocketIO worker threads just put audio onto a queue and wait.
  This avoids the OutputStream callback failures that occur when audio
  is started from non-main threads on Windows.
- VAD runs in its own thread, monitors the mic, and calls stop_speaking()
  which drains the queue and wakes the audio thread immediately.
- Whisper base.en transcribes in-memory (no temp file).
"""

from __future__ import annotations

import io
import os
import queue
import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Whisper ───────────────────────────────────────────────────────────────────
_whisper_model = None
_whisper_lock  = threading.Lock()


def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _whisper_model


def prewarm():
    _get_whisper()


# ── Audio playback engine ─────────────────────────────────────────────────────
# All playback goes through this queue → dedicated audio thread.
# Entries: (np.ndarray, int samplerate) or the sentinel STOP_TOKEN.

_STOP_TOKEN   = object()
_audio_queue  : queue.Queue = queue.Queue()
_play_done    = threading.Event()   # set when current item finishes
_is_speaking  = threading.Event()   # set while audio is playing
_stop_playback = threading.Event()  # set to interrupt current playback
_interrupt_cb : Optional[Callable] = None


def _audio_thread_main():
    """Dedicated thread — the ONLY place sd.play/sd.wait is called."""
    while True:
        item = _audio_queue.get()
        if item is _STOP_TOKEN:
            _play_done.set()
            continue

        audio_arr, sr = item
        _stop_playback.clear()
        _is_speaking.set()
        _play_done.clear()

        try:
            sd.play(audio_arr, samplerate=sr)
            # Poll sd.wait in small steps so _stop_playback can interrupt
            while True:
                if _stop_playback.is_set():
                    sd.stop()
                    break
                # sd.get_stream() is safe here — we are the only thread playing
                try:
                    if not sd.get_stream().active:
                        break
                except Exception:
                    break
                time.sleep(0.03)
            # Drain any leftover
            try:
                sd.stop()
            except Exception:
                pass
        except Exception as exc:
            print(f"[AUDIO ENGINE] playback error: {exc}")
        finally:
            _is_speaking.clear()
            _play_done.set()
            _audio_queue.task_done()


# Start the audio thread once at import
_audio_thread = threading.Thread(target=_audio_thread_main, daemon=True, name="jarvis-audio")
_audio_thread.start()


def _play_blocking(audio_arr: np.ndarray, samplerate: int) -> None:
    """Queue audio and block the calling thread until it finishes (or is stopped)."""
    _play_done.clear()
    _audio_queue.put((audio_arr, samplerate))
    _play_done.wait()


def stop_speaking() -> None:
    """Interrupt playback immediately."""
    _stop_playback.set()
    # Drain the queue
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
        except queue.Empty:
            break
    # Wake the audio thread with a stop token so it clears state
    _audio_queue.put(_STOP_TOKEN)
    _is_speaking.clear()


def set_interrupt_callback(cb: Callable):
    global _interrupt_cb
    _interrupt_cb = cb


# ── Always-on VAD ─────────────────────────────────────────────────────────────
_VAD_THRESHOLD   = 0.015
_VAD_HOLD_FRAMES = 3
_vad_running     = False


def start_vad_monitor():
    global _vad_running
    if _vad_running:
        return
    _vad_running = True
    threading.Thread(target=_vad_loop, daemon=True, name="jarvis-vad").start()


def _vad_loop():
    loud_count = 0
    BLOCK = 1600  # 100ms at 16kHz

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
    silence_threshold: float = 0.018,
    silence_duration: float = 0.55,
    max_duration: float = 12.0,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
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
    try:
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        model = _get_whisper()
        segments, _ = model.transcribe(
            buf, language="en", beam_size=1,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 200},
        )
        return " ".join(s.text.strip() for s in segments).strip() or None
    except Exception:
        return _transcribe_file(audio, sample_rate)


def _transcribe_file(audio: np.ndarray, sample_rate: int) -> Optional[str]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, sample_rate)
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(
            tmp_path, language="en", beam_size=1,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 200},
        )
        return " ".join(s.text.strip() for s in segments).strip() or None
    finally:
        os.unlink(tmp_path)


# ── Text-to-speech ────────────────────────────────────────────────────────────

# Persistent Edge TTS event loop
_edge_tts_loop : Optional[object] = None
_edge_tts_lock = threading.Lock()


def _get_edge_loop():
    global _edge_tts_loop
    import asyncio
    with _edge_tts_lock:
        if _edge_tts_loop is None or _edge_tts_loop.is_closed():
            _edge_tts_loop = asyncio.new_event_loop()
    return _edge_tts_loop


def speak(text: str) -> None:
    """Convert text to audio and play it through the audio engine."""
    api_key  = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "29vD33N1CtxCmqQRPOHJ").strip()

    if api_key and api_key != "your_elevenlabs_api_key_here":
        try:
            _speak_elevenlabs(text, api_key, voice_id)
            return
        except Exception as e:
            print(f"[ElevenLabs error: {type(e).__name__}: {e}] — falling back to Edge TTS")
    _speak_fallback(text)


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
    if not audio_bytes:
        raise RuntimeError("ElevenLabs returned empty audio")
    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"[TTS] ElevenLabs: {len(audio_arr)} samples @ 22050Hz")
    _play_blocking(audio_arr, 22050)


def _speak_fallback(text: str) -> None:
    try:
        import asyncio
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
        data, sr = sf.read(tmp_path)
        os.unlink(tmp_path)
        if data.ndim > 1:
            data = data[:, 0]
        print(f"[TTS] Edge TTS: {len(data)} samples @ {sr}Hz")
        _play_blocking(data.astype(np.float32), sr)
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
