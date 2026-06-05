"""Groq API client with automatic key rotation.

Tries GROQ_API_KEY first. If it hits a rate-limit or quota error,
automatically falls over to GROQ_API_KEY_2.

Model used: llama-3.3-70b-versatile — fast, free-tier friendly on Groq.
"""

from __future__ import annotations

import os
from typing import List

# ---------------------------------------------------------------------------
# Key pool — loaded once at import time
# ---------------------------------------------------------------------------

def _load_keys() -> List[str]:
    keys = []
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_2"):
        val = os.environ.get(name, "").strip()
        if val:
            keys.append(val)
    if not keys:
        raise EnvironmentError(
            "No Groq API keys found. Set GROQ_API_KEY (and optionally "
            "GROQ_API_KEY_2) in your .env file."
        )
    return keys


_KEYS: List[str] = []   # populated lazily on first call
_MODEL = "llama-3.3-70b-versatile"


def _get_keys() -> List[str]:
    global _KEYS
    if not _KEYS:
        _KEYS = _load_keys()
    return _KEYS


# ---------------------------------------------------------------------------
# Public call function
# ---------------------------------------------------------------------------

def chat(messages: list[dict], model: str | None = None) -> str:
    """Send a chat request to Groq, rotating keys on rate-limit errors.

    Args:
        messages: List of message dicts (role/content).
        model: Optional model override. Defaults to llama-3.3-70b-versatile.

    Returns:
        The assistant's response text.

    Raises:
        Exception: If all keys are exhausted or a non-quota error occurs.
    """
    from groq import Groq, RateLimitError, AuthenticationError

    keys = _get_keys()
    use_model = model or _MODEL
    last_exc: Exception | None = None

    for key in keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except RateLimitError as exc:
            # Quota exhausted on this key — try the next one
            last_exc = exc
            continue
        except AuthenticationError as exc:
            # Invalid key — skip it and try the next, but warn clearly
            print(f"[GROQ] Invalid API key (skipping): {exc}")
            last_exc = exc
            continue
        except Exception:
            raise

    # All keys failed — give a clear actionable message
    if last_exc and "invalid_api_key" in str(last_exc).lower():
        raise RuntimeError(
            "All Groq API keys are invalid or expired. "
            "Generate new keys at https://console.groq.com and update your .env file."
        )
    raise RuntimeError(
        f"All Groq API keys exhausted. Last error: {last_exc}"
    )
