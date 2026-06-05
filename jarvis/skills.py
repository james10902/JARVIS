"""Built-in JARVIS skills — Windows system integration.

Covers:
- App launching
- Open file/folder by exact or partial path
- File search (voice-friendly, no blocking input() calls)
- Sleep / shutdown / restart
- Wake greeting
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from jarvis.models import ActionResult, Skill
from jarvis.skill_registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEARCH_ROOTS = [
    Path.home(),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
]

_MAX_RESULTS = 5   # keep voice responses concise


def _find_files(name: str, roots: List[Path] = _SEARCH_ROOTS) -> List[Path]:
    """Recursively search for files/folders matching *name* (case-insensitive)."""
    name_lower = name.lower()
    matches: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if name_lower in p.name.lower():
                    matches.append(p)
                    if len(matches) >= _MAX_RESULTS:
                        return matches
        except PermissionError:
            continue
    return matches


def _open_path(target: Path) -> ActionResult:
    """Open a file or folder with the default Windows handler."""
    try:
        if sys.platform == "win32":
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        kind = "folder" if target.is_dir() else "file"
        return ActionResult.success(f"Opened {kind}: {target.name}")
    except Exception as exc:
        return ActionResult.failure(f"Could not open '{target}': {exc}")


def _time_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


# ---------------------------------------------------------------------------
# Skill implementations
# ---------------------------------------------------------------------------

def _execute_open_app(params: dict) -> ActionResult:
    """Open a named Windows application by name."""
    app = params.get("app", "").strip()
    if not app:
        return ActionResult.failure("No application name provided.")
    try:
        if sys.platform == "win32":
            os.startfile(app)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", app])
        else:
            subprocess.Popen(["xdg-open", app])
        return ActionResult.success(f"Opening {app}.")
    except Exception as exc:
        return ActionResult.failure(f"Could not open '{app}': {exc}")


def _execute_open_file(params: dict) -> ActionResult:
    """Open a file or folder by path.

    Accepts:
      - An exact absolute path: C:/Users/James/Documents/report.pdf
      - A relative path from the home directory: Documents/report.pdf
      - A filename only: will search and open the first match
    """
    path_str = params.get("path", "").strip()
    if not path_str:
        return ActionResult.failure("No file path provided.")

    # Try as absolute path first
    target = Path(path_str)
    if target.exists():
        return _open_path(target)

    # Try relative to home directory
    home_relative = Path.home() / path_str
    if home_relative.exists():
        return _open_path(home_relative)

    # Fall back to search
    matches = _find_files(path_str)
    if not matches:
        return ActionResult.failure(
            f"Could not find '{path_str}'. Check the name or path and try again."
        )

    # Open the first (best) match directly — no interactive prompt
    return _open_path(matches[0])


def _execute_find_file(params: dict) -> ActionResult:
    """Search for files and return results as a voice-friendly list.

    This skill is voice-UI safe: it never calls input(). Instead it returns
    a descriptive message JARVIS can speak, telling the user what was found.
    To actually open one of the results, say 'open file <name or path>'.
    """
    name = params.get("name", "").strip()
    if not name:
        return ActionResult.failure("No file name provided.")

    matches = _find_files(name)

    if not matches:
        return ActionResult.failure(
            f"No files or folders matching '{name}' were found."
        )

    if len(matches) == 1:
        # Auto-open the only result
        return _open_path(matches[0])

    # Multiple results — describe them so JARVIS can read them out
    lines = []
    for i, p in enumerate(matches, 1):
        kind = "folder" if p.is_dir() else "file"
        lines.append(f"{i}. {kind}: {p.name} in {p.parent}")

    summary = f"Found {len(matches)} results for '{name}': " + "; ".join(lines)
    summary += ". Say 'open file' followed by the name to open one."
    return ActionResult.success(summary)


def _execute_sleep(params: dict) -> ActionResult:
    """Put the computer to sleep."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                check=True,
            )
        elif sys.platform == "darwin":
            subprocess.run(["pmset", "sleepnow"], check=True)
        else:
            subprocess.run(["systemctl", "suspend"], check=True)
        return ActionResult.success("Putting the system to sleep.")
    except Exception as exc:
        return ActionResult.failure(f"Could not sleep: {exc}")


def _execute_shutdown(params: dict) -> ActionResult:
    """Shut down the computer (no interactive confirmation — confirmation
    should happen at the NLU/pipeline layer before dispatch)."""
    try:
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/s", "/t", "5"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        else:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        return ActionResult.success("Shutting down in 5 seconds.")
    except Exception as exc:
        return ActionResult.failure(f"Could not shut down: {exc}")


def _execute_restart(params: dict) -> ActionResult:
    """Restart the computer."""
    try:
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/r", "/t", "5"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["sudo", "shutdown", "-r", "now"], check=True)
        else:
            subprocess.run(["sudo", "reboot"], check=True)
        return ActionResult.success("Restarting in 5 seconds.")
    except Exception as exc:
        return ActionResult.failure(f"Could not restart: {exc}")


def _execute_wake_greeting(params: dict) -> ActionResult:
    """Deliver the JARVIS wake greeting."""
    greeting = _time_greeting()
    msg = (
        f"{greeting}. Welcome, Sir. "
        "So what are we working on this time — any interesting projects?"
    )
    return ActionResult.success(msg)


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def register_all(registry: SkillRegistry) -> None:
    """Register all built-in skills into the given registry."""

    registry.register(Skill(
        id="open_app",
        description="Opens a named Windows application",
        intent_tags=["open_app", "launch_app", "start_app", "run_app"],
        required_params=["app"],
        execute=_execute_open_app,
    ))

    registry.register(Skill(
        id="open_file",
        description="Open a file or folder by name or path",
        intent_tags=[
            "open_file", "open_document", "open_folder", "open_directory",
            "show_file", "launch_file", "view_file",
        ],
        required_params=["path"],
        execute=_execute_open_file,
    ))

    registry.register(Skill(
        id="find_file",
        description="Search for a file or folder by name and describe results",
        intent_tags=[
            "find_file", "search_file", "locate_file", "where_is_file",
            "search_for_file",
        ],
        required_params=["name"],
        execute=_execute_find_file,
    ))

    registry.register(Skill(
        id="sleep_computer",
        description="Put the computer to sleep",
        intent_tags=["sleep_computer", "sleep", "suspend"],
        required_params=[],
        execute=_execute_sleep,
    ))

    registry.register(Skill(
        id="shutdown_computer",
        description="Shut down the computer",
        intent_tags=["shutdown_computer", "shutdown", "power_off", "turn_off"],
        required_params=[],
        execute=_execute_shutdown,
    ))

    registry.register(Skill(
        id="restart_computer",
        description="Restart the computer",
        intent_tags=["restart_computer", "restart", "reboot"],
        required_params=[],
        execute=_execute_restart,
    ))

    registry.register(Skill(
        id="wake_greeting",
        description="JARVIS wake-up greeting",
        intent_tags=[
            "wake_greeting", "greet_assistant", "greeting",
            "wake_up", "hello_jarvis", "hey_jarvis",
        ],
        required_params=[],
        execute=_execute_wake_greeting,
    ))
