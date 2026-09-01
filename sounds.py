"""Terminal sound notifications. Stdlib only.

All sound config lives here so a different notification mechanism later
means editing only this file.
"""

from __future__ import annotations

import subprocess
import sys

FILL_SOUND = "/System/Library/Sounds/Glass.aiff"  # macOS built-in sound


def play_fill_sound() -> None:
    """Play the order-fill sound. Best-effort: never raises, never blocks."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["afplay", FILL_SOUND],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    # non-macOS, or afplay unavailable: ASCII terminal bell (audibility
    # depends on the terminal's bell setting)
    print("\a", end="", flush=True)
