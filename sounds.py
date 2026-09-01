"""Terminal sound notifications. Stdlib only (typer for the preview CLI).

All sound config lives here so a different notification mechanism later
means editing only this file. On non-macOS both events fall back to the
same ASCII terminal bell (audibility depends on the terminal's settings).

Preview from the terminal: uv run python sounds.py order|fill
"""

from __future__ import annotations

import subprocess
import sys

import typer

# macOS built-in sounds (see /System/Library/Sounds/)
ORDER_SOUND = "/System/Library/Sounds/Pop.aiff"  # order submitted
FILL_SOUND = "/System/Library/Sounds/Glass.aiff"  # order filled


def play_order_sound() -> None:
    """Play the order-submitted sound. Best-effort: never raises, never blocks."""
    _play(ORDER_SOUND)


def play_fill_sound() -> None:
    """Play the order-fill sound. Best-effort: never raises, never blocks."""
    _play(FILL_SOUND)


def _play(sound: str) -> None:
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["afplay", sound],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    print("\a", end="", flush=True)


# --- Preview CLI: uv run python sounds.py order|fill ---
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def order() -> None:
    """Play the order-submitted sound."""
    play_order_sound()


@app.command()
def fill() -> None:
    """Play the order-fill sound."""
    play_fill_sound()


if __name__ == "__main__":
    app()
