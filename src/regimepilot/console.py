"""Console output that survives a terminal which cannot encode every character.

Windows pipes and consoles often use cp1252. An LLM thesis or a news headline
can contain a character outside it (the first one seen in production was a
non-breaking hyphen, U+2011), and a strict stream then raises
``UnicodeEncodeError`` from ``print`` after all the work was done. Replacing
the character with ``?`` on the way out loses nothing that matters: the JSON
output escapes non-ASCII anyway, and the summary is for a human.
"""

from __future__ import annotations

import sys


def tolerant_console() -> None:
    """Make stdout and stderr replace unencodable characters instead of raising.

    A stream that cannot be reconfigured (a test harness capture, a custom
    host stream) is left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
