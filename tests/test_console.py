"""Console tests: a terminal that cannot encode a character must not crash a CLI.

Windows pipes and consoles often use cp1252. An LLM thesis or a news headline
can contain characters outside it (a non-breaking hyphen, U+2011, was the first
one seen in production), and a strict stream then raises UnicodeEncodeError
from ``print`` after all the work was done.
"""

import io
import sys

import pytest

from regimepilot.console import tolerant_console

SAMPLE = "non‑breaking hyphen"


def cp1252_stream(buffer):
    return io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", write_through=True)


def test_a_strict_cp1252_stream_really_does_raise_on_the_sample():
    stream = cp1252_stream(io.BytesIO())
    with pytest.raises(UnicodeEncodeError):
        print(SAMPLE, file=stream)


def test_tolerant_console_replaces_unencodable_characters_instead_of_raising(monkeypatch):
    out, err = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(sys, "stdout", cp1252_stream(out))
    monkeypatch.setattr(sys, "stderr", cp1252_stream(err))

    tolerant_console()
    print(SAMPLE)
    print(SAMPLE, file=sys.stderr)

    assert b"non?breaking hyphen" in out.getvalue()
    assert b"non?breaking hyphen" in err.getvalue()


def test_tolerant_console_leaves_a_utf8_stream_able_to_write_the_character(monkeypatch):
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="utf-8", write_through=True))

    tolerant_console()
    print(SAMPLE)

    assert SAMPLE.encode("utf-8") in out.getvalue()


def test_tolerant_console_tolerates_a_stream_without_reconfigure(monkeypatch):
    """Test harnesses and some hosts replace stdout with objects that cannot reconfigure."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    tolerant_console()  # must not raise
    print("still fine")

    assert "still fine" in sys.stdout.getvalue()
