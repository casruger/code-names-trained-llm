"""Turn raw model text into structured clues / guesses. Lenient on purpose:
small models are sloppy with formatting, especially early in training."""
from __future__ import annotations

import re

from .board import Board

# Tolerate decoration around the value: **bold**, <angle brackets>, [ ], quotes.
_DECOR = r"[\s*<\[\"'`]*"
_CLUE_RE = re.compile(r"CLUE\s*:" + _DECOR + r"([A-Za-z]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"NUMBER\s*:" + _DECOR + r"(\d+)", re.IGNORECASE)
_GUESS_RE = re.compile(r"GUESSES?\s*:\s*(.+)", re.IGNORECASE)


def parse_clue(text: str) -> tuple[str, int] | None:
    """Return (clue, number) from the LAST CLUE/NUMBER lines, or None."""
    clues = _CLUE_RE.findall(text)
    nums = _NUM_RE.findall(text)
    if not clues or not nums:
        return None
    return clues[-1].lower(), int(nums[-1])


def parse_guesses(text: str, board: Board, max_guesses: int) -> list[str]:
    """Return valid, hidden, de-duplicated board words in the order given."""
    m = _GUESS_RE.findall(text)
    raw = m[-1] if m else text
    hidden = set(board.hidden_words)
    out: list[str] = []
    for tok in re.split(r"[,\n;]+", raw):
        w = re.sub(r"[^a-z]", "", tok.lower())
        if w in hidden and w not in out:
            out.append(w)
        if len(out) >= max_guesses:
            break
    return out
