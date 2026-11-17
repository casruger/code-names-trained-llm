"""Clue legality checks. Illegal clues are the #1 way an RL model will cheat."""
from __future__ import annotations

import re

from .board import Board
from .prompts import EXAMPLE_CLUE

_WORD_RE = re.compile(r"^[a-z]{2,}$")


def _stem(w: str) -> str:
    """Very rough stemming so 'sharks' / 'shark' / 'sharking' count as the same word."""
    for suffix in ("ing", "ers", "er", "es", "s", "ed"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def validate_clue(clue: str, number: int, board: Board) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    clue = clue.lower().strip()
    if not _WORD_RE.match(clue):
        return False, "clue must be a single word of at least 2 letters a-z"
    if not 1 <= number <= 9:
        return False, "number must be between 1 and 9"
    if clue == EXAMPLE_CLUE:
        return False, f"clue '{clue}' copied from the example in the prompt"
    n_hidden_team = len(board.hidden_by_role("team"))
    if number > n_hidden_team:
        return False, f"number {number} > remaining team words {n_hidden_team}"
    for w in board.words:  # also revealed words: classic rule says any visible word
        if clue == w:
            return False, f"clue '{clue}' is a word on the board"
        # Compound/part-word check. Only for parts of >= 4 letters, otherwise
        # 'bat' would forbid 'battery' and 'tea' would forbid 'teacher'.
        shorter = min(w, clue, key=len)
        if len(shorter) >= 4 and (w in clue or clue in w):
            return False, f"clue '{clue}' contains or is contained in board word '{w}'"
        if _stem(clue) == _stem(w):
            return False, f"clue '{clue}' is a form of board word '{w}'"
    return True, "ok"
