"""One spymaster turn = one RL episode: board -> clue -> guesses -> reward."""
from __future__ import annotations

from .board import Board
from .parsing import parse_clue
from .reward import ILLEGAL_CLUE, INVALID_FORMAT, TurnResult, score_guesses
from .rules import validate_clue


def score_clue_text(board: Board, clue_text: str, guesser) -> TurnResult:
    """Score a raw spymaster output. This is exactly the reward used in training."""
    parsed = parse_clue(clue_text)
    if parsed is None:
        return TurnResult(reward=INVALID_FORMAT, error=f"unparseable: {clue_text[:80]!r}")
    clue, number = parsed
    ok, reason = validate_clue(clue, number, board)
    if not ok:
        return TurnResult(reward=ILLEGAL_CLUE, clue=clue, number=number, error=reason)
    guesses = guesser.guess(board, clue, number)
    result = score_guesses(board, clue, number, guesses)
    result.guesser_raw = getattr(guesser, "last_raw", None)
    return result


def play_turn(board: Board, spymaster, guesser) -> tuple[str, TurnResult]:
    clue_text = spymaster.give_clue(board)
    return clue_text, score_clue_text(board, clue_text, guesser)
