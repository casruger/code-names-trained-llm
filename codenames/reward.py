"""Reward function. This is where you steer what strategy the model learns.

Tweak the constants and watch the behaviour change, e.g. a huge assassin
penalty makes the model play safe '..., 1' clues.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .board import ASSASSIN, NEUTRAL, OPPONENT, TEAM, Board

REWARDS = {
    TEAM: 1.0,
    NEUTRAL: -0.5,
    OPPONENT: -1.0,
    ASSASSIN: -5.0,
}
NO_HIT = -1.0           # legal clue, but not a single team word guessed (useless clue)
INVALID_FORMAT = -3.0   # model output could not be parsed
ILLEGAL_CLUE = -3.0     # clue breaks the rules (on board, part of a word, copied example)

# Resulting scale, worst to best:
#   assassin -6 | illegal / unparseable -3 | opponent first -2 | neutral first -1.5
#   nothing guessed -1 | 1 correct +1 | 2 correct +2 | ...


@dataclass
class TurnResult:
    reward: float
    clue: str | None = None
    number: int | None = None
    guesses: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)   # role of each guessed word
    n_correct: int = 0
    error: str | None = None
    guesser_raw: str | None = None                       # unparsed guesser answer, for debugging

    def to_dict(self) -> dict:
        return asdict(self)


def score_guesses(board: Board, clue: str, number: int, guesses: list[str]) -> TurnResult:
    """Guesses are taken in order; the turn ends at the first non-team word."""
    reward, outcomes, n_correct = 0.0, [], 0
    for g in guesses:
        role = board.roles[g]
        outcomes.append(role)
        reward += REWARDS[role]
        if role != TEAM:
            break
        n_correct += 1
    if n_correct == 0:
        reward += NO_HIT
    return TurnResult(
        reward=reward, clue=clue, number=number,
        guesses=guesses[: len(outcomes)], outcomes=outcomes, n_correct=n_correct,
    )
