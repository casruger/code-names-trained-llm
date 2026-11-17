"""Board generation. Everything is from the perspective of the spymaster's team."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .words import WORDS

TEAM, OPPONENT, NEUTRAL, ASSASSIN = "team", "opponent", "neutral", "assassin"

# Boards with seed >= EVAL_SEED are reserved for evaluation. Training must use
# seeds below this, so the model is never tested on boards it trained on.
EVAL_SEED = 1_000_000


@dataclass
class Board:
    roles: dict[str, str]              # word -> role, in display order
    revealed: set[str] = field(default_factory=set)

    @property
    def words(self) -> list[str]:
        return list(self.roles)

    @property
    def hidden_words(self) -> list[str]:
        return [w for w in self.roles if w not in self.revealed]

    def hidden_by_role(self, role: str) -> list[str]:
        return [w for w in self.hidden_words if self.roles[w] == role]

    def to_dict(self) -> dict:
        return {"roles": self.roles, "revealed": sorted(self.revealed)}

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(roles=dict(d["roles"]), revealed=set(d.get("revealed", [])))


def generate_board(
    seed: int | None = None,
    n_team: int = 9,
    n_opponent: int = 8,
    n_neutral: int = 7,
    n_assassin: int = 1,
    reveal_fraction: float = 0.0,
) -> Board:
    """Random 25-word board. reveal_fraction > 0 simulates a mid-game state
    (some words already guessed), which gives more varied training boards."""
    rng = random.Random(seed)
    total = n_team + n_opponent + n_neutral + n_assassin
    words = rng.sample(WORDS, total)
    roles_list = (
        [TEAM] * n_team + [OPPONENT] * n_opponent
        + [NEUTRAL] * n_neutral + [ASSASSIN] * n_assassin
    )
    rng.shuffle(roles_list)
    board = Board(roles=dict(zip(words, roles_list)))

    if reveal_fraction > 0:
        # Never reveal the assassin, and always leave at least 2 team words.
        candidates = [w for w in words if board.roles[w] != ASSASSIN]
        rng.shuffle(candidates)
        k = int(reveal_fraction * len(candidates))
        for w in candidates[:k]:
            if board.roles[w] == TEAM and len(board.hidden_by_role(TEAM)) <= 2:
                continue
            board.revealed.add(w)
    return board
