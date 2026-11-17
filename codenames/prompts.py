"""Prompts shared by baseline and training, so the model is trained on exactly
what it is evaluated on.

Small models copy format placeholders literally ('CLUE: <word>') and tend to
repeat a board word as the clue, so the prompts use a concrete worked example
(one-shot: an example user turn + ideal assistant answer) instead of
abstract rules only.
"""
from __future__ import annotations

from .board import Board

SPYMASTER_SYSTEM = (
    "You are the spymaster in the board game Codenames.\n"
    "Think of ONE new English word (the clue) that is related to several of YOUR "
    "words, but NOT to the opponent words, neutral words or the assassin.\n"
    "Important rules:\n"
    "- The clue must NOT be any word from the lists, and not part of one.\n"
    "- NUMBER is how many of YOUR words the clue relates to.\n"
    "Reply with exactly two lines and nothing else: a line starting with "
    "'CLUE:' followed by your clue, and a line starting with 'NUMBER:' "
    "followed by the number."
)

GUESSER_SYSTEM = (
    "You are the guesser in the board game Codenames. Your spymaster gives a "
    "one-word clue and a number. Pick the board words that best match the clue, "
    "most confident first. Pick at most NUMBER words. Picking a wrong word ends "
    "your turn, and one word on the board is a deadly assassin.\n"
    "Reply with exactly one line and nothing else: 'GUESSES:' followed by "
    "your words, separated by commas."
)

# One worked example per role, shown as an earlier turn of the conversation.
_EXAMPLE_SPYMASTER_USER = (
    "Your words: whale, shark, piano, castle\n"
    "Opponent words: guitar, desert\n"
    "Neutral words: pencil, tiger\n"
    "Assassin: island"
)
# The example clue is forbidden in real games (see rules.py), so copying it never pays off.
EXAMPLE_CLUE = "fins"
_EXAMPLE_SPYMASTER_ANSWER = f"CLUE: {EXAMPLE_CLUE}\nNUMBER: 2"   # whale + shark; not on the board

_EXAMPLE_GUESSER_USER = (
    "Board words: whale, guitar, shark, pencil, island, castle\n"
    f"Clue: {EXAMPLE_CLUE}\n"
    "Number: 2"
)
_EXAMPLE_GUESSER_ANSWER = "GUESSES: whale, shark"


def spymaster_user_prompt(board: Board) -> str:
    def fmt(role: str) -> str:
        return ", ".join(board.hidden_by_role(role)) or "-"

    return (
        f"Your words: {fmt('team')}\n"
        f"Opponent words: {fmt('opponent')}\n"
        f"Neutral words: {fmt('neutral')}\n"
        f"Assassin: {fmt('assassin')}"
    )


def guesser_user_prompt(board: Board, clue: str, number: int) -> str:
    return (
        f"Board words: {', '.join(board.hidden_words)}\n"
        f"Clue: {clue}\n"
        f"Number: {number}"
    )


def spymaster_messages(board: Board) -> list[dict]:
    return [
        {"role": "system", "content": SPYMASTER_SYSTEM},
        {"role": "user", "content": _EXAMPLE_SPYMASTER_USER},
        {"role": "assistant", "content": _EXAMPLE_SPYMASTER_ANSWER},
        {"role": "user", "content": spymaster_user_prompt(board)},
    ]


def guesser_messages(board: Board, clue: str, number: int) -> list[dict]:
    return [
        {"role": "system", "content": GUESSER_SYSTEM},
        {"role": "user", "content": _EXAMPLE_GUESSER_USER},
        {"role": "assistant", "content": _EXAMPLE_GUESSER_ANSWER},
        {"role": "user", "content": guesser_user_prompt(board, clue, number)},
    ]
