"""Glue between the Codenames game and TRL's GRPOTrainer (used by train_grpo.ipynb).

Kept in the package (not only in the notebook) so it can be tested without a GPU.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .board import EVAL_SEED, Board, generate_board
from .game import score_clue_text
from .metrics import summarize
from .prompts import spymaster_messages
from .reward import TurnResult


# --------------------------------------------------------------------------- data

def build_dataset(n_boards: int, seed: int = 0, reveal_fraction: float = 0.0):
    """Training boards. Seeds stay below EVAL_SEED, so evaluation boards are never trained on."""
    from datasets import Dataset

    if seed + n_boards > EVAL_SEED:
        raise ValueError("training seeds would overlap with the evaluation boards")
    rows = []
    for i in range(n_boards):
        board = generate_board(seed=seed + i, reveal_fraction=reveal_fraction)
        rows.append({"prompt": spymaster_messages(board), "board": json.dumps(board.to_dict())})
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- guesser

class CachedGuesser:
    """Wraps a (deterministic, temperature 0) guesser and caches its answers.

    GRPO samples several clues per board; identical clues on the same board give
    identical guesses, so the cache saves API calls (= money and time).
    """

    def __init__(self, guesser):
        self.guesser = guesser
        self.cache: dict = {}
        self.key_locks: dict = {}
        self.lock = threading.Lock()
        self.hits = self.misses = 0
        self._local = threading.local()

    @property
    def last_raw(self):
        return getattr(self._local, "last_raw", None)

    def guess(self, board: Board, clue: str, number: int) -> list[str]:
        key = (tuple(board.hidden_words), tuple(sorted(board.revealed)), clue, number)
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:  # identical requests in parallel wait for the first one -> 1 API call
            cached = self.cache.get(key)
            if cached is None:
                self.misses += 1
                guesses = self.guesser.guess(board, clue, number)
                cached = (list(guesses), getattr(self.guesser, "last_raw", None))
                self.cache[key] = cached
            else:
                self.hits += 1
        self._local.last_raw = cached[1]
        return list(cached[0])


# --------------------------------------------------------------------------- reward

def completion_text(completion) -> str:
    """TRL passes plain strings, or (chat datasets) a list of message dicts."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


class CodenamesReward:
    """Reward function for TRL's GRPOTrainer: one score per sampled clue.

    Also writes every sample to a JSONL log, so you can see *what* the model
    learns (e.g. reward hacking: the same clue over and over).
    """

    def __init__(self, guesser, max_workers: int = 16, log_path: str | Path | None = None):
        self.__name__ = "codenames"          # TRL uses this name in its logs
        self.guesser = CachedGuesser(guesser)
        self.max_workers = max_workers
        self.log_path = Path(log_path) if log_path else None
        self.calls = 0

    def _score(self, text: str, board: Board) -> TurnResult:
        for attempt in range(3):
            try:
                return score_clue_text(board, text, self.guesser)
            except Exception as e:  # API hiccup: retry, never crash a long training run
                err = e
                time.sleep(2 * (attempt + 1))
        return TurnResult(reward=0.0, error=f"api_error: {err!r}"[:200])

    def __call__(self, prompts, completions, board, **kwargs) -> list[float]:
        texts = [completion_text(c) for c in completions]
        boards = [Board.from_dict(json.loads(b)) for b in board]
        with ThreadPoolExecutor(self.max_workers) as pool:
            results = list(pool.map(self._score, texts, boards))
        self.calls += 1
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                for text, r in zip(texts, results):
                    f.write(json.dumps({"call": self.calls, "raw": text, **r.to_dict()}) + "\n")
        return [r.reward for r in results]


def read_reward_log(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------- evaluation

def evaluate_hf(model, tokenizer, guesser, n: int = 50, seed: int = EVAL_SEED,
                temperature: float = 0.7, batch_size: int = 25, max_new_tokens: int = 32,
                max_workers: int = 16) -> tuple[dict, list[dict]]:
    """Play the evaluation boards with a Hugging Face model as spymaster.

    Same boards, prompts and scoring as baseline.py. Returns (summary, rows).
    """
    import torch

    boards = [generate_board(seed=seed + i) for i in range(n)]
    texts: list[str] = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for i in range(0, n, batch_size):
        chunk = boards[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(spymaster_messages(b), tokenize=False,
                                                 add_generation_prompt=True) for b in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=temperature > 0, temperature=temperature or None,
                                 max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        texts += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def play(i):
        res = score_clue_text(boards[i], texts[i], guesser)
        return {"seed": seed + i, "board": boards[i].to_dict(), "raw": texts[i], **res.to_dict()}

    with ThreadPoolExecutor(max_workers) as pool:
        rows = list(pool.map(play, range(n)))
    return summarize(rows), rows
