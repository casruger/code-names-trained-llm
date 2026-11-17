"""Summary statistics over a list of played boards (rows as written to run_k.jsonl)."""
from __future__ import annotations

import statistics


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    valid = [r for r in rows if not r["error"]]
    hit = lambda role: sum(role in r["outcomes"] for r in valid) / n
    empty = sum(1 for r in valid if not r["guesses"])
    return {
        "boards": n,
        "mean_reward": statistics.mean(r["reward"] for r in rows),
        "valid_rate": len(valid) / n,
        "unparseable": sum(1 for r in rows if r["error"] and r["error"].startswith("unparseable")),
        "illegal": sum(1 for r in rows if r["error"] and not r["error"].startswith("unparseable")),
        "mean_number": statistics.mean(r["number"] for r in valid) if valid else 0.0,
        "mean_correct": statistics.mean(r["n_correct"] for r in valid) if valid else 0.0,
        "guesser_empty_rate": empty / n,
        "all_correct_rate": sum(r["n_correct"] == r["number"] for r in valid) / n,
        "assassin_rate": hit("assassin"),
        "opponent_rate": hit("opponent"),
        "neutral_rate": hit("neutral"),
    }
