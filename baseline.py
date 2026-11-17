"""Evaluate a spymaster on a FIXED set of evaluation boards, several runs.

The same boards are used in every run (and later for the trained model), so
differences between runs show the randomness of the model (sampling), and
differences between models show what training changed.

Examples
--------
# Baseline: local Qwen spymaster (Ollama) vs Llama 3.3 70B guesser (Inworld API).
# Set INWORLD_API_KEY first. Copy the exact model ID from the Inworld portal.
python baseline.py --model qwen2.5:1.5b --guesser-provider inworld --guesser-model <inworld-llama-id> --runs 5

# Later: the trained model, same boards, same guesser
python baseline.py --model codenames-spymaster --guesser-provider inworld --guesser-model <inworld-llama-id> --runs 5

# Sanity check without LLM: you are the spymaster, random guesser
python baseline.py --spymaster human --guesser random --n 3 --runs 1

# Play yourself as guesser against the LLM spymaster
python baseline.py --model qwen2.5:1.5b --guesser human --n 5 --runs 1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from codenames import generate_board, play_turn
from codenames.board import EVAL_SEED
from codenames.players import (
    PROVIDERS, HumanGuesser, HumanSpymaster, LLMClient, LLMGuesser, LLMSpymaster, RandomGuesser,
)


def run_once(spymaster, guesser, n: int, seed: int, reveal: float, out: Path, tag: str) -> dict:
    """Play n boards, write per-board details to `out`, return summary stats."""
    rows = []
    with out.open("w", encoding="utf-8") as f:
        for i in range(n):
            board = generate_board(seed=seed + i, reveal_fraction=reveal)
            clue_text, res = play_turn(board, spymaster, guesser)
            row = {"seed": seed + i, "board": board.to_dict(), "raw": clue_text, **res.to_dict()}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            status = res.error or f"{res.guesses} -> {res.outcomes}"
            print(f"{tag} [{i + 1}/{n}] {res.clue!s:>12} {res.number!s:>2} | {res.reward:+.1f} | {status}")
    return summarize(rows)


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


def main() -> None:
    # Read API keys from .env next to this script (never committed, see .gitignore).
    # override=True: a value in .env wins over an (empty) variable already set in the shell.
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=5, help="how many times to play the same boards")
    p.add_argument("--n", type=int, default=50, help="boards per run")
    p.add_argument("--seed", type=int, default=EVAL_SEED, help="first board seed (eval set; keep training seeds separate)")
    p.add_argument("--reveal", type=float, default=0.0, help="fraction of words already revealed (mid-game boards)")
    p.add_argument("--spymaster", choices=["llm", "human"], default="llm")
    p.add_argument("--guesser", choices=["llm", "human", "random"], default="llm")
    p.add_argument("--model", default="qwen2.5:1.5b", help="spymaster model name")
    p.add_argument("--guesser-model", default=None, help="guesser model (default: same as --model)")
    p.add_argument("--provider", choices=list(PROVIDERS), default="ollama", help="spymaster API provider")
    p.add_argument("--guesser-provider", choices=list(PROVIDERS), default="ollama")
    p.add_argument("--base-url", default=None, help="override the spymaster provider URL")
    p.add_argument("--guesser-base-url", default=None, help="override the guesser provider URL")
    p.add_argument("--label", default=None, help="name of the results folder (default: model name)")
    args = p.parse_args()

    spymaster = (HumanSpymaster() if args.spymaster == "human"
                 else LLMSpymaster(LLMClient(args.model, args.provider, args.base_url)))
    guesser_model = args.guesser_model or args.model
    if args.guesser == "human":
        guesser = HumanGuesser()
    elif args.guesser == "random":
        guesser = RandomGuesser(seed=args.seed)
    else:
        guesser = LLMGuesser(LLMClient(
            guesser_model, args.guesser_provider, args.guesser_base_url,
            temperature=0.0,  # deterministic guesser = less noisy reward
        ))

    sm_name = args.model if args.spymaster == "llm" else "human"
    g_name = guesser_model if args.guesser == "llm" else args.guesser
    label = args.label or re.sub(r"[^A-Za-z0-9._-]+", "_", f"{sm_name}__vs__{g_name}")
    out_dir = Path("eval_results") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for k in range(1, args.runs + 1):
        stats = run_once(spymaster, guesser, args.n, args.seed, args.reveal,
                         out_dir / f"run_{k}.jsonl", tag=f"run {k}/{args.runs}")
        runs.append({"run": k, **stats})
        print(f"--> run {k}: mean reward {stats['mean_reward']:+.2f}, valid {stats['valid_rate']:.0%}\n")

    # per-run table
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    # across-run summary: mean / std / min / max per metric
    metrics = [k for k in runs[0] if k not in ("run", "boards")]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in runs]
        agg[m] = {"mean": statistics.mean(vals),
                  "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                  "min": min(vals), "max": max(vals)}
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "spymaster": sm_name, "guesser": g_name,
        "runs": args.runs, "boards_per_run": args.n, "first_seed": args.seed, "reveal": args.reveal,
        "per_run": runs, "aggregate": agg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== summary over runs =====")
    print(f"{'metric':<18}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    for m, a in agg.items():
        print(f"{m:<18}{a['mean']:>8.2f}{a['std']:>8.2f}{a['min']:>8.2f}{a['max']:>8.2f}")
    print(f"\nsaved to {out_dir}/ (runs.csv, summary.json, run_*.jsonl)")


if __name__ == "__main__":
    main()
