"""Plot how the spymaster changed during GRPO training, from the reward log.

Example
-------
python plot_training.py training_results/grpo-v1/reward_log.jsonl --out docs/training.png
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = "#2a78d6"


def per_step(path: Path) -> list[dict]:
    """Group the log by training step. If a step appears twice (a restart), keep the last one."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        groups[r["call"]].append(r)
    steps = []
    for step in sorted(groups):
        rows = groups[step][-32:] if len(groups[step]) > 32 else groups[step]
        valid = [r for r in rows if not r["error"]]
        steps.append({
            "step": step,
            "reward": statistics.mean(r["reward"] for r in rows),
            "valid": len(valid) / len(rows),
            "number_1": sum(r["number"] == 1 for r in valid) / len(valid) if valid else 0.0,
        })
    return steps


def rolling(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        part = values[max(0, i - window + 1): i + 1]
        out.append(sum(part) / len(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="reward_log.jsonl written during training")
    p.add_argument("--out", default="docs/training.png")
    p.add_argument("--window", type=int, default=10, help="steps in the moving average")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    steps = per_step(Path(args.log))
    x = [s["step"] for s in steps]
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    panels = [
        ("reward", "Mean reward per clue", False),
        ("valid", "Valid clues", True),
        ("number_1", "Clues for only ONE word (NUMBER = 1)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    for ax, (key, title, pct) in zip(axes, panels):
        y = [s[key] for s in steps]
        ax.set_facecolor(SURFACE)
        ax.scatter(x, y, s=6, color=COLOR, alpha=0.18, linewidths=0, zorder=2)
        ax.plot(x, rolling(y, args.window), color=COLOR, linewidth=2, zorder=3)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("training step", color=MUTED)
        if pct:
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        else:
            ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(length=0)
    fig.suptitle(f"During training  ·  32 clues per step on training boards  ·  line = {args.window}-step average",
                 x=0.01, ha="left", fontsize=11, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"chart saved to {args.out}")


if __name__ == "__main__":
    main()
