"""Compare two evaluation folders (made by baseline.py) and draw a results chart.

Example
-------
python compare.py \
    eval_results/qwen2.5_1.5b__vs__deepinfra_meta-llama_Llama-3.3-70B-Instruct-Turbo \
    eval_results/codenames-spymaster__vs__deepinfra_meta-llama_Llama-3.3-70B-Instruct-Turbo \
    --names "Before training" "After training" --out docs/results.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Metrics shown in the table: (key in summary.json, label, format)
METRICS = [
    ("mean_reward", "Mean reward per clue", "{:+.2f}"),
    ("valid_rate", "Valid clues", "{:.0%}"),
    ("mean_number", "Words per clue (the NUMBER)", "{:.2f}"),
    ("mean_correct", "Correct guesses per valid clue", "{:.2f}"),
    ("all_correct_rate", "All words of the clue guessed", "{:.0%}"),
    ("assassin_rate", "Hit the assassin", "{:.0%}"),
    ("opponent_rate", "Hit an opponent word", "{:.0%}"),
    ("neutral_rate", "Hit a neutral word", "{:.0%}"),
]

# Chart style (light surface, two categorical colours, quiet axes)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834"]  # 1 = before, 2 = after


def load(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))


def table(summaries: list[dict], names: list[str]) -> str:
    head = "| Metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names) + "\n"
    rows = []
    for key, label, fmt in METRICS:
        cells = []
        for s in summaries:
            a = s["aggregate"][key]
            cell = fmt.format(a["mean"])
            if key == "mean_reward":
                cell += f" ({fmt.format(a['min'])} to {fmt.format(a['max'])})"
            cells.append(cell)
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def chart(summaries: list[dict], names: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    panels = [  # (key, title, as percent)
        ("mean_reward", "Mean reward per clue\n(higher is better)", False),
        ("valid_rate", "Valid clues", True),
        ("assassin_rate", "Clues that hit the assassin", True),
        ("mean_number", "Words per clue\n(the NUMBER)", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.6), facecolor=SURFACE)
    for ax, (key, title, pct) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        means = [s["aggregate"][key]["mean"] for s in summaries]
        xs = range(len(summaries))
        ax.bar(xs, means, width=0.55, color=COLORS[: len(summaries)], zorder=2)
        # every run as a small dot, so the spread between runs is visible
        for x, s in zip(xs, summaries):
            vals = [r[key] for r in s["per_run"]]
            ax.scatter([x] * len(vals), vals, s=14, color=INK, alpha=0.55, zorder=3, linewidths=0)
        runs = [[r[key] for r in s["per_run"]] for s in summaries]
        lo = min(0, *means, *(min(v) for v in runs))
        hi = max(0, *means, *(max(v) for v in runs))
        pad = (hi - lo) * 0.2 or 0.1
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
        for x, m, v in zip(xs, means, runs):
            label = f"{m:.0%}" if pct else f"{m:+.2f}" if key == "mean_reward" else f"{m:.2f}"
            # put the label just outside the bar AND outside the run dots
            y, dy, va = (max(m, *v), 5, "bottom") if m >= 0 else (min(m, *v), -5, "top")
            ax.annotate(label, (x, y), xytext=(0, dy), textcoords="offset points",
                        ha="center", va=va, fontsize=10, color=INK, fontweight="semibold")
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xticks(list(xs), names)
        if pct:
            ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= 1.0001])
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(length=0)
    fig.suptitle("Spymaster before and after GRPO training  ·  50 evaluation boards × 5 runs (dots = single runs)",
                 x=0.01, ha="left", fontsize=11, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"chart saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs=2, help="two eval_results/<...> folders: before, after")
    p.add_argument("--names", nargs=2, default=["Before training", "After training"])
    p.add_argument("--out", default="docs/results.png", help="where to save the chart")
    args = p.parse_args()

    summaries = [load(f) for f in args.folders]
    print(table(summaries, args.names))
    chart(summaries, args.names, Path(args.out))


if __name__ == "__main__":
    main()
