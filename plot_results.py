"""
Turn an experiments.py summary CSV into the figures for question 4.

For each environment family present in the file, draws two panels side by side:

    success rate vs the family parameter   (error bars: Wilson interval)
    search time  vs the family parameter   (error bars: t interval, solved runs)

One line per planner, so the comparison is direct.

    python plot_results.py --summary results_summary.csv --out figures

Uses only csv + matplotlib, so it needs no pandas and works headless.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# Axis labels per family; anything else falls back to a generic label.
X_LABELS = {
    "passage": "opening width (smaller = harder)",
    "clutter": "number of obstacles",
}


def _float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_summary(path: str) -> dict[str, dict[str, list[dict]]]:
    """Group summary rows as family -> planner -> [row, ...], sorted by param."""
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {k: _float(v) for k, v in row.items()
                      if k not in ("env", "family", "planner")}
            parsed["env"] = row["env"]
            grouped[row["family"]][row["planner"]].append(parsed)

    for planners in grouped.values():
        for rows in planners.values():
            rows.sort(key=lambda r: r["param"])
    return grouped


def _series(rows: list[dict], mean_key: str, lo_key: str, hi_key: str):
    """x, y and asymmetric error bars, skipping rows with no data."""
    xs, ys, lo, hi = [], [], [], []
    for r in rows:
        y = r[mean_key]
        if y != y:                      # NaN: no solved runs, nothing to plot
            continue
        low, high = r[lo_key], r[hi_key]
        xs.append(x := r["param"])
        ys.append(y)
        # NaN bounds mean "only one solved run, spread unknown" - plot the point
        # with no whisker rather than dropping it.
        lo.append(0.0 if low != low else max(0.0, y - low))
        hi.append(0.0 if high != high else max(0.0, high - y))
    return xs, ys, [lo, hi]


def plot_family(family: str, planners: dict[str, list[dict]], out_dir: str,
                k: str = "") -> str:
    fig, (ax_rate, ax_time) = plt.subplots(1, 2, figsize=(11, 4))

    for planner in sorted(planners):
        rows = planners[planner]

        xs, ys, err = _series(rows, "success_rate", "sr_lo", "sr_hi")
        ax_rate.errorbar(xs, ys, yerr=err, marker="o", capsize=3,
                         linewidth=1.6, label=planner)

        xs, ys, err = _series(rows, "time_mean", "time_lo", "time_hi")
        ax_time.errorbar(xs, ys, yerr=err, marker="s", capsize=3,
                         linewidth=1.6, label=planner)

    label = X_LABELS.get(family, "parameter")
    ax_rate.set_ylabel("success rate")
    ax_rate.set_ylim(-0.05, 1.05)
    ax_time.set_ylabel("search time (s), solved runs only")

    for ax in (ax_rate, ax_time):
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    runs = int(planners[sorted(planners)[0]][0].get("runs", 0))
    fig.suptitle(f"{family} family{k}   ({runs} runs per point, 95% CI)")
    fig.tight_layout()

    path = os.path.join(out_dir, f"{family}{k.replace(', ', '_').replace('=', '')}.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment summaries.")
    parser.add_argument("--summary", default="results_summary.csv")
    parser.add_argument("--out", default="figures", help="output directory")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grouped = load_summary(args.summary)
    if not grouped:
        raise SystemExit(f"no rows in {args.summary}")

    for family, planners in sorted(grouped.items()):
        first = planners[sorted(planners)[0]][0]
        k = f", K={int(first['K'])}" if first.get("K") == first.get("K") else ""
        plot_family(family, planners, args.out, k=k)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
