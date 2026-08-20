"""
Batch experiment runner for the complexity study (Part B, question 4).

Runs every environment in a directory against one or more planners, repeats
each combination with different seeds, and writes two CSV files:

  <out>_raw.csv       one row per run - env, planner, seed, success, timings
  <out>_summary.csv   one row per (env, planner) - success rate and the
                      mean + 95% CI of each metric

It also prints the summary as a table you can read directly or paste into the
report. Rows are flushed to disk as they complete, so a long sweep can be
inspected while it is still running.

    python experiments.py --envs envs --drones 5 --runs 30
    python experiments.py --envs envs --drones 5 --runs 30 --planners rrt
    python experiments.py --envs envs --drones 2 --runs 30 --out results_k2

Success rate is reported with a Wilson interval rather than the t-interval used
for the continuous metrics: it is a proportion, and at 0/30 or 30/30 a
t-interval would claim zero uncertainty, which is wrong.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time

import numpy as np

from rrt_planner import rrt_plan, _mean_and_ci
from rrt_connect import rrt_connect_plan

PLANNERS = {"rrt": rrt_plan, "rrt-connect": rrt_connect_plan}


def make_sim(num_drones: int, environment_file: str):
    """Build a simulator for one environment."""
    from multi_drone import MultiDrone
    return MultiDrone(num_drones=num_drones, environment_file=environment_file)


def describe(path: str) -> tuple[str, float | str]:
    """Split 'passage_w030.yaml' into ('passage', 30.0). Unrecognised names get
    ('other', '')."""
    name = os.path.basename(path)
    match = re.match(r"([a-zA-Z]+)_[a-zA-Z](\d+)\.ya?ml$", name)
    if match:
        return match.group(1), float(match.group(2))
    return "other", ""


def wilson_ci(successes: int, trials: int) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if trials == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import binomtest
        lo, hi = binomtest(successes, trials).proportion_ci(
            confidence_level=0.95, method="wilson")
        return float(lo), float(hi)
    except Exception:
        z, n, p = 1.96, trials, successes / trials
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)


RAW_FIELDS = ["env", "family", "param", "planner", "K", "seed", "success",
              "search_time", "total_time", "path_length", "waypoints",
              "nodes", "iterations", "error"]

SUMMARY_FIELDS = ["env", "family", "param", "planner", "K", "runs", "successes",
                  "success_rate", "sr_lo", "sr_hi",
                  "time_mean", "time_lo", "time_hi",
                  "len_mean", "len_lo", "len_hi",
                  "nodes_mean", "nodes_lo", "nodes_hi"]


def run_cell(sim, planner_name: str, planner, runs: int, time_limit: float,
             smooth: bool, env_name: str, family: str, param, K: int,
             raw_writer, raw_file) -> dict:
    """Run one (environment, planner) combination `runs` times."""
    successes, times, lengths, nodes = 0, [], [], []

    for seed in range(runs):
        row = dict(env=env_name, family=family, param=param, planner=planner_name,
                   K=K, seed=seed, success=0, search_time="", total_time="",
                   path_length="", waypoints="", nodes="", iterations="", error="")
        try:
            path, s = planner(sim, time_limit=time_limit, seed=seed, smooth=smooth)
            row.update(success=int(s["success"]),
                       search_time=round(s["search_time"], 4),
                       total_time=round(s.get("total_time", s["search_time"]), 4),
                       nodes=s["nodes"], iterations=s.get("iterations", ""))
            if s["success"]:
                length = s.get("smoothed_path_length", s["path_length"])
                row.update(path_length=round(length, 3),
                           waypoints=s.get("smoothed_waypoints", s["waypoints"]))
                successes += 1
                times.append(s["search_time"])
                lengths.append(length)
                nodes.append(s["nodes"])
        except Exception as exc:                       # invalid start, blocked goal, ...
            row["error"] = f"{type(exc).__name__}: {exc}"

        raw_writer.writerow(row)
        raw_file.flush()

    sr_lo, sr_hi = wilson_ci(successes, runs)
    t_mean, t_lo, t_hi = _mean_and_ci(times)
    l_mean, l_lo, l_hi = _mean_and_ci(lengths)
    n_mean, n_lo, n_hi = _mean_and_ci(nodes)

    return dict(env=env_name, family=family, param=param, planner=planner_name, K=K,
                runs=runs, successes=successes,
                success_rate=round(successes / runs, 4),
                sr_lo=round(sr_lo, 4), sr_hi=round(sr_hi, 4),
                time_mean=round(t_mean, 4), time_lo=round(t_lo, 4), time_hi=round(t_hi, 4),
                len_mean=round(l_mean, 3), len_lo=round(l_lo, 3), len_hi=round(l_hi, 3),
                nodes_mean=round(n_mean, 1), nodes_lo=round(n_lo, 1), nodes_hi=round(n_hi, 1))


def print_header() -> None:
    print(f"\n{'environment':<22}{'planner':<14}{'success':>20}"
          f"{'search time (s)':>26}{'path length':>26}")
    print("-" * 108)


def print_row(r: dict) -> None:
    sr = f"{r['successes']}/{r['runs']} [{r['sr_lo']:.0%},{r['sr_hi']:.0%}]"
    if r["successes"]:
        t = f"{r['time_mean']:.2f} [{r['time_lo']:.2f},{r['time_hi']:.2f}]"
        l = f"{r['len_mean']:.1f} [{r['len_lo']:.1f},{r['len_hi']:.1f}]"
    else:
        t = l = "-"
    print(f"{r['env']:<22}{r['planner']:<14}{sr:>20}{t:>26}{l:>26}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiments over an environment set.")
    parser.add_argument("--envs", default="envs", help="directory of environment YAMLs")
    parser.add_argument("--pattern", default="*", help="filename filter, e.g. 'passage_*'")
    parser.add_argument("--drones", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30, help="seeds per (env, planner)")
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--planners", nargs="*", default=["rrt", "rrt-connect"],
                        choices=sorted(PLANNERS), help="which planners to run")
    parser.add_argument("--smooth", action="store_true",
                        help="enable shortcutting (off by default: it burns its whole "
                             "budget and inflates the timings)")
    parser.add_argument("--out", default="results", help="output file prefix")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.envs, f"{args.pattern}.yaml")) +
                   glob.glob(os.path.join(args.envs, f"{args.pattern}.yml")))
    if not paths:
        raise SystemExit(f"no environment files matching '{args.pattern}' in {args.envs}/")

    raw_path = f"{args.out}_raw.csv"
    summary_path = f"{args.out}_summary.csv"
    total_cells = len(paths) * len(args.planners)
    print(f"{len(paths)} environments x {len(args.planners)} planners x {args.runs} runs "
          f"= {total_cells * args.runs} planning attempts "
          f"(worst case {total_cells * args.runs * args.time_limit / 60:.0f} min)")

    started = time.perf_counter()
    summaries = []

    # Both files are written incrementally: a sweep of this length is likely to
    # be interrupted, and a summary only written at the end would be lost.
    with open(raw_path, "w", newline="", encoding="utf-8") as raw_file, \
            open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
        raw_writer.writeheader()
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        summary_writer.writeheader()
        print_header()

        for path in paths:
            family, param = describe(path)
            env_name = os.path.basename(path)
            try:
                sim = make_sim(args.drones, path)
            except Exception as exc:
                print(f"{env_name:<22}SKIPPED: {type(exc).__name__}: {exc}", flush=True)
                continue

            for planner_name in args.planners:
                summary = run_cell(sim, planner_name, PLANNERS[planner_name],
                                   args.runs, args.time_limit, args.smooth,
                                   env_name, family, param, args.drones,
                                   raw_writer, raw_file)
                summaries.append(summary)
                summary_writer.writerow(summary)
                summary_file.flush()
                print_row(summary)

    print(f"\nfinished in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"  per-run rows : {raw_path}")
    print(f"  summary table: {summary_path}")


if __name__ == "__main__":
    main()
