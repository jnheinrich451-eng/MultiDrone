"""
Environment generator for the complexity study (Part B, question 4).

Produces two *independently controlled* families of environments. Everything
except the named parameter is held constant across a family - same bounds, same
initial configuration, same goals - so any change in planner performance is
attributable to that parameter alone.

  passage family   wall across the middle of the workspace with a W x W opening,
                   W shrinking. This is the theoretically interesting axis: the
                   probability of sampling into a corridor of width W scales as
                   W^3 per drone, so free space stops being "expansive" and
                   sampling-based planners degrade sharply.

  clutter family   N randomly placed obstacles, no wall, N increasing. This is
                   the obvious reading of "more complex", and the point of
                   generating it is to show it is the *cheap* axis - it adds
                   detours, which planners handle well, rather than bottlenecks.

Usage:

    python make_envs.py --out envs --drones 2                    # K=2 family for the Q5 comparison
    python make_envs.py --out envs_small --workspace 50          # scales all geometry proportionally
    python make_envs.py --out envs --openings 60 40 20 --counts 0 10 20
    python make_envs.py --out envs2 --seed 7                     # different random clutter layout


The generator is standalone: it validates the environments it writes using its
own analytic clearance tests, so it needs neither fcl nor the simulator.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

DRONE_RADIUS = 0.3          # must match MultiDrone._drone_radius
GOAL_RADIUS = 2.0


# ---------------------------------------------------------------------------
# Analytic clearance tests (used only to validate what we generate)
# ---------------------------------------------------------------------------

def _rotation(euler_deg) -> np.ndarray:
    if not euler_deg or not any(euler_deg):
        return np.eye(3)
    from scipy.spatial.transform import Rotation as R
    return R.from_euler("xyz", euler_deg, degrees=True).as_matrix()


def clearance(point: np.ndarray, obstacle: dict) -> float:
    """Distance from `point` to the obstacle surface. 0.0 if inside."""
    p = np.asarray(point, dtype=float)

    if obstacle["type"] == "sphere":
        centre = np.asarray(obstacle["position"], dtype=float)
        return max(float(np.linalg.norm(p - centre)) - float(obstacle["radius"]), 0.0)

    if obstacle["type"] == "box":
        centre = np.asarray(obstacle["position"], dtype=float)
        half = np.asarray(obstacle["size"], dtype=float) / 2.0
        local = _rotation(obstacle.get("rotation")).T @ (p - centre)
        return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))

    if obstacle["type"] == "cylinder":
        a, b = (np.asarray(e, dtype=float) for e in obstacle["endpoints"])
        axis = b - a
        height = float(np.linalg.norm(axis))
        unit = axis / height
        h = float((p - a) @ unit)
        radial = float(np.linalg.norm((p - a) - h * unit))
        d_axial = max(-h, h - height, 0.0)
        d_radial = max(radial - float(obstacle["radius"]), 0.0)
        return float(np.hypot(d_axial, d_radial))

    raise ValueError(f"unknown obstacle type: {obstacle['type']}")


def min_clearance(obstacle: dict, points: np.ndarray) -> float:
    return min(clearance(p, obstacle) for p in points)


# ---------------------------------------------------------------------------
# Fixed elements: start and goal configurations
# ---------------------------------------------------------------------------

def make_starts(k: int, workspace: float) -> list[list[float]]:
    """K start positions in a grid near the origin corner, spaced well apart."""
    spacing = max(4.0, workspace * 0.04)
    base = workspace * 0.05
    return [[base + spacing * (i % 3), base + spacing * (i // 3), base]
            for i in range(k)]


def make_goals(k: int, workspace: float) -> list[dict]:
    """K goals in the far corner, at staggered heights so the team cannot
    simply fly in formation."""
    spacing = max(8.0, workspace * 0.08)
    base = workspace * 0.90
    heights = [workspace * h for h in (0.10, 0.12, 0.12, 0.28, 0.40)]
    goals = []
    for i in range(k):
        goals.append({
            "position": [round(base - spacing * (i % 3), 2),
                         round(base - spacing * (i // 3), 2),
                         round(heights[i % len(heights)], 2)],
            "radius": GOAL_RADIUS,
            "color": "yellow",
        })
    return goals


# ---------------------------------------------------------------------------
# Obstacle construction
# ---------------------------------------------------------------------------

def wall_with_opening(opening: float, workspace: float,
                      thickness: float = 4.0) -> list[dict]:
    """Four boxes forming a wall across y = workspace/2 with a square opening
    of side `opening` at the centre."""
    mid = workspace / 2.0
    side = (workspace - opening) / 2.0          # width of each flanking slab
    box = lambda pos, size: dict(type="box", position=[round(v, 2) for v in pos],
                                 size=[round(v, 2) for v in size],
                                 rotation=[0, 0, 0], color="red")
    return [
        box([side / 2, mid, mid], [side, thickness, workspace]),               # left
        box([workspace - side / 2, mid, mid], [side, thickness, workspace]),   # right
        box([mid, mid, side / 2], [opening, thickness, side]),                 # below
        box([mid, mid, workspace - side / 2], [opening, thickness, side]),     # above
    ]


def fixed_clutter(workspace: float) -> list[dict]:
    """Three obstacles, one of each geometry type, scaled to the workspace."""
    s = workspace / 100.0
    return [
        dict(type="cylinder", endpoints=[[72 * s, 72 * s, 0], [72 * s, 72 * s, 60 * s]],
             radius=round(6 * s, 2), rotation=[0, 0, 0], color="darkred"),
        dict(type="sphere", position=[round(30 * s, 2), round(72 * s, 2), round(35 * s, 2)],
             radius=round(10 * s, 2), color="darkred"),
        dict(type="box", position=[round(70 * s, 2), round(22 * s, 2), round(30 * s, 2)],
             size=[round(22 * s, 2)] * 3, rotation=[0, 0, 30], color="darkred"),
    ]


def random_clutter(n: int, workspace: float, keep_clear: np.ndarray,
                   rng: np.random.Generator, margins: np.ndarray) -> list[dict]:
    """`n` random obstacles that leave every start and goal free.

    `keep_clear` are the points to avoid and `margins` the required clearance
    at each - larger for goals, since a drone may sit anywhere in its goal ball.
    """
    obstacles, attempts = [], 0
    while len(obstacles) < n and attempts < 200 * n:
        attempts += 1
        kind = rng.choice(["box", "sphere", "cylinder"])
        centre = rng.uniform(workspace * 0.12, workspace * 0.88, size=3)

        if kind == "sphere":
            obs = dict(type="sphere", position=[round(v, 2) for v in centre],
                       radius=round(float(rng.uniform(0.04, 0.09) * workspace), 2),
                       color="red")
        elif kind == "box":
            size = rng.uniform(0.08, 0.20, size=3) * workspace
            obs = dict(type="box", position=[round(v, 2) for v in centre],
                       size=[round(v, 2) for v in size],
                       rotation=[0, 0, int(rng.integers(0, 90))], color="red")
        else:
            half = float(rng.uniform(0.10, 0.30)) * workspace
            z0 = max(0.0, centre[2] - half)
            z1 = min(workspace, centre[2] + half)
            obs = dict(type="cylinder",
                       endpoints=[[round(centre[0], 2), round(centre[1], 2), round(z0, 2)],
                                  [round(centre[0], 2), round(centre[1], 2), round(z1, 2)]],
                       radius=round(float(rng.uniform(0.03, 0.07) * workspace), 2),
                       rotation=[0, 0, 0], color="red")

        if all(clearance(p, obs) > m for p, m in zip(keep_clear, margins)):
            obstacles.append(obs)

    if len(obstacles) < n:
        raise RuntimeError(f"could only place {len(obstacles)} of {n} obstacles; "
                           f"the workspace is too crowded")
    return obstacles


# ---------------------------------------------------------------------------
# Writing and validation
# ---------------------------------------------------------------------------

def build_env(obstacles: list[dict], starts, goals, workspace: float,
              header: str) -> tuple[dict, str]:
    config = {
        "bounds": {"x": [0, workspace], "y": [0, workspace], "z": [0, workspace]},
        "initial_configuration": starts,
        "obstacles": obstacles,
        "goals": goals,
    }
    validate(config)
    return config, header


def validate(config: dict) -> None:
    """Check that no start or goal is blocked, and that drones fit apart."""
    starts = np.asarray(config["initial_configuration"], dtype=float)
    goal_centres = np.asarray([g["position"] for g in config["goals"]], dtype=float)

    for name, points, needed in (("start", starts, DRONE_RADIUS),
                                 ("goal", goal_centres, DRONE_RADIUS)):
        for i, p in enumerate(points):
            for obs in config["obstacles"]:
                if clearance(p, obs) <= needed:
                    raise AssertionError(
                        f"{name} {i} at {p.tolist()} is inside a {obs['type']} obstacle")

    for label, points in (("start", starts), ("goal", goal_centres)):
        for i in range(len(points)):
            for j in range(i):
                if np.linalg.norm(points[i] - points[j]) <= 2 * DRONE_RADIUS:
                    raise AssertionError(f"{label}s {i} and {j} are closer than two drone radii")


def _plain(obj):
    """Convert numpy scalars/strings to built-ins so yaml.safe_dump accepts them."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    return obj


def write_env(path: str, config: dict, header: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n\n")
        yaml.safe_dump(_plain(config), f, sort_keys=False, default_flow_style=None)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate complexity-sweep environments.")
    parser.add_argument("--out", default="envs", help="output directory")
    parser.add_argument("--drones", type=int, default=5, help="number of drones K")
    parser.add_argument("--workspace", type=float, default=100.0, help="cube side length")
    parser.add_argument("--openings", type=float, nargs="*",
                        default=[60, 50, 40, 30, 20, 15, 10],
                        help="opening widths for the passage family")
    parser.add_argument("--counts", type=int, nargs="*",
                        default=[0, 5, 10, 20, 30, 40],
                        help="obstacle counts for the clutter family")
    parser.add_argument("--seed", type=int, default=0, help="seed for clutter placement")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    W = args.workspace
    starts = make_starts(args.drones, W)
    goals = make_goals(args.drones, W)

    # Goals need more clearance than starts: a drone may sit anywhere inside
    # its goal ball, not just at the centre.
    keep_clear = np.asarray(starts + [g["position"] for g in goals], dtype=float)
    margins = np.asarray([DRONE_RADIUS + 1.0] * len(starts)
                         + [GOAL_RADIUS + DRONE_RADIUS + 1.0] * len(goals))

    written = []

    for opening in args.openings:
        name = f"passage_w{int(opening):03d}.yaml"
        header = (f"# Passage family, opening = {opening:g} of {W:g}.\n"
                  f"# A wall across y = {W/2:g} with a {opening:g} x {opening:g} opening,\n"
                  f"# plus three fixed clutter obstacles. Only the opening width varies\n"
                  f"# across this family; bounds, starts and goals are identical.\n"
                  f"# K = {args.drones}.")
        config, header = build_env(wall_with_opening(opening, W) + fixed_clutter(W),
                                   starts, goals, W, header)
        write_env(os.path.join(args.out, name), config, header)
        written.append((name, len(config["obstacles"])))

    rng = np.random.default_rng(args.seed)
    for count in args.counts:
        name = f"clutter_n{count:03d}.yaml"
        header = (f"# Clutter family, {count} randomly placed obstacles, no wall.\n"
                  f"# Only the obstacle count varies across this family; bounds,\n"
                  f"# starts and goals are identical. Seed = {args.seed}, K = {args.drones}.")
        config, header = build_env(random_clutter(count, W, keep_clear, rng, margins),
                                   starts, goals, W, header)
        write_env(os.path.join(args.out, name), config, header)
        written.append((name, len(config["obstacles"])))

    print(f"wrote {len(written)} environments to {args.out}/ "
          f"(K={args.drones}, workspace={W:g})\n")
    for name, n_obs in written:
        print(f"  {name:24s} {n_obs:3d} obstacles")
    print("\nAll environments validated: no start or goal is blocked.")


if __name__ == "__main__":
    main()
