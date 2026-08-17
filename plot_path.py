"""
Matplotlib path viewer for the MultiDrone environment.

``MultiDrone.visualize_paths`` renders through VTK/vedo, which needs a display.
On a headless machine (Colab, a remote shell, CI) that fails with
``bad X server connection``. This module draws the same scene with matplotlib
and writes a PNG, so a path can be inspected anywhere.

    from plot_path import plot_path
    plot_path(path, "environment.yaml", out="path.png")

Or from the command line, which plans and plots in one go:

    python plot_path.py --env environment.yaml --drones 2 --out path.png

Only numpy / pyyaml / scipy / matplotlib are needed for plotting itself; fcl is
only required by the command-line mode, which runs the planner first.
"""

from __future__ import annotations

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Obstacle geometry
# ---------------------------------------------------------------------------

def _rotation(euler_deg) -> np.ndarray:
    if not euler_deg:
        return np.eye(3)
    return R.from_euler("xyz", euler_deg, degrees=True).as_matrix()


def _box_faces(position, size, rot) -> list[np.ndarray]:
    """The six quadrilateral faces of a rotated box."""
    half = np.asarray(size, dtype=float) / 2.0
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                       dtype=float) * half
    corners = corners @ rot.T + np.asarray(position, dtype=float)
    # Corner index bits are (x, y, z) with x the most significant.
    faces = [[0, 1, 3, 2], [4, 5, 7, 6],   # x = -/+
             [0, 1, 5, 4], [2, 3, 7, 6],   # y = -/+
             [0, 2, 6, 4], [1, 3, 7, 5]]   # z = -/+
    return [corners[f] for f in faces]


def _sphere_surface(centre, radius, n=20):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _cylinder_surface(p1, p2, radius, n=24):
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    axis = p2 - p1
    height = np.linalg.norm(axis)
    if height < 1e-9:
        return None
    unit = axis / height

    # Any vector not parallel to the axis gives us a perpendicular basis.
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(unit, helper)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(unit, b1)

    t = np.linspace(0, 2 * np.pi, n)
    ring = np.outer(np.cos(t), b1) + np.outer(np.sin(t), b2)
    bottom, top = p1 + radius * ring, p2 + radius * ring
    return (np.vstack([bottom[:, 0], top[:, 0]]),
            np.vstack([bottom[:, 1], top[:, 1]]),
            np.vstack([bottom[:, 2], top[:, 2]]))


def _draw_environment(ax, config) -> None:
    for obs in config.get("obstacles", []) or []:
        colour = obs.get("color", "gray")
        if obs["type"] == "box":
            faces = _box_faces(obs["position"], obs["size"], _rotation(obs.get("rotation")))
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=colour, edgecolors="k", linewidths=0.3, alpha=0.25))
        elif obs["type"] == "sphere":
            x, y, z = _sphere_surface(np.asarray(obs["position"], dtype=float),
                                      float(obs["radius"]))
            ax.plot_surface(x, y, z, color=colour, alpha=0.25, linewidth=0)
        elif obs["type"] == "cylinder":
            surface = _cylinder_surface(obs["endpoints"][0], obs["endpoints"][1],
                                        float(obs["radius"]))
            if surface is not None:
                ax.plot_surface(*surface, color=colour, alpha=0.25, linewidth=0)

    for goal in config.get("goals", []):
        x, y, z = _sphere_surface(np.asarray(goal["position"], dtype=float),
                                  float(goal["radius"]))
        ax.plot_surface(x, y, z, color=goal.get("color", "gold"), alpha=0.35, linewidth=0)


# ---------------------------------------------------------------------------
# Path plotting
# ---------------------------------------------------------------------------

def plot_path(
    path: list[np.ndarray],
    environment_file: str,
    out: str | None = "path.png",
    show: bool = False,
    title: str | None = None,
    elev: float = 22.0,
    azim: float = -60.0,
):
    """Draw the environment and one coloured trajectory per drone.

    Args:
        path: list of (K, 3) configurations, as returned by the planner.
        environment_file: the YAML the simulator was built from.
        out: PNG to write, or None to skip saving.
        show: also open an interactive window (needs a display).
        title: figure title; a default is generated when omitted.

    Returns:
        (fig, ax)
    """
    assert path is not None and len(path) >= 2, "Path must contain at least 2 configurations."
    waypoints = np.asarray([np.asarray(q, dtype=float) for q in path])  # (T, K, 3)
    T, K, _ = waypoints.shape

    with open(environment_file, "r") as f:
        config = yaml.safe_load(f)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_environment(ax, config)

    cmap = colormaps["jet"]
    for i in range(K):
        trajectory = waypoints[:, i, :]
        colour = cmap(i / max(K - 1, 1))
        ax.plot(*trajectory.T, color=colour, linewidth=2, label=f"drone {i}")
        ax.scatter(*trajectory[1:-1].T, color=colour, s=12, depthshade=False)
        ax.scatter(*trajectory[0], color=colour, s=90, marker="o",
                   edgecolors="k", linewidths=0.8, depthshade=False)
        ax.scatter(*trajectory[-1], color=colour, s=150, marker="*",
                   edgecolors="k", linewidths=0.8, depthshade=False)

    bounds = config["bounds"]
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])
    ax.set_box_aspect([bounds["x"][1] - bounds["x"][0],
                       bounds["y"][1] - bounds["y"][0],
                       bounds["z"][1] - bounds["z"][0]])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)

    length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0).reshape(T - 1, -1), axis=1)))
    ax.set_title(title or f"K={K}, {T} waypoints, path length {length:.1f}"
                          "   (o = start, * = goal)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    if show:
        plt.show()
    return fig, ax


def plot_drone_separation(path: list[np.ndarray], drone_radius: float = 0.3,
                          out: str | None = "separation.png", show: bool = False):
    """Plot the closest distance between any two drones along the path.

    A useful sanity check: the curve must stay above 2 * drone_radius, otherwise
    the drones intersect somewhere on the trajectory.
    """
    waypoints = np.asarray([np.asarray(q, dtype=float) for q in path])
    T, K, _ = waypoints.shape
    if K < 2:
        print("only one drone - nothing to separate")
        return None

    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    closest = [np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)[mask].min()
               for q in waypoints]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(closest, marker="o", linewidth=1.5, label="closest pair")
    ax.axhline(2 * drone_radius, color="red", linestyle="--",
               label=f"collision at {2 * drone_radius:g}")
    ax.set_xlabel("waypoint")
    ax.set_ylabel("distance")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title("drone-drone separation along the path")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    if show:
        plt.show()
    return fig, ax


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plan and plot a path with matplotlib.")
    parser.add_argument("--env", default="environment.yaml")
    parser.add_argument("--drones", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="path.png")
    parser.add_argument("--separation-out", default="separation.png")
    parser.add_argument("--show", action="store_true", help="open a window as well")
    args = parser.parse_args()

    from multi_drone import MultiDrone
    from rrt_planner import rrt_plan

    sim = MultiDrone(num_drones=args.drones, environment_file=args.env)
    path, stats = rrt_plan(sim, time_limit=args.time_limit, seed=args.seed)
    if path is None:
        print(f"no path found in {stats['search_time']:.2f}s "
              f"({stats['nodes']} nodes) - nothing to plot")
        return

    print(f"solved in {stats['search_time']:.2f}s, {len(path)} waypoints, "
          f"length {stats.get('smoothed_path_length', stats['path_length']):.2f}")
    plot_path(path, args.env, out=args.out, show=args.show)
    plot_drone_separation(path, drone_radius=getattr(sim, "_drone_radius", 0.3),
                          out=args.separation_out, show=args.show)


if __name__ == "__main__":
    main()
