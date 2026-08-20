"""
Matplotlib path viewer for the MultiDrone environment.

``MultiDrone.visualize_paths`` renders through VTK/vedo, which needs a display.
On a headless machine (Colab, a remote shell, CI) that fails with
``bad X server connection``. This module draws the same scene with matplotlib
and writes a PNG, so a path can be inspected anywhere.

    from plot_path import plot_path
    plot_path(path, "environment.yaml", out="path.png")

Or from the command line, which plans and plots in one go:

    python plot_path.py --env env_0.yaml --drones 5
        -> figures/path_5_env_0.png and figures/separation_5_env_0.png

Only numpy / pyyaml / scipy / matplotlib are needed for plotting itself; fcl is
only required by the command-line mode, which runs the planner first.
"""

from __future__ import annotations

import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Obstacle geometry
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create the parent directory of `path` if needed.

    Note os.makedirs(..., exist_ok=True) still raises when the target already
    exists as a *file* - exist_ok only tolerates an existing directory. That
    happens when an output filename is passed where a directory was expected,
    so translate it into a message that says what to do.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not parent or os.path.isdir(parent):
        return
    if os.path.exists(parent):
        raise NotADirectoryError(
            f"cannot write {path!r}: {parent!r} already exists as a file, not a "
            f"directory. If you meant to name the output file, pass it with "
            f"--out instead of --out-dir.")
    os.makedirs(parent, exist_ok=True)


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
        _ensure_dir(out)
        fig.savefig(out, dpi=130)
        print(f"wrote {os.path.abspath(out)}")
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
        _ensure_dir(out)
        fig.savefig(out, dpi=130)
        print(f"wrote {os.path.abspath(out)}")
    if show:
        plt.show()
    return fig, ax


def infer_drones(environment_file: str) -> int:
    """Number of drones an environment expects, from its initial_configuration."""
    with open(environment_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return len(config["initial_configuration"])


def plot_one(environment_file: str, drones: int | None, planner, time_limit: float,
             seed: int, out_dir: str, out: str | None, separation_out: str | None,
             show: bool) -> bool:
    """Plan and plot one environment. Returns True if a path was found."""
    from multi_drone import MultiDrone

    k = drones if drones is not None else infer_drones(environment_file)
    env_stem = os.path.splitext(os.path.basename(environment_file))[0]

    sim = MultiDrone(num_drones=k, environment_file=environment_file)
    path, stats = planner(sim, time_limit=time_limit, seed=seed)
    if path is None:
        print(f"  {env_stem}: no path in {stats['search_time']:.2f}s "
              f"({stats['nodes']} nodes) - nothing to plot")
        return False

    print(f"  {env_stem}: solved in {stats['search_time']:.2f}s, "
          f"{len(path)} waypoints, "
          f"length {stats.get('smoothed_path_length', stats['path_length']):.2f}")

    def resolve(name, default):
        """Put bare filenames inside out_dir; leave explicit paths alone."""
        name = name or default
        if os.path.isabs(name) or os.path.dirname(name):
            return name
        return os.path.join(out_dir, name)

    fig, _ = plot_path(path, environment_file, show=show,
                       out=resolve(out, f"path_{k}_{env_stem}.png"))
    fig2, _ = plot_drone_separation(path, drone_radius=getattr(sim, "_drone_radius", 0.3),
                                    show=show,
                                    out=resolve(separation_out,
                                                f"separation_{k}_{env_stem}.png"))
    if not show:                      # batches would otherwise pile up open figures
        plt.close(fig)
        plt.close(fig2)
    return True


def main() -> None:
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Plan and plot paths with matplotlib, for one environment "
                    "or a whole folder.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--env", default=None, help="a single environment YAML")
    source.add_argument("--env-folder", default=None,
                        help="process every environment in this folder")
    parser.add_argument("--pattern", default="*",
                        help="filename filter for --env-folder, e.g. 'passage_*'")
    parser.add_argument("--drones", type=int, default=None,
                        help="number of drones; inferred from each environment "
                             "file when omitted")
    parser.add_argument("--planner", default="rrt-connect",
                        choices=["rrt", "rrt-connect"])
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="figures",
                        help="directory for the PNGs (created if missing)")
    parser.add_argument("--out", default=None,
                        help="path figure filename; defaults to "
                             "path_<drones>_<env>.png. Single-environment mode only")
    parser.add_argument("--separation-out", default=None,
                        help="separation figure filename; defaults to "
                             "separation_<drones>_<env>.png")
    parser.add_argument("--show", action="store_true", help="open a window as well")
    args = parser.parse_args()

    if os.path.isfile(args.out_dir):
        raise SystemExit(
            f"--out-dir {args.out_dir!r} is an existing file. It names a "
            f"directory for the figures; use --out to name the file.")

    if args.env_folder:
        environments = sorted(glob.glob(os.path.join(args.env_folder, f"{args.pattern}.yaml")) +
                              glob.glob(os.path.join(args.env_folder, f"{args.pattern}.yml")))
        if not environments:
            raise SystemExit(f"no environments matching '{args.pattern}' "
                             f"in {args.env_folder}/")
        if args.out or args.separation_out:
            raise SystemExit("--out/--separation-out name a single file; they "
                             "cannot be combined with --env-folder")
    else:
        environments = [args.env or "environment.yaml"]

    from rrt_planner import rrt_plan
    from rrt_connect import rrt_connect_plan
    planner = {"rrt": rrt_plan, "rrt-connect": rrt_connect_plan}[args.planner]

    print(f"{len(environments)} environment(s), planner={args.planner}, "
          f"{args.time_limit:g}s each")
    solved, failed = [], []
    for environment_file in environments:
        try:
            ok = plot_one(environment_file, args.drones, planner, args.time_limit,
                          args.seed, args.out_dir, args.out, args.separation_out,
                          args.show)
        except Exception as exc:      # a bad environment must not kill the batch
            print(f"  {os.path.basename(environment_file)}: "
                  f"{type(exc).__name__}: {exc}")
            ok = False
        (solved if ok else failed).append(os.path.basename(environment_file))

    print(f"\nplotted {len(solved)}/{len(environments)} into "
          f"{os.path.abspath(args.out_dir)}")
    if failed:
        print("no figure for: " + ", ".join(failed))


if __name__ == "__main__":
    main()
