"""
Centralised RRT motion planner for the MultiDrone simulator.

C-space
-------
For K drones, each modelled as a translating bounding sphere in a 3D workspace
W = [x_lo, x_hi] x [y_lo, y_hi] x [z_lo, z_hi], the centralised configuration
space is the Cartesian product of the K individual workspaces:

    C = W^K  subset of  R^(3K),   q = (p_1, ..., p_K),  p_i in R^3

so dim(C) = 3K (K in [1, 5]  =>  3..15 dimensions). The free space C_free is the
subset of C in which (a) every drone is inside the bounds, (b) no drone
intersects an obstacle, and (c) no two drones intersect each other. Conditions
(a)-(c) are exactly what ``MultiDrone.is_valid`` tests, so this module never
re-implements geometry: all validity queries go through the simulator.

The metric on C is the Euclidean metric of R^(3K), i.e.
    d(q, q') = sqrt( sum_i ||p_i - p_i'||^2 ).

Representation note
-------------------
Internally a configuration is a flat vector of shape (3K,) -- convenient for
nearest-neighbour and steering arithmetic. The simulator expects shape (K, 3),
so ``_to_cfg`` reshapes (and casts to float32) at every API boundary.

Usage
-----
    from multi_drone import MultiDrone
    from rrt_planner import rrt_plan

    sim = MultiDrone(num_drones=2, environment_file="environment.yaml")
    path, stats = rrt_plan(sim, time_limit=20.0, seed=0)
    if path is not None:
        sim.visualize_paths(path)

Run ``python rrt_planner.py --help`` for the command-line interface, which also
provides a batch mode reporting mean and 95% confidence intervals.
"""

from __future__ import annotations

import time
import numpy as np


# ---------------------------------------------------------------------------
# Environment accessors
#
# MultiDrone exposes public getters only for the initial configuration and the
# goal positions. Bounds, goal radii and the drone radius are stored as private
# attributes; we read them through these helpers (with sensible fallbacks) so
# that multi_drone.py can stay untouched.
# ---------------------------------------------------------------------------

def get_bounds(sim) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) workspace bounds, each of shape (3,)."""
    bounds = np.asarray(sim._bounds, dtype=float)  # shape (3, 2)
    return bounds[:, 0].copy(), bounds[:, 1].copy()


def get_goal_radii(sim, num_drones: int) -> np.ndarray:
    """Return the goal-sphere radii, shape (K,). Falls back to zeros."""
    radii = getattr(sim, "_goal_radii", None)
    if radii is None:
        return np.zeros(num_drones, dtype=float)
    return np.asarray(radii, dtype=float).reshape(-1)


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class Tree:
    """Flat-array RRT tree with vectorised nearest-neighbour search.

    Configurations are kept in one contiguous (capacity, 3K) array so that a
    nearest-neighbour query is a single vectorised norm over all nodes, rather
    than a Python loop. Parents are stored as integer indices.
    """

    def __init__(self, root: np.ndarray, capacity: int = 1024):
        self.dim = root.size
        self._q = np.empty((capacity, self.dim), dtype=float)
        self._parent = np.empty(capacity, dtype=np.int64)
        self._q[0] = root
        self._parent[0] = -1
        self.size = 1

    def add(self, q: np.ndarray, parent: int) -> int:
        """Append a node and return its index."""
        if self.size == self._q.shape[0]:
            self._q = np.resize(self._q, (2 * self.size, self.dim))
            self._parent = np.resize(self._parent, 2 * self.size)
        self._q[self.size] = q
        self._parent[self.size] = parent
        self.size += 1
        return self.size - 1

    def nearest(self, q: np.ndarray) -> tuple[int, float]:
        """Return (index, distance) of the node closest to ``q`` under the
        Euclidean metric of R^(3K)."""
        diff = self._q[:self.size] - q
        dists = np.linalg.norm(diff, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def config(self, idx: int) -> np.ndarray:
        return self._q[idx].copy()

    def reconstruct_path(self, idx: int) -> list[np.ndarray]:
        """Walk parent pointers from ``idx`` to the root and return the path
        root -> idx as a list of flat (3K,) configurations."""
        path = []
        while idx != -1:
            path.append(self._q[idx].copy())
            idx = int(self._parent[idx])
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class RRTPlanner:
    """Single-tree RRT with goal biasing for centralised multi-drone planning.

    Pseudo-code (one iteration):

        q_rand <- SampleGoal()          with probability goal_bias
                  SamplePartialGoal()   with probability partial_goal_bias
                  SampleUniform()       otherwise
        idx    <- Nearest(T, q_rand)
        repeat                                  # once, unless greedily connecting
            q_new <- Steer(q[idx], q_rand, step_size)
            if not IsValid(q_new) or not MotionValid(q[idx], q_new): break
            idx <- T.add(q_new, parent=idx)
            if IsGoal(q_new): return Path(idx)
        until q_new = q_rand or not greedy
        if d(q_new, q_goal) <= connect_radius and
           IsValid(q_goal) and MotionValid(q_new, q_goal):
            return Path(T.add(q_goal, parent=idx))

    ``greedy_goal_connect`` applies the RRT-Connect extension heuristic to
    goal-biased samples only: a single goal sample is followed as far as the
    obstacles allow instead of yielding one ``step_size`` extension. Uniform
    samples keep the plain single-step extension, which preserves the Voronoi
    bias that makes RRT explore.

    ``partial_goal_bias`` draws samples in which a random subset of the drones
    sits at its goal while the rest are uniform. In a composite C-space the
    probability of sampling near the full goal configuration decays with K, so
    without this the tree must bring all K drones home in one correlated push;
    partial goal samples let the drones arrive a few at a time.
    """

    def __init__(
        self,
        sim,
        step_size: float | None = None,
        goal_bias: float = 0.1,
        partial_goal_bias: float = 0.25,
        goal_connect_factor: float = 2.0,
        greedy_goal_connect: bool = True,
        seed: int | None = None,
    ):
        self.sim = sim
        self.K = sim.N
        self.dim = 3 * self.K
        self.rng = np.random.default_rng(seed)

        self.lower, self.upper = get_bounds(sim)
        # is_valid() bounds-checks the drone *centres*, so the sampling region is
        # exactly the bounding box -- no inset by the drone radius is needed.
        extent = float(np.min(self.upper - self.lower))
        self.step_size = float(step_size) if step_size is not None else 0.1 * extent

        self.goal_bias = float(goal_bias)
        self.partial_goal_bias = float(partial_goal_bias)
        self.connect_radius = goal_connect_factor * self.step_size
        self.greedy_goal_connect = bool(greedy_goal_connect)

        self.q_start = np.asarray(sim.initial_configuration, dtype=float).reshape(-1)
        self.q_goal = np.asarray(sim.goal_positions, dtype=float).reshape(-1)
        self.goal_radii = get_goal_radii(sim, self.K)

        # The configuration made of the goal *centres* may itself be invalid
        # (e.g. two goal spheres closer than two drone radii). If so, goal
        # samples are drawn from inside the goal spheres instead.
        self._goal_centre_valid = self.is_valid(self.q_goal)

        self.stats: dict = {}

    # -- simulator boundary --------------------------------------------------

    def _to_cfg(self, q: np.ndarray) -> np.ndarray:
        """Flat (3K,) -> simulator configuration (K, 3), float32."""
        return np.asarray(q, dtype=np.float32).reshape(self.K, 3)

    def is_valid(self, q: np.ndarray) -> bool:
        return bool(self.sim.is_valid(self._to_cfg(q)))

    def motion_valid(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        return bool(self.sim.motion_valid(self._to_cfg(q_from), self._to_cfg(q_to)))

    def is_goal(self, q: np.ndarray) -> bool:
        return bool(self.sim.is_goal(self._to_cfg(q)))

    # -- sampling ------------------------------------------------------------

    def sample_uniform(self) -> np.ndarray:
        """Uniform sample over the bounding box of C = W^K."""
        return self.rng.uniform(self.lower, self.upper, size=(self.K, 3)).reshape(-1)

    def sample_goal(self) -> np.ndarray:
        """A configuration in the goal region: the goal centres if that
        configuration is valid, otherwise a uniform sample inside each drone's
        goal sphere."""
        if self._goal_centre_valid:
            return self.q_goal.copy()

        centres = self.q_goal.reshape(self.K, 3)
        # Uniform in the ball: isotropic direction, radius scaled by u^(1/3).
        directions = self.rng.normal(size=(self.K, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = self.goal_radii * self.rng.random(self.K) ** (1.0 / 3.0)
        return (centres + radii[:, None] * directions).reshape(-1)

    def sample_partial_goal(self) -> np.ndarray:
        """A uniform sample in which each drone is independently replaced by its
        own goal position with probability 1/2."""
        q = self.sample_uniform().reshape(self.K, 3)
        g = self.sample_goal().reshape(self.K, 3)
        at_goal = self.rng.random(self.K) < 0.5
        q[at_goal] = g[at_goal]
        return q.reshape(-1)

    # -- steering ------------------------------------------------------------

    def steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """Move from ``q_near`` towards ``q_rand`` by at most ``step_size``.

        All K drones translate simultaneously along the straight line in C,
        which is what makes this a *centralised* planner."""
        direction = q_rand - q_near
        distance = float(np.linalg.norm(direction))
        if distance <= self.step_size or distance < 1e-12:
            return q_rand.copy()
        return q_near + (self.step_size / distance) * direction

    # -- main loop -----------------------------------------------------------

    def plan(self, time_limit: float = 20.0) -> list[np.ndarray] | None:
        """Grow the tree until the goal region is reached or time runs out.

        Returns the path as a list of (K, 3) float32 configurations (suitable
        for ``MultiDrone.visualize_paths``), or None on failure. Run statistics
        are left in ``self.stats``.
        """
        t0 = time.perf_counter()
        self.stats = {
            "success": False,
            "search_time": 0.0,
            "iterations": 0,
            "nodes": 1,
            "rejected_configs": 0,
            "rejected_motions": 0,
            "path_length": float("nan"),
            "waypoints": 0,
        }

        if not self.is_valid(self.q_start):
            self.stats["search_time"] = time.perf_counter() - t0
            raise ValueError("The initial configuration is in collision.")

        tree = Tree(self.q_start)

        if self.is_goal(self.q_start):
            return self._finish(tree, 0, t0)

        while time.perf_counter() - t0 < time_limit:
            self.stats["iterations"] += 1

            u = self.rng.random()
            if u < self.goal_bias:
                q_rand, towards_goal = self.sample_goal(), True
            elif u < self.goal_bias + self.partial_goal_bias:
                q_rand, towards_goal = self.sample_partial_goal(), True
            else:
                q_rand, towards_goal = self.sample_uniform(), False
            greedy = towards_goal and self.greedy_goal_connect

            near_idx, _ = tree.nearest(q_rand)
            q_near = tree.config(near_idx)
            q_new = None

            # Extend once, or repeatedly while greedily following a goal sample.
            while True:
                q_step = self.steer(q_near, q_rand)

                if not self.is_valid(q_step):
                    self.stats["rejected_configs"] += 1
                    break
                if not self.motion_valid(q_near, q_step):
                    self.stats["rejected_motions"] += 1
                    break

                near_idx = tree.add(q_step, near_idx)
                q_new = q_step

                # The goal is a region, so a new node may already satisfy it.
                if self.is_goal(q_new):
                    return self._finish(tree, near_idx, t0)

                reached_sample = np.linalg.norm(q_rand - q_new) < 1e-9
                if not greedy or reached_sample:
                    break
                if time.perf_counter() - t0 >= time_limit:
                    break
                q_near = q_new

            if q_new is None:
                continue

            # Try to connect the frontier directly to the goal configuration.
            q_goal = self.sample_goal()
            if np.linalg.norm(q_goal - q_new) <= self.connect_radius:
                if self.is_valid(q_goal) and self.motion_valid(q_new, q_goal):
                    goal_idx = tree.add(q_goal, near_idx)
                    return self._finish(tree, goal_idx, t0)

        self.stats["search_time"] = time.perf_counter() - t0
        self.stats["nodes"] = tree.size
        return None

    def _finish(self, tree: Tree, goal_idx: int, t0: float) -> list[np.ndarray]:
        flat_path = tree.reconstruct_path(goal_idx)
        if len(flat_path) == 1:  # start already in the goal region
            flat_path.append(flat_path[0].copy())
        self.stats.update(
            success=True,
            search_time=time.perf_counter() - t0,
            nodes=tree.size,
            waypoints=len(flat_path),
            path_length=path_length(flat_path),
        )
        return [self._to_cfg(q) for q in flat_path]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def path_length(path: list[np.ndarray]) -> float:
    """Length of a path under the Euclidean metric of R^(3K)."""
    flat = [np.asarray(q, dtype=float).reshape(-1) for q in path]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(flat, flat[1:])))


def shortcut_path(
    sim,
    path: list[np.ndarray],
    time_limit: float = 1.0,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Random shortcutting: repeatedly try to replace a sub-path between two
    waypoints by the straight line joining them.

    RRT paths are jagged because every edge is a ``step_size`` extension towards
    a random sample; this removes most of that without changing homotopy class.
    """
    if path is None or len(path) < 3:
        return path

    rng = np.random.default_rng(seed)
    K = sim.N
    current = [np.asarray(q, dtype=np.float32).reshape(K, 3) for q in path]
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < time_limit and len(current) > 2:
        i, j = sorted(rng.choice(len(current), size=2, replace=False))
        if j - i < 2:
            continue
        if sim.motion_valid(current[i], current[j]):
            current = current[:i + 1] + current[j:]

    return current


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def rrt_plan(
    sim,
    time_limit: float = 20.0,
    step_size: float | None = None,
    goal_bias: float = 0.1,
    partial_goal_bias: float = 0.25,
    greedy_goal_connect: bool = True,
    seed: int | None = None,
    smooth: bool = True,
    smooth_fraction: float = 0.1,
) -> tuple[list[np.ndarray] | None, dict]:
    """Plan a centralised path for all drones in ``sim``.

    The whole call (search + optional smoothing) stays inside ``time_limit``.

    Returns:
        (path, stats). ``path`` is a list of (K, 3) configurations from the
        initial configuration to a configuration in the goal region, or None if
        no path was found within the budget. ``stats`` separates ``search_time``
        (the quantity to report when comparing environments) from
        ``smoothing_time`` and ``total_time``.
    """
    t_start = time.perf_counter()
    smooth_budget = time_limit * smooth_fraction if smooth else 0.0
    planner = RRTPlanner(
        sim,
        step_size=step_size,
        goal_bias=goal_bias,
        partial_goal_bias=partial_goal_bias,
        greedy_goal_connect=greedy_goal_connect,
        seed=seed,
    )
    path = planner.plan(time_limit=time_limit - smooth_budget)
    stats = dict(planner.stats)

    stats["smoothing_time"] = 0.0
    if path is not None and smooth and smooth_budget > 0.0:
        path = shortcut_path(sim, path, time_limit=smooth_budget, seed=seed)
        stats["smoothing_time"] = time.perf_counter() - t_start - stats["search_time"]
        stats["smoothed_waypoints"] = len(path)
        stats["smoothed_path_length"] = path_length(path)

    stats["total_time"] = time.perf_counter() - t_start
    return path, stats


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _mean_and_ci(values: list[float], floor: float | None = 0.0
                 ) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for a 95% confidence interval of the mean.

    With fewer than two samples the bounds are NaN rather than equal to the
    mean: one observation tells you nothing about the spread, and reporting a
    zero-width interval would claim certainty that does not exist.

    ``floor`` clips the lower bound, defaulting to 0 because every quantity
    measured here (time, path length, node counts) is non-negative. At small n
    the t interval is wide enough to run past zero, which is arithmetically
    correct but not a meaningful claim about a duration.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, float("nan"), float("nan")
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    try:
        from scipy.stats import t as _t
        crit = float(_t.ppf(0.975, df=n - 1))
    except Exception:
        crit = 1.96
    low, high = mean - crit * sem, mean + crit * sem
    if floor is not None:
        low = max(floor, low)
    return mean, low, high


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Centralised RRT for MultiDrone.")
    parser.add_argument("--env", default="environment.yaml", help="environment YAML file")
    parser.add_argument("--drones", type=int, default=2, help="number of drones K")
    parser.add_argument("--time-limit", type=float, default=20.0, help="seconds per run")
    parser.add_argument("--step-size", type=float, default=None, help="RRT extension length")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="goal sampling probability")
    parser.add_argument("--partial-goal-bias", type=float, default=0.25,
                        help="probability of a partial (per-drone) goal sample")
    parser.add_argument("--no-greedy", action="store_true",
                        help="disable greedy extension towards goal samples")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--runs", type=int, default=1, help="repetitions for batch statistics")
    parser.add_argument("--no-smooth", action="store_true", help="disable shortcutting")
    parser.add_argument("--no-viz", action="store_true", help="skip the 3D visualisation")
    args = parser.parse_args()

    from multi_drone import MultiDrone

    sim = MultiDrone(num_drones=args.drones, environment_file=args.env)

    search_times, total_times, lengths, nodes, iterations = [], [], [], [], []
    successes = 0
    last_path = None

    for run in range(args.runs):
        path, stats = rrt_plan(
            sim,
            time_limit=args.time_limit,
            step_size=args.step_size,
            goal_bias=args.goal_bias,
            partial_goal_bias=args.partial_goal_bias,
            greedy_goal_connect=not args.no_greedy,
            seed=args.seed + run,
            smooth=not args.no_smooth,
        )
        status = "solved" if stats["success"] else "FAILED"
        print(
            f"run {run + 1:3d}/{args.runs}: {status}  "
            f"search={stats['search_time']:6.2f}s  total={stats['total_time']:6.2f}s  "
            f"nodes={stats['nodes']:6d}  iters={stats['iterations']:7d}  "
            f"len={stats.get('smoothed_path_length', stats['path_length']):8.2f}"
        )
        if stats["success"]:
            successes += 1
            search_times.append(stats["search_time"])
            total_times.append(stats["total_time"])
            lengths.append(stats.get("smoothed_path_length", stats["path_length"]))
            nodes.append(stats["nodes"])
            iterations.append(stats["iterations"])
            last_path = path

    if args.runs > 1:
        print(f"\nK={args.drones}, env={args.env}, {args.runs} runs")
        print(f"success rate      : {successes / args.runs:.2%} ({successes}/{args.runs})")
        for label, data in (
            ("search time (s)", search_times),
            ("total time (s)", total_times),
            ("path length", lengths),
            ("tree nodes", nodes),
            ("iterations", iterations),
        ):
            mean, lo, hi = _mean_and_ci(data)
            print(f"{label:18s}: mean={mean:10.3f}  95% CI=[{lo:10.3f}, {hi:10.3f}]")

    if last_path is not None and not args.no_viz:
        sim.visualize_paths(last_path)


if __name__ == "__main__":
    main()
