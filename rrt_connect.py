"""
Bidirectional RRT (RRT-Connect) for the MultiDrone simulator.

Grows two trees - one rooted at the initial configuration, one rooted at a
valid configuration inside the goal region - and greedily drives each toward
the newest node of the other. Compared with the single-tree planner in
``rrt_planner.py``, the goal side no longer has to be discovered by chance: it
advances toward the start using the same greedy extension the start side uses,
which is what makes bidirectional search strong in cluttered spaces.

Everything except the search loop is inherited from ``RRTPlanner``: the
samplers, ``steer``, the flat (3K,) <-> (K, 3) boundary, and the vectorised
nearest-neighbour tree.

    from multi_drone import MultiDrone
    from rrt_connect import rrt_connect_plan

    sim = MultiDrone(num_drones=5, environment_file="env_0.yaml")
    path, stats = rrt_connect_plan(sim, time_limit=20.0, seed=0)
"""

from __future__ import annotations

import time
import numpy as np

from rrt_planner import RRTPlanner, Tree, path_length, shortcut_path

REACHED, TRAPPED, ADVANCED = "REACHED", "TRAPPED", "ADVANCED"


class RRTConnectPlanner(RRTPlanner):
    """Bidirectional RRT.

    Pseudo-code (one iteration):

        q_rand <- Sample()
        q_new  <- Extend(T_a, q_rand)              # one step_size extension
        if q_new is not None:
            status, node_b <- Connect(T_b, q_new)  # greedy, until blocked
            if status = REACHED: return Join(...)
        Swap(T_a, T_b)                             # unconditionally

    Two details that are easy to get wrong:

    * The swap must happen even when the extension fails, otherwise one tree
      gets extended repeatedly while the other stalls.
    * After an odd number of swaps ``T_a`` is the *goal* tree, so the join has
      to know which tree is which, or it returns the path backwards.

    Goal biasing is off by default. In a bidirectional planner the goal tree
    already pulls toward the start, and biasing samples toward ``q_goal`` while
    the roles are swapped makes the goal tree grow back toward its own root.
    """

    def __init__(self, sim, goal_bias: float = 0.0, partial_goal_bias: float = 0.0,
                 max_goal_root_tries: int = 200, **kwargs):
        super().__init__(sim, goal_bias=goal_bias,
                         partial_goal_bias=partial_goal_bias, **kwargs)
        self.max_goal_root_tries = int(max_goal_root_tries)

    # -- goal root -----------------------------------------------------------

    def _goal_root(self) -> np.ndarray:
        """A *valid* configuration inside the goal region, to root the second
        tree at.

        The goal is a region, not a point, and the configuration made of the
        goal centres may itself be in collision - so we cannot simply root the
        tree at ``q_goal`` the way a point-goal formulation would.
        """
        for _ in range(self.max_goal_root_tries):
            q = self.sample_goal()
            if self.is_valid(q):
                return q
        raise ValueError(
            "No valid configuration found inside the goal region; a goal may be "
            "blocked by an obstacle, or two goals may be too close together.")

    # -- the two primitives --------------------------------------------------

    def _extend(self, tree: Tree, q_target: np.ndarray):
        """One single-step extension of ``tree`` toward ``q_target``."""
        idx, _ = tree.nearest(q_target)
        q_near = tree.config(idx)
        q_new = self.steer(q_near, q_target)

        if not self.is_valid(q_new):
            self.stats["rejected_configs"] += 1
            return TRAPPED, idx, None
        if not self.motion_valid(q_near, q_new):
            self.stats["rejected_motions"] += 1
            return TRAPPED, idx, None

        new_idx = tree.add(q_new, idx)
        reached = float(np.linalg.norm(q_target - q_new)) < 1e-9
        return (REACHED if reached else ADVANCED), new_idx, q_new

    def _connect(self, tree: Tree, q_target: np.ndarray, deadline: float):
        """Extend ``tree`` toward ``q_target`` repeatedly until it arrives or is
        blocked. This greedy step is what distinguishes RRT-Connect."""
        idx, _ = tree.nearest(q_target)
        q_cur = tree.config(idx)

        while True:
            q_next = self.steer(q_cur, q_target)

            if not self.is_valid(q_next):
                self.stats["rejected_configs"] += 1
                return TRAPPED, idx
            if not self.motion_valid(q_cur, q_next):
                self.stats["rejected_motions"] += 1
                return TRAPPED, idx

            idx = tree.add(q_next, idx)
            if float(np.linalg.norm(q_target - q_next)) < 1e-9:
                return REACHED, idx
            if time.perf_counter() >= deadline:
                return TRAPPED, idx
            q_cur = q_next

    # -- joining -------------------------------------------------------------

    @staticmethod
    def _join(tree_a: Tree, idx_a: int, tree_b: Tree, idx_b: int,
              a_is_start: bool) -> list[np.ndarray]:
        """Splice the two branches into one start -> goal path.

        ``Tree.reconstruct_path`` returns root -> node, so the start branch is
        already correctly ordered and the goal branch must be reversed. The
        meeting configuration appears in both branches, hence the ``[1:]``.
        """
        branch_a = tree_a.reconstruct_path(idx_a)
        branch_b = tree_b.reconstruct_path(idx_b)

        if a_is_start:
            return branch_a + list(reversed(branch_b))[1:]
        return branch_b + list(reversed(branch_a))[1:]

    # -- main loop -----------------------------------------------------------

    def plan(self, time_limit: float = 20.0) -> list[np.ndarray] | None:
        t0 = time.perf_counter()
        deadline = t0 + time_limit
        self.stats = {
            "success": False, "search_time": 0.0, "iterations": 0, "nodes": 2,
            "rejected_configs": 0, "rejected_motions": 0,
            "path_length": float("nan"), "waypoints": 0,
            "nodes_start": 1, "nodes_goal": 1,
        }

        if not self.is_valid(self.q_start):
            self.stats["search_time"] = time.perf_counter() - t0
            raise ValueError("The initial configuration is in collision.")

        tree_start = Tree(self.q_start)
        tree_goal = Tree(self._goal_root())

        if self.is_goal(self.q_start):
            return self._finish([self.q_start, self.q_start.copy()],
                                t0, tree_start, tree_goal)

        tree_a, tree_b = tree_start, tree_goal
        a_is_start = True

        while time.perf_counter() < deadline:
            self.stats["iterations"] += 1

            u = self.rng.random()
            if u < self.goal_bias:
                q_rand = self.sample_goal()
            elif u < self.goal_bias + self.partial_goal_bias:
                q_rand = self.sample_partial_goal()
            else:
                q_rand = self.sample_uniform()

            _, idx_a, q_new = self._extend(tree_a, q_rand)

            if q_new is not None:
                status_b, idx_b = self._connect(tree_b, q_new, deadline)
                if status_b == REACHED:
                    flat = self._join(tree_a, idx_a, tree_b, idx_b, a_is_start)
                    return self._finish(flat, t0, tree_start, tree_goal)

            # Swap unconditionally, including after a failed extension.
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start

        self.stats["search_time"] = time.perf_counter() - t0
        self.stats["nodes_start"] = tree_start.size
        self.stats["nodes_goal"] = tree_goal.size
        self.stats["nodes"] = tree_start.size + tree_goal.size
        return None

    def _finish(self, flat_path, t0, tree_start, tree_goal):
        self.stats.update(
            success=True,
            search_time=time.perf_counter() - t0,
            nodes_start=tree_start.size,
            nodes_goal=tree_goal.size,
            nodes=tree_start.size + tree_goal.size,
            waypoints=len(flat_path),
            path_length=path_length(flat_path),
        )
        return [self._to_cfg(q) for q in flat_path]


def rrt_connect_plan(sim, time_limit: float = 20.0, step_size: float | None = None,
                     goal_bias: float = 0.0, partial_goal_bias: float = 0.0,
                     seed: int | None = None, smooth: bool = True,
                     smooth_fraction: float = 0.1):
    """Plan with bidirectional RRT. Mirrors ``rrt_planner.rrt_plan``."""
    t_start = time.perf_counter()
    smooth_budget = time_limit * smooth_fraction if smooth else 0.0
    planner = RRTConnectPlanner(sim, step_size=step_size, goal_bias=goal_bias,
                                partial_goal_bias=partial_goal_bias, seed=seed)
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bidirectional RRT for MultiDrone.")
    parser.add_argument("--env", default="environment.yaml")
    parser.add_argument("--drones", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from multi_drone import MultiDrone
    from rrt_planner import _mean_and_ci

    sim = MultiDrone(num_drones=args.drones, environment_file=args.env)
    times, lengths, nodes, successes, last = [], [], [], 0, None

    for run in range(args.runs):
        path, s = rrt_connect_plan(sim, time_limit=args.time_limit,
                                   step_size=args.step_size, seed=args.seed + run,
                                   smooth=not args.no_smooth)
        print(f"run {run + 1:3d}/{args.runs}: "
              f"{'solved' if s['success'] else 'FAILED'}  "
              f"search={s['search_time']:6.2f}s  "
              f"nodes={s['nodes']:6d} (start {s['nodes_start']}, goal {s['nodes_goal']})  "
              f"len={s.get('smoothed_path_length', s['path_length']):8.2f}")
        if s["success"]:
            successes += 1
            times.append(s["search_time"])
            lengths.append(s.get("smoothed_path_length", s["path_length"]))
            nodes.append(s["nodes"])
            last = path

    if args.runs > 1:
        print(f"\nK={args.drones}, env={args.env}, {args.runs} runs")
        print(f"success rate      : {successes / args.runs:.2%} ({successes}/{args.runs})")
        for label, data in (("search time (s)", times), ("path length", lengths),
                            ("tree nodes", nodes)):
            mean, lo, hi = _mean_and_ci(data)
            print(f"{label:18s}: mean={mean:10.3f}  95% CI=[{lo:10.3f}, {hi:10.3f}]")

    if last is not None and not args.no_viz:
        sim.visualize_paths(last)


if __name__ == "__main__":
    main()
