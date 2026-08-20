"""
A MultiDrone whose vedo view matches the workspace size.

``MultiDrone._init_plot`` hardcodes the axes to (0, 50) on all three axes, and
sizes the drone markers for a 50-unit box (``Sphere(r=0.1)``, ``arm_len = 0.5``).
That is correct for the provided ``environment.yaml``, which is exactly 50^3,
but on any other workspace the grid covers the wrong region - in a 100^3 world
it draws only the near octant, so the scene appears to float outside its own
axes and the drones are too small to see.

This subclass reads the real bounds out of the environment and scales both the
axes and the drone markers to match. ``multi_drone.py`` is left untouched.

    from scaled_drone import ScaledMultiDrone

    sim = ScaledMultiDrone(num_drones=5, environment_file="env_0.yaml")
    path, stats = rrt_connect_plan(sim, time_limit=20.0, seed=0)
    sim.visualize_paths(path)

Collision checking is inherited unchanged - only the rendering differs.
"""

from __future__ import annotations

import numpy as np
from vedo import Plotter, Line, Sphere, Cylinder

from multi_drone import MultiDrone

# The marker sizes in multi_drone.py are tuned for a workspace of this side length.
REFERENCE_EXTENT = 50.0


class ScaledMultiDrone(MultiDrone):
    """MultiDrone with axes and drone markers scaled to the actual bounds."""

    @property
    def viz_scale(self) -> float:
        """How much bigger this workspace is than the one the defaults assume."""
        extent = float(np.max(self._bounds[:, 1] - self._bounds[:, 0]))
        return extent / REFERENCE_EXTENT

    def _init_plot(self) -> None:
        scale = self.viz_scale
        self._plotter = Plotter(interactive=False)
        self._drone_visuals = []

        for i in range(self.N):
            body = Sphere(r=0.1 * scale).c("cyan")
            arm1 = Cylinder(r=0.03 * scale, height=1.0 * scale).c("black")
            arm2 = Cylinder(r=0.03 * scale, height=1.0 * scale).c("black")
            traj = Line(np.array(self.trajectories[i])).lw(2).c("blue")
            self._drone_visuals.append((body, arm1, arm2, traj))

        visuals_flat = []
        for i in range(self.N):
            visuals_flat.extend(self._drone_visuals[i])
        visuals_flat.extend(self._obstacles_viz)
        visuals_flat.extend(self._goal_viz)

        lower, upper = self._bounds[:, 0], self._bounds[:, 1]
        self._plotter.show(
            *visuals_flat,
            axes=dict(
                xrange=(float(lower[0]), float(upper[0])),
                yrange=(float(lower[1]), float(upper[1])),
                zrange=(float(lower[2]), float(upper[2])),
                xygrid=True,
                yzgrid=True,
                zxgrid=True,
            ),
            viewup="z",
            interactive=False,
            mode=8,
        )

    def _update_plot(self) -> None:
        scale = self.viz_scale
        arm_len = 0.5 * scale

        for i in range(self.N):
            pos = self.configuration[i]
            body, arm1, arm2, traj = self._drone_visuals[i]

            arm1_p1 = pos + np.array([-arm_len, 0, 0])
            arm1_p2 = pos + np.array([arm_len, 0, 0])
            arm2_p1 = pos + np.array([0, -arm_len, 0])
            arm2_p2 = pos + np.array([0, arm_len, 0])

            self._plotter.remove(arm1)
            self._plotter.remove(arm2)
            self._plotter.remove(traj)

            new_arm1 = Cylinder(pos=[arm1_p1, arm1_p2], r=0.03 * scale).c("black")
            new_arm2 = Cylinder(pos=[arm2_p1, arm2_p2], r=0.03 * scale).c("black")
            new_traj = Line(np.array(self.trajectories[i])).lw(2).c(traj.color())

            self._plotter.add(new_arm1)
            self._plotter.add(new_arm2)
            self._plotter.add(new_traj)

            body.pos(pos)
            self._drone_visuals[i] = (body, new_arm1, new_arm2, new_traj)

        self._plotter.reset_camera()
        self._plotter.render()
