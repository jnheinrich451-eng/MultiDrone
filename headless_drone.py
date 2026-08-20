"""
A MultiDrone that never opens a 3D window.

In ``multi_drone.py``, vedo is used *only* for visualisation: ``_init_plot``,
``_update_plot`` and ``visualize_paths``. Every collision query - ``is_valid``,
``motion_valid``, ``is_goal`` - goes through fcl and touches no vedo object at
all. So the 3D renderer is optional for planning, and skipping it costs nothing.

That matters because ``MultiDrone.__init__`` calls ``reset()``, which calls
``_init_plot()``, so a VTK render window is created the moment you construct a
simulator - even with ``--no-viz``, and even on a headless machine where it
prints ``bad X server connection``. Building one per environment in a sweep is
pure overhead.

This subclass overrides the three plotting methods and leaves everything else
untouched, so ``multi_drone.py`` stays exactly as provided:

    from headless_drone import HeadlessMultiDrone
    from rrt_connect import rrt_connect_plan

    sim = HeadlessMultiDrone(num_drones=5, environment_file="env_0.yaml")
    path, stats = rrt_connect_plan(sim, time_limit=20.0, seed=0)
    sim.visualize_paths(path, out="path.png")     # matplotlib, not VTK

``visualize_paths`` keeps its name and meaning but renders through
``plot_path``, so existing code that calls it keeps working - it just produces
a PNG instead of an interactive window.

Note this still *imports* vedo (``multi_drone`` does, at module level); it only
stops the window being created. If you want to drop the dependency entirely you
have to edit ``multi_drone.py`` itself.
"""

from __future__ import annotations

from multi_drone import MultiDrone


class HeadlessMultiDrone(MultiDrone):
    """MultiDrone with the VTK/vedo window disabled.

    Collision checking, bounds, goals and every public query are inherited
    unchanged - only the rendering is replaced.
    """

    def __init__(self, num_drones: int, environment_file: str = "obstacles.yaml"):
        # Recorded before super().__init__, because that calls reset() -> _init_plot()
        # and visualize_paths needs the path to re-read the obstacle geometry.
        self.environment_file = environment_file
        super().__init__(num_drones, environment_file)

    # -- rendering, disabled -------------------------------------------------

    def _init_plot(self) -> None:
        """No Plotter, no render window, no X server required."""
        return None

    def _update_plot(self) -> None:
        return None

    def visualize_paths(self, path, out: str | None = "path.png",
                        show: bool = False, **kwargs):
        """Render the path with matplotlib instead of VTK.

        Args:
            path: list of (N, 3) configurations, as returned by the planners.
            out: PNG to write, or None to only build the figure (useful in a
                notebook, where the inline backend displays it).
            show: also open a window - needs a display.

        Returns:
            (fig, ax) from ``plot_path``.
        """
        from plot_path import plot_path
        return plot_path(path, self.environment_file, out=out, show=show, **kwargs)

    def plot_separation(self, path, out: str | None = "separation.png",
                        show: bool = False):
        """Plot the closest drone-drone distance along the path against the
        collision threshold - an independent check that the path is safe."""
        from plot_path import plot_drone_separation
        return plot_drone_separation(path, drone_radius=self._drone_radius,
                                     out=out, show=show)
