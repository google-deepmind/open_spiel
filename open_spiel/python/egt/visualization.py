# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization for single/multi-population dynamics in normal-form games.

  Example:

  game = pyspiel.load_game("matrix_pd")
  payoff_tensor = utils.game_payoffs_array(game)
  dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)

  ax = plt.subplot(projection="2x2")
  ax.quiver(dyn)
"""

from absl import logging

# pylint: disable=g-import-not-at-top
try:
  from matplotlib import axes
  from matplotlib import projections
  from matplotlib import transforms
  from matplotlib import font_manager
  from matplotlib import rcParams
  from matplotlib.text import Text
  from matplotlib.path import Path
  from matplotlib.patches import PathPatch
  from matplotlib.patches import FancyArrowPatch
  from matplotlib.collections import LineCollection
  import matplotlib.cm
  import matplotlib.colors
except ImportError as e:
  logging.info("If your tests failed with the error 'ImportError: No module "
               "named functools_lru_cache', this is a known bug in matplotlib "
               "and there is a workaround (run sudo apt install "
               "python-backports.functools-lru-cache. See: "
               "https://github.com/matplotlib/matplotlib/issues/9344.")
  raise ImportError(str(e))

import numpy as np

from open_spiel.python.egt import utils


def _eval_dynamics_2x2_grid(dynamics, num_points):
  """Evaluates dynamics on a 2-D mesh-grid.

  Args:
    dynamics: Population dynamics of type `dynamics.MultiPopulationDynamics`.
    num_points: Number of points along each dimension of the grid.

  Returns:
    Mesh-grid (x, y) and corresponding derivatives of the first action for
      player 1 and 2 (u, v).
  """
  assert dynamics.payoff_tensor.shape == (2, 2, 2)

  x = np.linspace(0., 1., num_points + 2)[1:-1]
  x, y = np.meshgrid(x, x)
  u = np.empty(x.shape)
  v = np.empty(x.shape)

  for i in range(num_points):
    for j in range(num_points):
      row_state = np.array([x[i, j], 1. - x[i, j]])
      col_state = np.array([y[i, j], 1. - y[i, j]])
      state = np.concatenate((row_state, col_state))
      dstate = dynamics(state)
      u[i][j] = dstate[0]
      v[i][j] = dstate[2]
  return x, y, u, v


def _rk12_step(func, y0, dt):
  """Improved Euler-Integration step to integrate dynamics.

  Args:
    func: Function handle to time derivative.
    y0:   Current state.
    dt:   Integration step.

  Returns:
    Next state.
  """
  dy = func(y0)
  y_ = y0 + dt * dy
  return y0 + dt / 2. * (dy + func(y_))


class Dynamics2x2Axes(axes.Axes):
  """Axes for 2x2 game dynamics.

  This class provides plotting functions for dynamics in two-player 2x2 games.

  Attributes:
    name: Used for projection keyword when creating a new axes.
  """
  name = "2x2"

  def cla(self):
    """Clear the current axes."""
    super(Dynamics2x2Axes, self).cla()
    self.set_aspect("equal")
    self.set_xlim(0, 1)
    self.set_ylim(0, 1)

  def quiver(self,
             dynamics,
             num_points=9,
             normalize=False,
             pivot="middle",
             **kwargs):
    """Visualizes the dynamics as a directional field plot.

    Args:
      dynamics: Population dynamics of type `dynamics.MultiPopulationDynamics`.
      num_points: Number of points along each dimension of the plot.
      normalize: Normalize each arrow to unit-length.
      pivot: In `{"tail", "middle", "tip"}`, optional, default: "middle". The
        part of the arrow that is anchored to the X, Y grid. The arrow rotates
        about this point.
      **kwargs: Additional keyword arguments passed on to `Axes.quiver`.

    Returns:
      The `quiver.Quiver` object created by calling `Axes.quiver`.
    """
    x, y, u, v = _eval_dynamics_2x2_grid(dynamics, num_points)

    if normalize:
      norm = np.sqrt(u**2 + v**2)
      u = np.divide(u, norm, out=np.zeros_like(u), where=norm != 0)
      v = np.divide(v, norm, out=np.zeros_like(v), where=norm != 0)

    return super(Dynamics2x2Axes, self).quiver(
        x, y, u, v, pivot=pivot, **kwargs)

  def streamplot(self,
                 dynamics,
                 num_points=50,
                 linewidth=None,
                 color=None,
                 **kwargs):
    """Visualizes the dynamics as a streamline plot.

    Args:
      dynamics: Population dynamics of type `dynamics.MultiPopulationDynamics`.
      num_points: Number of points along each dimension of the plot.
      linewidth: In `{None, float, "velocity"}`, optional, default: None. If
        `linewidth="velocity"`, line width is scaled by the velocity of the
        dynamics. Defaults to `rcParams` if `linewidth=None`.
      color: In `{None, string, (r,g,b), (r,g,b,a), "velocity"}`, default: None.
        If `color="velocity"`, velocity of dynamics is used to color the
        streamlines. Defaults to `rcParams` if `color=None`.
      **kwargs: Additional keyword arguments passed on to `Axes.streamplot`.

    Returns:
      The `streamplot.StreamplotSet` created by calling `Axes.streamplot`.
    """

    x, y, u, v = _eval_dynamics_2x2_grid(dynamics, num_points)

    if linewidth == "velocity" or color == "velocity":
      vel = np.sqrt(u**2 + v**2)
      vel = vel - np.min(vel)
      vel = vel / np.max(vel)

      if linewidth == "velocity":
        linewidth = 3. * vel

      if color == "velocity":
        color = vel

    return super(Dynamics2x2Axes, self).streamplot(
        x, y, u, v, minlength=0.1, linewidth=linewidth, color=color, **kwargs)


projections.register_projection(Dynamics2x2Axes)


class SimplexTransform(transforms.Transform):
  """Affine transform to project the 2-simplex to 2D Cartesian space."""
  input_dims = 3
  output_dims = 2

  _MATRIX = np.array([[0., 0.], [1., 0.], [0.5, np.sqrt(3) / 2.]])

  def transform_affine(self, values):
    return np.matmul(values, SimplexTransform._MATRIX)


class SimplexStreamMask(object):
  """Mask of regular discrete cells to track trajectories/streamlines.

  Also see `matplotlib.streamplot.StreamMask`.
  """

  def __init__(self, density=1.):
    self._n = np.int(30. * density)
    self._mask = np.zeros([self._n + 1] * 2 + [2], dtype=np.bool)
    self.shape = self._mask.shape

  def index(self, point):
    """Computes index given a point on the simplex."""
    point = np.array(point)
    idx = np.floor(point[:2] * self._n).astype(int)
    x, y = point[:2] * self._n - idx
    z = int(x + y > 1)
    return tuple(idx.tolist() + [z])

  def point(self, index):
    """Computes point on the simplex given an index."""
    p = np.empty((3,))
    p[0] = (index[0] + (1 + index[2]) / 3.) / float(self._n)
    p[1] = (index[1] + (1 + index[2]) / 3.) / float(self._n)
    p[2] = 1. - p[0] - p[1]
    return p if p[2] > 0. else None

  def __getitem__(self, point):
    return self._mask.__getitem__(self.index(point))

  def __setitem__(self, point, val):
    return self._mask.__setitem__(self.index(point), val)


class Dynamics3x3Axes(axes.Axes):
  """Axes for 3x3 game dynamics.

  This class provides plotting functions for dynamics in symmetric 3x3 games.

  Attributes:
    name: Used for projection keyword when creating a new axes.
  """
  name = "3x3"
  _VERTICES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

  def __init__(self, fig, rect, *args, **kwargs):
    self._simplex_transform = SimplexTransform()
    self._labels = None
    super(axes.Axes, self).__init__(fig, rect, *args, **kwargs)

  def cla(self):
    """Clear the current axes."""
    super(axes.Axes, self).cla()
    self.set_aspect("equal")
    self.get_xaxis().set_visible(False)
    self.get_yaxis().set_visible(False)
    self.patch.set_visible(False)
    self.set_frame_on(False)

    # draw invisiple vertices to set x/y limits of plot
    self.scatter(Dynamics3x3Axes._VERTICES, alpha=0.)
    self.margins(0.15)

    self.bgpatch = self._create_bgpatch(
        facecolor=rcParams["axes.facecolor"],
        edgecolor=rcParams["axes.edgecolor"],
        linewidth=rcParams["axes.linewidth"],
        zorder=-1)
    self.add_artist(self.bgpatch)

    if rcParams["axes.grid"]:
      self.grid = self._create_grid(
          color=rcParams["grid.color"],
          alpha=rcParams["grid.alpha"],
          linestyle=rcParams["grid.linestyle"],
          linewidth=rcParams["grid.linewidth"],
          zorder=0)
      self.add_collection(self.grid)

    self.ticks, self.tick_labels = self._create_ticks(
        color=rcParams["xtick.color"], zorder=0)
    self.add_collection(self.ticks)
    for label in self.tick_labels:
      self.add_artist(label)

  def _create_bgpatch(self, **kwargs):
    codes = [Path.MOVETO] + [Path.LINETO] * 2 + [Path.CLOSEPOLY]
    vertices = self._VERTICES + [self._VERTICES[0]]
    vertices = self._simplex_transform.transform(np.array(vertices))
    return PathPatch(Path(vertices, codes), **kwargs)

  def _create_grid(self, step=0.2, **kwargs):
    x = np.arange(step, 1., step)
    n = x.shape[0]
    line_start, line_end = np.zeros((n, 3)), np.zeros((n, 3))
    line_start[:, 0] = line_end[::-1, 1] = x
    line_start[:, 2] = line_end[::-1, 0] = 1. - x
    segs = np.zeros((3 * n, 2, 2))
    for i, perm in enumerate([(0, 2, 1), (1, 0, 2), (2, 1, 0)]):
      start = self._simplex_transform.transform(line_start[:, perm])
      end = self._simplex_transform.transform(line_end[:, perm])
      segs[i * n:(i + 1) * n, 0, :], segs[i * n:(i + 1) * n, 1, :] = start, end
    line_segments = LineCollection(segs, **kwargs)
    return line_segments

  def _create_ticks(self, step=0.2, tick_length=0.025, **kwargs):
    x = np.arange(step, 1., step)
    n = x.shape[0]

    tick_start, tick_end = np.zeros((n, 3)), np.zeros((n, 3))
    tick_start[:, 0] = x
    tick_start[:, 2] = 1. - x
    tick_end[:, 0] = x
    tick_end[:, 2] = 1. - x + tick_length
    tick_end[:, 1] = -tick_length

    tick_labels = []
    ha = ["center", "left", "right"]
    va = ["top", "bottom", "center"]
    rot = [-60, 60, 0]

    segs = np.zeros((n * 3, 2, 2))
    for i, perm in enumerate([(0, 2, 1), (1, 0, 2), (2, 1, 0)]):
      start = self._simplex_transform.transform(tick_start[:, perm])
      end = self._simplex_transform.transform(tick_end[:, perm])
      segs[i * n:(i + 1) * n, 0, :], segs[i * n:(i + 1) * n, 1, :] = start, end

      for j, x_ in enumerate(x):
        tick_labels.append(
            Text(
                end[j, 0],
                end[j, 1],
                "{0:.1f}".format(x_),
                horizontalalignment=ha[i],
                verticalalignment=va[i],
                rotation=rot[i],
                color=kwargs["color"],
                fontsize=rcParams["xtick.labelsize"]))
    line_segments = LineCollection(segs, **kwargs)
    return line_segments, tick_labels

  def _create_labels(self, labels, padding):
    artists = []
    aligns = ["top", "top", "bottom"]
    for label, pos, align in zip(labels, self._VERTICES, aligns):
      x, y = self._simplex_transform.transform(pos)
      labelpad = padding if align == "bottom" else -padding
      label = Text(
          x=x,
          y=y + labelpad,
          text=label,
          fontproperties=font_manager.FontProperties(
              size=rcParams["axes.labelsize"],
              weight=rcParams["axes.labelweight"]),
          color=rcParams["axes.labelcolor"],
          verticalalignment=align,
          horizontalalignment="center")
      artists.append(label)
    return artists

  def get_labels(self):
    return self._labels

  def set_labels(self, labels, padding=0.02):
    assert len(labels) == 3
    if self._labels is None:
      self._labels = self._create_labels(labels, padding)
      for label in self._labels:
        self.add_artist(label)
    else:
      for artist, label in zip(self._labels, labels):
        artist.set_text(label)

  labels = property(get_labels, set_labels)

  def can_zoom(self):
    return False

  def can_pan(self):
    return False

  def plot(self, points, **kwargs):
    """Creates a line plot.

    Args:
      points: Points in policy space.
      **kwargs: Additional keyword arguments passed on to `Axes.plot`.

    Returns:
      The line plot.
    """
    points = np.array(points)
    assert points.shape[1] == 3
    points = self._simplex_transform.transform(points)
    return super(Dynamics3x3Axes, self).plot(points[:, 0], points[:, 1],
                                             **kwargs)

  def scatter(self, points, **kwargs):
    """Creates a scatter plot.

    Args:
      points: Points in policy space.
      **kwargs: Additional keyword arguments passed on to `Axes.scatter`.

    Returns:
      The scatter plot.
    """
    points = np.array(points)
    assert points.shape[1] == 3
    points = self._simplex_transform.transform(points)
    return super(Dynamics3x3Axes, self).scatter(points[:, 0], points[:, 1],
                                                **kwargs)

  def quiver(self,
             dynamics,
             step=0.05,
             boundary=False,
             normalize=False,
             pivot="middle",
             **kwargs):
    """Visualizes the dynamics as a directional field plot.

    Args:
      dynamics: Population dynamics of type `dynamics.SinglePopulationDynamics`.
      step: Distance between arrows along one dimension.
      boundary: Include arrows on the boundary/face of the simplex.
      normalize: Normalize each arrow to unit-length.
      pivot: In `{"tail", "middle", "tip"}`, optional, default: "middle". The
        part of the arrow that is anchored to the X, Y grid. The arrow rotates
        about this point.
      **kwargs: Additional keyword arguments passed on to `Axes.quiver`.

    Returns:
      The `quiver.Quiver` object created by calling `Axes.quiver`.
    """
    x = np.array([x for x in utils.grid_simplex(step=step, boundary=boundary)])
    dx = np.apply_along_axis(dynamics, 1, x)

    p = self._simplex_transform.transform(x)
    v = self._simplex_transform.transform(dx)

    x, y = p[:, 0], p[:, 1]
    u, v = v[:, 0], v[:, 1]

    if normalize:
      norm = np.sqrt(u**2 + v**2)
      u, v = u / norm, v / norm

    if "pivot" not in kwargs:
      kwargs["pivot"] = "middle"

    return super(Dynamics3x3Axes, self).quiver(x, y, u, v, **kwargs)

  def _linecollection(self, points, linewidth, color):
    points = self._simplex_transform.transform(points).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=linewidth, color=color)
    return lc

  def _integrate(self, x, func, mask, dt, min_dist=0.01):
    cells = []
    trajectory = [x]
    x_ = x
    for dt in [dt, -dt]:
      while not mask[x]:
        cell = mask.index(x)
        cells.append(cell)
        while mask.index(x) == cell:
          # integrate up to cell boundary
          if np.sqrt(np.sum((x_ - x)**2)) > min_dist:
            x_ = x
            if dt > 0:
              trajectory.append(x)
            else:
              trajectory.insert(0, x)

          x = _rk12_step(func, x, dt=dt)

        if dt > 0:
          mask[trajectory[-1]] = True
        else:
          mask[trajectory[0]] = True

      # restore to integrate backwards
      if dt > 0. and len(cells):
        trajectory.append(_rk12_step(func, x, dt=-dt))
        mask[mask.point(cells[0])] = False
        x = trajectory[0]
        x_ = x
      else:
        trajectory.insert(0, _rk12_step(func, x, dt=-dt))
    return (np.array(trajectory), cells) if len(trajectory) > 2 else None

  def streamplot(self,
                 dynamics,
                 initial_points=None,
                 dt=0.01,
                 density=1.,
                 min_length=0.4,
                 linewidth=None,
                 color="k",
                 **kwargs):
    """Visualizes the dynamics as a streamline plot.

    Mimics the visuals of `Axes.streamplot` for simplex plots.

    Args:
      dynamics: Population dynamics of type `dynamics.SinglePopulationDynamics`.
      initial_points: Starting points for streamlines
      dt: Integration step.
      density: Controls the density of streamlines in the plot.
      min_length: Streamlines with length < min_length will be discarded.
      linewidth: In `{None, float, "velocity"}`, optional, default: None. If
        `linewidth="velocity"`, line width is scaled by the velocity of the
        dynamics. Defaults to `rcParams` if `linewidth=None`.
      color: In `{None, string, (r,g,b), (r,g,b,a), "velocity"}`, default: None.
        If `color="velocity"`, velocity of dynamics is used to color the
        streamlines. Defaults to `rcParams` if `color=None`.
      **kwargs: Additional keyword arguments passed on to `Axes.streamplot`.

    Returns:
      The `SimplexStreamMask`.
    """
    mask = SimplexStreamMask(density=density)
    trajectories = []

    if initial_points is None:
      eps = 0.1
      initial_points = np.array([[1. - eps, eps / 2., eps / 2.],
                                 [eps / 2., 1. - eps, eps / 2.],
                                 [eps / 2., eps / 2., 1. - eps]])
      initial_points = np.vstack(
          (initial_points, utils.sample_from_simplex(100)))
      # TODO(author10): add heuristic for initial points

    else:
      initial_points = np.array(initial_points)
      assert initial_points.ndim == 2
      assert initial_points.shape[1] == 3

    # generate trajectories
    for p in initial_points:
      # center initial point on grid cell
      p = mask.point(mask.index(p))
      res = self._integrate(p, dynamics, mask, dt=dt)
      if res is not None:
        t, cells = res
        cum_len = np.cumsum(
            np.sqrt(
                np.diff(t[:, 0])**2 + np.diff(t[:, 1])**2 +
                np.diff(t[:, 2])**2))
        if cum_len[-1] < min_length:
          for cell in cells:
            mask[mask.point(cell)] = False
          continue
        trajectories.append(t)

    lc_color = arrow_color = color
    lc_linewidth = linewidth

    if linewidth == "velocity" or color == "velocity":
      vel_max = 0
      vel_min = np.float("inf")
      velocities = []
      for t in trajectories:
        dx = np.apply_along_axis(dynamics, 1, t)
        vel = np.sqrt(np.sum(dx**2, axis=1))
        vel_max = max(np.max(vel), vel_max)
        vel_min = min(np.min(vel), vel_min)
        velocities.append(vel)

    # add trajectories to plot
    for i, t in enumerate(trajectories):
      cum_len = np.cumsum(
          np.sqrt(
              np.diff(t[:, 0])**2 + np.diff(t[:, 1])**2 + np.diff(t[:, 2])**2))
      mid_idx = np.searchsorted(cum_len, cum_len[-1] / 2.)

      if linewidth == "velocity" or color == "velocity":
        vel = (velocities[i] - vel_min) / vel_max

        if linewidth == "velocity":
          lc_linewidth = 3. * vel + 0.5

        if color == "velocity":
          cmap = matplotlib.cm.get_cmap(rcParams["image.cmap"])
          lc_color = cmap(vel)
          arrow_color = cmap(vel[mid_idx])

      lc = self._linecollection(t, linewidth=lc_linewidth, color=lc_color)
      self.add_collection(lc)

      # add arrow centered on trajectory
      arrow_tail = self._simplex_transform.transform(t[mid_idx - 1])
      arrow_head = self._simplex_transform.transform(t[mid_idx])
      arrow_kw = dict(arrowstyle="-|>", mutation_scale=10 * 1.)
      arrow_patch = FancyArrowPatch(
          arrow_tail,
          arrow_head,
          linewidth=None,
          color=arrow_color,
          zorder=3,
          **arrow_kw)
      self.add_patch(arrow_patch)
    return mask


projections.register_projection(Dynamics3x3Axes)
