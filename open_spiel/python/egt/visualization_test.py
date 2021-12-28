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

"""Tests for open_spiel.python.egt.visualization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import absltest

# pylint: disable=g-import-not-at-top
try:
  from matplotlib.figure import Figure
  from matplotlib.quiver import Quiver
  from matplotlib.streamplot import StreamplotSet
except ImportError as e:
  logging.info("If your tests failed with the error 'ImportError: No module "
               "named functools_lru_cache', this is a known bug in matplotlib "
               "and there is a workaround (run sudo apt install "
               "python-backports.functools-lru-cache. See: "
               "https://github.com/matplotlib/matplotlib/issues/9344.")
  raise ImportError(str(e))

import numpy as np

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
import pyspiel


def _build_dynamics2x2():
  """Build multi-population dynamics."""
  game = pyspiel.load_game("matrix_pd")
  payoff_tensor = utils.game_payoffs_array(game)
  return dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)


def _build_dynamics3x3():
  """Build single-population dynamics."""
  game = pyspiel.load_game("matrix_rps")
  payoff_tensor = utils.game_payoffs_array(game)
  return dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)


def _identity_dynamics(x):
  """Returns same input as output."""
  return x


class VisualizationTest(absltest.TestCase):

  def test_meshgrid(self):
    n = 10
    payoff_tensor = np.ones(shape=(2, 2, 2))
    identity = lambda x, f: x
    allzero = lambda x, f: np.zeros(x.shape)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, (identity, allzero))
    x, y, u, v = visualization._eval_dynamics_2x2_grid(dyn, n)
    np.testing.assert_allclose(x, u)
    np.testing.assert_allclose(v, np.zeros(shape=(n, n)))

    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, (allzero, identity))
    x, y, u, v = visualization._eval_dynamics_2x2_grid(dyn, n)
    np.testing.assert_allclose(u, np.zeros(shape=(n, n)))
    np.testing.assert_allclose(y, v)

  def test_quiver2x2(self):
    """Test 2x2 quiver plot."""
    dyn = _build_dynamics2x2()
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="2x2")
    res = ax.quiver(dyn)
    self.assertIsInstance(res, Quiver)

  def test_streamplot2x2(self):
    """Test 2x2 quiver plot."""
    dyn = _build_dynamics2x2()
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="2x2")
    res = ax.streamplot(dyn)
    self.assertIsInstance(res, StreamplotSet)

  def test_quiver3x3(self):
    """Test 3x3 quiver plot."""
    dyn = _build_dynamics3x3()
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3x3")
    res = ax.quiver(dyn)
    self.assertIsInstance(res, Quiver)

  def test_streamplot3x3(self):
    """Test 3x3 quiver plot."""
    dyn = _build_dynamics3x3()
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3x3")
    res = ax.streamplot(dyn)
    self.assertIsInstance(res, visualization.SimplexStreamMask)


if __name__ == "__main__":
  absltest.main()
