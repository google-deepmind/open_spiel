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
"""Tests for RNaD algorithm under open_spiel."""

import pickle

from absl.testing import absltest
import jax
import numpy as np

from open_spiel.python.algorithms.rnad import rnad

# TODO(author18): test the losses and jax ops


class RNADTest(absltest.TestCase):

  def test_run_kuhn(self):
    solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="kuhn_poker"))
    for _ in range(10):
      solver.step()

  def test_serialization(self):
    solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="kuhn_poker"))
    solver.step()

    state_bytes = pickle.dumps(solver)
    solver2 = pickle.loads(state_bytes)

    self.assertEqual(solver.config, solver2.config)
    np.testing.assert_equal(
        jax.device_get(solver.params), jax.device_get(solver2.params))

    # TODO(author16): figure out the last bits of the non-determinism
    #             and reenable the checks below.
    # Now run both solvers for the same number of steps and verify
    # they behave in exactly the same way.
    # for _ in range(10):
    #   solver.step()
    #   solver2.step()
    # np.testing.assert_equal(
    #     jax.device_get(solver.params), jax.device_get(solver2.params))


if __name__ == "__main__":
  absltest.main()
