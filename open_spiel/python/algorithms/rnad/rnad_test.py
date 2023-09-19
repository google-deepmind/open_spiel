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
from absl.testing import parameterized
import jax
import numpy as np

from open_spiel.python.algorithms.rnad import rnad

# TODO(author18): test the losses and jax ops


class RNADTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      dict(
          testcase_name="3x2_5x1_6",
          sizes=[3, 5, 6],
          repeats=[2, 1, 1],
          cover_steps=24,
          expected=[
              (0, False),
              (2 / 3, False),
              (1, True),  # 3
              (0, False),
              (2 / 3, False),
              (1, True),  # 3 x 2
              (0, False),
              (0.4, False),
              (0.8, False),
              (1, False),
              (1, True),  # 5
              (0, False),
              (1 / 3, False),
              (2 / 3, False),
              (1, False),
              (1, False),
              (1, True),  # 6
              (0, False),
              (1 / 3, False),
              (2 / 3, False),
              (1, False),
              (1, False),
              (1, True),  # 6 x 2
              (0, False),
          ],
      ),
  )
  def test_entropy_schedule(self, sizes, repeats, cover_steps, expected):
    schedule = rnad.EntropySchedule(sizes=sizes, repeats=repeats)
    computed = [schedule(i) for i in range(cover_steps)]
    np.testing.assert_almost_equal(computed, expected)


if __name__ == "__main__":
  absltest.main()
