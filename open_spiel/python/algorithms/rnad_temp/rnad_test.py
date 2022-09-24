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

"""Tests for google3.third_party.open_spiel.python.algorithms.rnad_temp.rnad."""

from absl.testing import absltest

from open_spiel.python.algorithms.rnad_temp import rnad
import pyspiel

# TODO(perolat): test the losses and jax ops


class RNADTest(absltest.TestCase):

  def test_run_kuhn(self):
    game = pyspiel.load_game("kuhn_poker")
    rnad_solver = rnad.RNaDSolver(game=game)
    for _ in range(10):
      rnad_solver.step()
    rnad_state = rnad_solver.__getstate__()
    rnad_solver = rnad.RNaDSolver(game=game)
    rnad_solver.__setstate__(rnad_state)
    for _ in range(10):
      rnad_solver.step()


if __name__ == "__main__":
  absltest.main()
