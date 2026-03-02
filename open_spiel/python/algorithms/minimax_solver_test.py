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

from absl.testing import absltest

from open_spiel.python.algorithms import minimax_solver
import pyspiel


class MinimaxSolverTest(absltest.TestCase):

  def test_compute_game_optimal_value(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    solver = minimax_solver.MinimaxSolver("tic_tac_toe")
    solver.solve()
    state = tic_tac_toe.new_initial_state()
    action_values = solver.action_values_from_state(state)
    print(f"Optimal action values at initial state: {action_values}")
    self.assertAlmostEqual(action_values.max(), 0.0)

  def test_compute_game_trembling_hand_optimal_value(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    solver = minimax_solver.MinimaxSolver("tic_tac_toe", epsilon=0.1)
    solver.solve()
    state = tic_tac_toe.new_initial_state()
    action_values = solver.action_values_from_state(state)
    print(f"Trembling opt. action values at initial state: {action_values}")
    # center is best in this case
    self.assertGreater(action_values.max(), 0.0)
    self.assertEqual(action_values.max(), action_values[4])
    # sides all equal, corners all equal, and corners better than sides
    self.assertAlmostEqual(action_values[0], action_values[2])
    self.assertAlmostEqual(action_values[2], action_values[6])
    self.assertAlmostEqual(action_values[6], action_values[8])
    self.assertAlmostEqual(action_values[1], action_values[3])
    self.assertAlmostEqual(action_values[3], action_values[5])
    self.assertAlmostEqual(action_values[5], action_values[7])
    self.assertGreater(action_values[0], action_values[1])


if __name__ == "__main__":
  absltest.main()
