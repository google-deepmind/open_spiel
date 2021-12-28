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

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel


def main(_):
  game = pyspiel.load_game("kuhn_poker")

  cfr_solver = cfr.CFRSolver(game)
  iterations = 1000

  for i in range(iterations):
    cfr_value = cfr_solver.evaluate_and_update_policy()
    print("Game util at iteration {}: {}".format(i, cfr_value))

  average_policy = cfr_solver.average_policy()
  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  print("Computed player 0 value: {}".format(average_policy_values[0]))
  print("Expected player 0 value: {}".format(-1 / 18))


if __name__ == "__main__":
  app.run(main)
