# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This compares the speed/results of Jax CFR to of the original impl of CFR.

The results slightly differ due to different rounding of regrets between
original implmentation and CFR. When setting clamping of regrets to 1e-8 the
results are exactly the same.
"""


# pylint: disable=g-importing-member

import time
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.cfr import CFRPlusSolver
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
import pyspiel


def compare_cfr_with_jax_cfr(game):
  """Do the comparison."""

  start = time.time()
  jax_cfr = JaxCFR(game)
  print(time.time() - start)
  jax_cfr.multiple_steps(10000)
  print(time.time() - start)

  start = time.time()
  print(time.time() - start)
  cfr = CFRPlusSolver(game)
  for _ in range(1000):
    cfr.evaluate_and_update_policy()

  print(time.time() - start)

  jax_strat = jax_cfr.average_policy()
  jax_br1 = BestResponsePolicy(jax_cfr.game, 1, jax_strat)
  jax_br2 = BestResponsePolicy(jax_cfr.game, 0, jax_strat)

  cfr_strat = jax_cfr.average_policy()
  cfr_br1 = BestResponsePolicy(jax_cfr.game, 1, cfr_strat)
  cfr_br2 = BestResponsePolicy(jax_cfr.game, 0, cfr_strat)

  print("Jax P1: ", jax_br1.value(jax_cfr.game.new_initial_state()))
  print("CFR P1: ", cfr_br1.value(jax_cfr.game.new_initial_state()))
  print("Jax P2: ", jax_br2.value(jax_cfr.game.new_initial_state()))
  print("CFR P2: ", cfr_br2.value(jax_cfr.game.new_initial_state()))


# Speed Results:
#   Original: 139.60753107070923
#   Jax CPU: 3.7404067516326904
def compare_leduc():
  game = pyspiel.load_game("leduc_poker")
  compare_cfr_with_jax_cfr(game)


# Speed Results:
#   Original: 335.6707363128662
#   Jax CPU: 7.59996485710144
def compare_battleship():
  game_params = {
      "board_height": 2,
      "board_width": 2,
      "num_shots": 4,
      "ship_sizes": "[2]",
      "ship_values": "[1]",
      "allow_repeated_shots": False,
  }
  game = pyspiel.load_game("battleship", game_params)
  compare_cfr_with_jax_cfr(game)


# Speed Results:
#   Original: 14.667663097381592
#   Jax CPU: 1.068636417388916
def compare_goofspiel_descending():
  game_params = {"num_cards": 4, "imp_info": True, "points_order": "descending"}
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  compare_cfr_with_jax_cfr(game)


# Speed Results:
#   Original: 6.639796733856201
#   Jax CPU: 0.8599820137023926
def compare_goofspiel_randomized():
  game_params = {"num_cards": 3, "imp_info": True, "points_order": "random"}
  game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
  compare_cfr_with_jax_cfr(game)


if __name__ == "__main__":
  compare_leduc()
  compare_battleship()
  compare_goofspiel_descending()
  compare_goofspiel_randomized()
