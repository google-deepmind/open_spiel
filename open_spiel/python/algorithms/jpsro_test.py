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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.jpsro."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.algorithms import jpsro
import pyspiel

GAMES = (
    "sheriff_2p_gabriele",
)
SWEEP_KWARGS = [
    dict(  # pylint: disable=g-complex-comprehension
        game_name=game,
        iterations=iterations,
        policy_init=policy_init,
        update_players_strategy=update_players_strategy,
        target_equilibrium=target_equilibrium,
        br_selection=br_selection,
        train_meta_solver=train_meta_solver,
        eval_meta_solver=eval_meta_solver,
        ignore_repeats=ignore_repeats,
    ) for (
        iterations,
        game,
        policy_init,
        update_players_strategy,
        target_equilibrium,
        br_selection,
        train_meta_solver,
        eval_meta_solver,
        ignore_repeats) in itertools.product(
            [2],
            GAMES,
            jpsro.INIT_POLICIES,
            jpsro.UPDATE_PLAYERS_STRATEGY,
            jpsro.BRS,
            jpsro.BR_SELECTIONS,
            jpsro.META_SOLVERS,
            ["mwcce"],
            [True, False])
]
TEST_COUNT_LIMIT = 100

interval = len(SWEEP_KWARGS) // TEST_COUNT_LIMIT
interval = interval if interval % 2 != 0 else interval + 1  # Odd interval.
SWEEP_KWARGS = SWEEP_KWARGS[::interval]


def get_game(game_name):
  """Returns the game."""
  if game_name == "kuhn_poker_3p":
    game_name = "kuhn_poker"
    game_kwargs = {"players": int(3)}
  elif game_name == "trade_comm_2p_2i":
    game_name = "trade_comm"
    game_kwargs = {"num_items": int(2)}
  elif game_name == "sheriff_2p_gabriele":
    game_name = "sheriff"
    game_kwargs = {
        "item_penalty": float(1.0),
        "item_value": float(5.0),
        "max_bribe": int(2),
        "max_items": int(10),
        "num_rounds": int(2),
        "sheriff_penalty": float(1.0),
    }

  else:
    raise ValueError("Unrecognised game: %s" % game_name)
  return pyspiel.load_game_as_turn_based(game_name, game_kwargs)


class JPSROTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(*SWEEP_KWARGS)
  def test_jpsro_cce(self, **kwargs):
    game = get_game(kwargs["game_name"])
    jpsro.run_loop(game=game, **kwargs)


if __name__ == "__main__":
  absltest.main()
