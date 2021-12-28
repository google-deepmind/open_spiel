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

"""Joint Policy-Space Response Oracles.

An implementation of JSPRO, described in https://arxiv.org/abs/2106.09435.

Bibtex / Cite:

```
@misc{marris2021multiagent,
    title={Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium
           Meta-Solvers},
    author={Luke Marris and Paul Muller and Marc Lanctot and Karl Tuyls and
            Thore Graepel},
    year={2021},
    eprint={2106.09435},
    archivePrefix={arXiv},
    primaryClass={cs.MA}
}
```
"""

from absl import app
from absl import flags

from open_spiel.python.algorithms import jpsro
import pyspiel


GAMES = (
    "kuhn_poker_2p",
    "kuhn_poker_3p",
    "kuhn_poker_4p",
    "leduc_poker_2p",
    "leduc_poker_3p",
    "leduc_poker_4p",
    "trade_comm_2p_2i",
    "trade_comm_2p_3i",
    "trade_comm_2p_4i",
    "trade_comm_2p_5i",
    "tiny_bridge_2p",
    "tiny_bridge_4p",
    "sheriff_2p_1r",
    "sheriff_2p_2r",
    "sheriff_2p_3r",
    "sheriff_2p_gabriele",
    "goofspiel_2p_3c_total",
    "goofspiel_2p_4c_total",
    "goofspiel_2p_5c_total",
)

FLAGS = flags.FLAGS

# Game.
flags.DEFINE_string(
    "game", "kuhn_poker_3p",
    "Game and settings name.")

# JPSRO - General.
flags.DEFINE_integer(
    "iterations", 40,
    "Number of JPSRO iterations.",
    lower_bound=0)
flags.DEFINE_integer(
    "seed", 1,
    "Pseduo random number generator seed.")
flags.DEFINE_enum(
    "policy_init", "uniform", jpsro.INIT_POLICIES,
    "Initial policy sampling strategy.")
flags.DEFINE_enum(
    "update_players_strategy", "all", jpsro.UPDATE_PLAYERS_STRATEGY,
    "Which player's policies to update at each iteration.")

# JPSRO - Best Response.
flags.DEFINE_enum(
    "target_equilibrium", "cce", jpsro.BRS,
    "The target equilibrium, either ce or cce.")
flags.DEFINE_enum(
    "br_selection", "largest_gap", jpsro.BR_SELECTIONS,
    "The best response operator. Primarily used with CE target equilibrium.")

# JPSRO - Meta-Solver.
flags.DEFINE_enum(
    "train_meta_solver", "mgcce", jpsro.META_SOLVERS,
    "Meta-solver to use for training.")
flags.DEFINE_enum(
    "eval_meta_solver", "mwcce", jpsro.META_SOLVERS,
    "Meta-solver to use for evaluation.")
flags.DEFINE_bool(
    "ignore_repeats", False,
    "Whether to ignore policy repeats when calculating meta distribution. "
    "This is relevant for some meta-solvers (such as Maximum Gini) that will "
    "spread weight over repeats. This may or may not be a desireable property "
    "depending on how one wishes to search the game space. A uniform "
    "meta-solver requires this to be False.")


def get_game(game_name):
  """Returns the game."""

  if game_name == "kuhn_poker_2p":
    game_name = "kuhn_poker"
    game_kwargs = {"players": int(2)}
  elif game_name == "kuhn_poker_3p":
    game_name = "kuhn_poker"
    game_kwargs = {"players": int(3)}
  elif game_name == "kuhn_poker_4p":
    game_name = "kuhn_poker"
    game_kwargs = {"players": int(4)}

  elif game_name == "leduc_poker_2p":
    game_name = "leduc_poker"
    game_kwargs = {"players": int(2)}
  elif game_name == "leduc_poker_3p":
    game_name = "leduc_poker"
    game_kwargs = {"players": int(3)}
  elif game_name == "leduc_poker_4p":
    game_name = "leduc_poker"
    game_kwargs = {"players": int(4)}

  elif game_name == "trade_comm_2p_2i":
    game_name = "trade_comm"
    game_kwargs = {"num_items": int(2)}
  elif game_name == "trade_comm_2p_3i":
    game_name = "trade_comm"
    game_kwargs = {"num_items": int(3)}
  elif game_name == "trade_comm_2p_4i":
    game_name = "trade_comm"
    game_kwargs = {"num_items": int(4)}
  elif game_name == "trade_comm_2p_5i":
    game_name = "trade_comm"
    game_kwargs = {"num_items": int(5)}

  elif game_name == "tiny_bridge_2p":
    game_name = "tiny_bridge_2p"
    game_kwargs = {}
  elif game_name == "tiny_bridge_4p":
    game_name = "tiny_bridge_4p"
    game_kwargs = {}  # Too big game.

  elif game_name == "sheriff_2p_1r":
    game_name = "sheriff"
    game_kwargs = {"num_rounds": int(1)}
  elif game_name == "sheriff_2p_2r":
    game_name = "sheriff"
    game_kwargs = {"num_rounds": int(2)}
  elif game_name == "sheriff_2p_3r":
    game_name = "sheriff"
    game_kwargs = {"num_rounds": int(3)}
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

  elif game_name == "goofspiel_2p_3c_total":
    game_name = "goofspiel"
    game_kwargs = {
        "players": int(2),
        "returns_type": "total_points",
        "num_cards": int(3)}
  elif game_name == "goofspiel_2p_4c_total":
    game_name = "goofspiel"
    game_kwargs = {
        "players": int(2),
        "returns_type": "total_points",
        "num_cards": int(4)}
  elif game_name == "goofspiel_2p_5c_total":
    game_name = "goofspiel"
    game_kwargs = {
        "players": int(2),
        "returns_type": "total_points",
        "num_cards": int(5)}

  else:
    raise ValueError("Unrecognised game: %s" % game_name)

  return pyspiel.load_game_as_turn_based(game_name, game_kwargs)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  game = get_game(FLAGS.game)
  jpsro.run_loop(
      game=game,
      game_name=FLAGS.game,
      seed=FLAGS.seed,
      iterations=FLAGS.iterations,
      policy_init=FLAGS.policy_init,
      update_players_strategy=FLAGS.update_players_strategy,
      target_equilibrium=FLAGS.target_equilibrium,
      br_selection=FLAGS.br_selection,
      train_meta_solver=FLAGS.train_meta_solver,
      eval_meta_solver=FLAGS.eval_meta_solver,
      ignore_repeats=FLAGS.ignore_repeats)


if __name__ == "__main__":
  app.run(main)
