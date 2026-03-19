# Copyright 2026 DeepMind Technologies Limited
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

"""Python ESCHER example."""

# pylint: disable=logging-fstring-interpolation

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel
from open_spiel.python.pytorch import escher


FLAGS = flags.FLAGS
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")


def main(unused_argv):
  logging.info(f"Loading {FLAGS.game_name}")
  game = pyspiel.load_game(FLAGS.game_name)

  agent = escher.Agent(game, escher.Config())
  train_cfg = escher.TrainConfig(game)
  train_cfg.iterations = 100
  train_cfg.evaluation_interval = 20
  train_cfg.nashconv = True

  escher.train(train_cfg, agent)

  average_policy = policy.tabular_policy_from_callable(
      game, lambda s: _action_probabilities(agent, s)
  )
  pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
  conv = pyspiel.nash_conv(game, pyspiel_policy)
  logging.info(f"ESCHER in {FLAGS.game_name} - NashConv: {conv}")

  avg_policy_vals = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2
  )
  if FLAGS.game_name == "kuhn_poker":
    # We know EVs
    logging.info(
        f"Computed player 0 value: {avg_policy_vals[0]:.2f} (expected:"
        f" {-1/18:.2f})."
    )
    logging.info(
        f"Computed player 1 value: {avg_policy_vals[1]:.2f} (expected:"
        f" {1/18:.2f})."
    )
  else:
    logging.info(f"Computed player 0 value: {avg_policy_vals[0]:.2f}")
    logging.info(f"Computed player 1 value: {avg_policy_vals[1]:.2f}")


def _action_probabilities(agent, state):
  probs = agent.action_probabilities(state)

  prob_dict = {}
  for a, m in enumerate(state.legal_actions_mask()):
    if m == 1:
      prob_dict[a] = probs[a]
  return prob_dict


if __name__ == "__main__":
  app.run(main)
