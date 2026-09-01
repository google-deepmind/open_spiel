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

"""Tests for the PyTorch NashPG agent."""

from absl.testing import absltest
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg
import pyspiel


SIMPLE_TWO_PLAYER_EFG = """
  EFG 2 R "Simple two-player problem" { "Player 1" "Player 2" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { 1.0 -1.0 }
    t "R" 2 "Outcome R" { -1.0 1.0 }
"""


class NashPGTest(absltest.TestCase):

  def _make_agent(self, game, player_id=0, **kwargs):
    return nash_pg.NashPG(
        player_id=player_id,
        state_representation_size=game.information_state_tensor_shape()[0],
        num_actions=game.num_distinct_actions(),
        hidden_layers_sizes=[16],
        batch_size=2,
        ppo_epochs=1,
        learn_every=2,
        seed=17 + player_id,
        **kwargs,
    )

  def test_runs_and_updates(self):
    game = pyspiel.load_efg_game(SIMPLE_TWO_PLAYER_EFG)
    env = rl_environment.Environment(game=game)
    agent = self._make_agent(game)

    for _ in range(6):
      time_step = env.reset()
      while not time_step.last():
        output = agent.step(time_step)
        time_step = env.step([output.action])
      agent.step(time_step)

    self.assertEqual(agent.num_updates, 3)
    self.assertIsNotNone(agent.loss)
    self.assertIn("magnet_kl", agent.metrics)

  def test_action_probabilities_are_legal(self):
    game = pyspiel.load_efg_game(SIMPLE_TWO_PLAYER_EFG)
    env = rl_environment.Environment(game=game)
    agent = self._make_agent(game)
    time_step = env.reset()
    output = agent.step(time_step, is_evaluation=True)
    legal_actions = time_step.observations["legal_actions"][0]

    self.assertIn(output.action, legal_actions)
    np.testing.assert_allclose(output.probs.sum(), 1.0)
    illegal_actions = [
        action
        for action in range(game.num_distinct_actions())
        if action not in legal_actions
    ]
    np.testing.assert_allclose(output.probs[illegal_actions], 0.0)

  def test_magnet_can_be_manually_refreshed(self):
    game = pyspiel.load_efg_game(SIMPLE_TWO_PLAYER_EFG)
    agent = self._make_agent(
        game, auto_update_magnet=False, magnet_update_period=10
    )
    agent.policy_network.zero_grad()
    agent.update_magnet()

    for policy_parameter, magnet_parameter in zip(
        agent.policy_network.parameters(), agent.magnet_network.parameters()
    ):
      np.testing.assert_allclose(
          policy_parameter.detach().numpy(), magnet_parameter.detach().numpy()
      )


if __name__ == "__main__":
  absltest.main()
