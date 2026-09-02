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

"""Tests for open_spiel.python.pytorch.nash_pg."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nash_pg
import pyspiel

SEED = 24984617


def _run_episode(env, agents):
  time_step = env.reset()
  while not time_step.last():
    if time_step.is_simultaneous_move():
      actions = [agents[i].step(time_step).action for i in range(len(agents))]
    else:
      player = time_step.observations["current_player"]
      actions = [agents[player].step(time_step).action]
    time_step = env.step(actions)
  for agent in agents:
    agent.step(time_step)


def _agents(env, **kwargs):
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  return [
      nash_pg.NashPG(
          pid, info_state_size, num_actions, seed=SEED + pid, **kwargs
      )
      for pid in range(env.num_players)
  ]


class NashPGTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters("kuhn_poker", "leduc_poker")
  def test_runs_on_sequential_game(self, game_name):
    env = rl_environment.Environment(game_name)
    env.seed(SEED)
    agents = _agents(
        env, hidden_layers_sizes=[16], batch_size=8, num_minibatches=2
    )
    for _ in range(10):
      _run_episode(env, agents)
    self.assertIsNotNone(agents[0].loss)

  @parameterized.parameters(("matrix_rps", {}), ("goofspiel", {"num_cards": 3}))
  def test_runs_on_simultaneous_game(self, game_name, params):
    env = rl_environment.Environment(pyspiel.load_game(game_name, params))
    env.seed(SEED)
    agents = _agents(
        env, hidden_layers_sizes=[16], batch_size=8, num_minibatches=2
    )
    for _ in range(10):
      _run_episode(env, agents)
    self.assertIsNotNone(agents[0].loss)

  def test_evaluation_has_no_side_effects(self):
    env = rl_environment.Environment("kuhn_poker")
    agent = _agents(env)[0]
    out = agent.step(env.reset(), is_evaluation=True)
    self.assertIsNotNone(out.action)
    self.assertIsNone(agent.loss)
    self.assertEqual(agent.magnet_refreshes, 0)

  def test_magnet_refreshes_automatically(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    agents = _agents(
        env,
        hidden_layers_sizes=[16],
        batch_size=8,
        num_minibatches=2,
        magnet_update_period=1,
    )
    for _ in range(10):
      _run_episode(env, agents)
    self.assertGreater(agents[0].magnet_refreshes, 0)

  def test_manual_magnet_refresh(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    agents = _agents(
        env,
        hidden_layers_sizes=[16],
        batch_size=8,
        num_minibatches=2,
        auto_update_magnet=False,
    )
    for _ in range(10):
      _run_episode(env, agents)
    self.assertIsNotNone(agents[0].loss)
    self.assertEqual(agents[0].magnet_refreshes, 0)
    agents[0].update_magnet()
    self.assertEqual(agents[0].magnet_refreshes, 1)


if __name__ == "__main__":
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  absltest.main()
