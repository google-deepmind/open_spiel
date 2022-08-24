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

"""Tests for open_spiel.python.algorithms.rpg."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.examples import kuhn_policy_gradient
import pyspiel
from open_spiel.python.pytorch import policy_gradient
from open_spiel.python.pytorch.losses import rl_losses

SEED = 24984617


class PolicyGradientTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(
      itertools.product(("rpg", "qpg", "rm", "a2c", "neurd"),
                        ("kuhn_poker", "leduc_poker")))
  def test_run_game(self, loss_str, game_name):
    env = rl_environment.Environment(game_name)
    env.seed(SEED)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        policy_gradient.PolicyGradient(  # pylint: disable=g-complex-comprehension
            player_id=player_id,
            info_state_size=info_state_size,
            num_actions=num_actions,
            loss_str=loss_str,
            hidden_layers_sizes=[32, 32],
            batch_size=16,
            entropy_cost=0.001,
            critic_learning_rate=0.01,
            pi_learning_rate=0.01,
            num_critic_before_pi=4) for player_id in [0, 1]
    ]

    for _ in range(2):
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = agents[current_player]
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])

      for agent in agents:
        agent.step(time_step)

  def test_neurd_kuhn(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        policy_gradient.PolicyGradient(  # pylint: disable=g-complex-comprehension
            player_id=player_id,
            info_state_size=info_state_size,
            num_actions=num_actions,
            loss_str="neurd",
            hidden_layers_sizes=[32],
            batch_size=16,
            entropy_cost=0.001,
            critic_learning_rate=0.01,
            pi_learning_rate=0.01,
            num_critic_before_pi=4) for player_id in [0, 1]
    ]
    expl_policies_avg = kuhn_policy_gradient.PolicyGradientPolicies(env, agents)

    for _ in range(100):
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = agents[current_player]
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])

      for agent in agents:
        agent.step(time_step)

    expl = exploitability.exploitability(env.game, expl_policies_avg)
    # Check the exploitability is less than the target upper bound.
    self.assertLess(expl, 0.7)

  def test_run_hanabi(self):
    # Hanabi is an optional game, so check we have it before running the test.
    game = "hanabi"
    if game not in pyspiel.registered_names():
      return

    num_players = 3
    env_configs = {
        "players": num_players,
        "max_life_tokens": 1,
        "colors": 2,
        "ranks": 3,
        "hand_size": 2,
        "max_information_tokens": 3,
        "discount": 0.99
    }
    env = rl_environment.Environment(game, **env_configs)
    env.seed(SEED)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        policy_gradient.PolicyGradient(  # pylint: disable=g-complex-comprehension
            player_id=player_id,
            info_state_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[8, 8],
            batch_size=16,
            entropy_cost=0.001,
            critic_learning_rate=0.001,
            pi_learning_rate=0.001,
            num_critic_before_pi=4) for player_id in range(num_players)
    ]

    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      agent_output = [agent.step(time_step) for agent in agents]
      time_step = env.step([agent_output[current_player].action])

    for agent in agents:
      agent.step(time_step)

  def test_loss_modes(self):
    loss_dict = {
        "qpg": rl_losses.BatchQPGLoss,
        "rpg": rl_losses.BatchRPGLoss,
        "rm": rl_losses.BatchRMLoss,
        "a2c": rl_losses.BatchA2CLoss,
        "neurd": rl_losses.BatchNeuRDLoss,
    }

    for loss_str, loss_class in loss_dict.items():
      agent_by_str = policy_gradient.PolicyGradient(
          player_id=0,
          info_state_size=32,
          num_actions=2,
          loss_str=loss_str,
          loss_class=None)
      agent_by_class = policy_gradient.PolicyGradient(
          player_id=0,
          info_state_size=32,
          num_actions=2,
          loss_str=None,
          loss_class=loss_class)

      self.assertEqual(agent_by_str._loss_class, agent_by_class._loss_class)


if __name__ == "__main__":
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  absltest.main()
