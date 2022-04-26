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

"""Tests for open_spiel.python.algorithms.dqn."""

import random
from absl.testing import absltest
import numpy as np
import torch

from open_spiel.python import rl_environment
import pyspiel
from open_spiel.python.pytorch import dqn

# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""
SEED = 24261711


class DQNTest(absltest.TestCase):

  def test_simple_game(self):
    game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
    env = rl_environment.Environment(game=game)
    agent = dqn.DQN(
        0,
        state_representation_size=game.information_state_tensor_shape()[0],
        num_actions=game.num_distinct_actions(),
        min_buffer_size_to_learn=10,
        hidden_layers_sizes=[16],
        replay_buffer_capacity=1000,
        update_target_network_every=100,
        learn_every=10,
        discount_factor=0.99,
        epsilon_decay_duration=1000,
        batch_size=32,
        epsilon_start=0.5,
        epsilon_end=0.01)
    total_eval_reward = 0
    for _ in range(1000):
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step)
        time_step = env.step([agent_output.action])
      agent.step(time_step)
    for _ in range(1000):
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
        total_eval_reward += time_step.rewards[0]
    self.assertGreaterEqual(total_eval_reward, 250)

  def test_run_tic_tac_toe(self):
    env = rl_environment.Environment("tic_tac_toe")
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        dqn.DQN(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[16],
            replay_buffer_capacity=10,
            batch_size=5) for player_id in [0, 1]
    ]
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      current_agent = agents[current_player]
      agent_output = current_agent.step(time_step)
      time_step = env.step([agent_output.action])

    for agent in agents:
      agent.step(time_step)

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
        "discount": 0.
    }
    env = rl_environment.Environment(game, **env_configs)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        dqn.DQN(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[16],
            replay_buffer_capacity=10,
            batch_size=5) for player_id in range(num_players)
    ]
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      agent_output = [agent.step(time_step) for agent in agents]
      time_step = env.step([agent_output[current_player].action])

    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
