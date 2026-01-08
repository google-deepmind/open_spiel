# Copyright 2022 DeepMind Technologies Limited
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
"""Tests for open_spiel.python.algorithms.ppo."""

import random
from absl.testing import absltest
import numpy as np
import torch

from open_spiel.python import rl_environment
import pyspiel
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.vector_env import SyncVectorEnv

# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""
SEED = 24261711


class PPOTest(absltest.TestCase):

  def test_simple_game(self):
    game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
    env = rl_environment.Environment(game=game)
    envs = SyncVectorEnv([env])
    agent_fn = PPOAgent
    anneal_lr = True

    info_state_shape = tuple(
        np.array(env.observation_spec()["info_state"]).flatten())

    total_timesteps = 1000
    steps_per_batch = 8
    batch_size = int(len(envs) * steps_per_batch)
    num_updates = total_timesteps // batch_size
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0,
        num_envs=1,
        agent_fn=agent_fn,
    )

    time_step = envs.reset()
    for update in range(num_updates):
      for _ in range(steps_per_batch):
        agent_output = agent.step(time_step)
        time_step, reward, done, _ = envs.step(
            agent_output, reset_if_done=True)
        agent.post_step(reward, done)

      if anneal_lr:
        agent.anneal_learning_rate(update, num_updates)

      agent.learn(time_step)

    total_eval_reward = 0
    n_total_evaluations = 1000
    n_evaluations = 0
    time_step = envs.reset()
    while n_evaluations < n_total_evaluations:
      agent_output = agent.step(time_step, is_evaluation=True)
      time_step, reward, done, _ = envs.step(
          agent_output, reset_if_done=True)
      total_eval_reward += reward[0][0]
      n_evaluations += sum(done)
    self.assertGreaterEqual(total_eval_reward, 900)


if __name__ == "__main__":
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  absltest.main()
