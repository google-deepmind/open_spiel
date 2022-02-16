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
"""Tests for open_spiel.python.jax.dqn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import boltzmann_dqn
import pyspiel

# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""


class DQNTest(absltest.TestCase):

  def test_train(self):
    game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
    env = rl_environment.Environment(game=game)
    agent = boltzmann_dqn.BoltzmannDQN(
        0,
        state_representation_size=game.information_state_tensor_shape()[0],
        num_actions=game.num_distinct_actions(),
        hidden_layers_sizes=[16],
        replay_buffer_capacity=100,
        batch_size=5,
        epsilon_start=0.02,
        epsilon_end=0.01,
        eta=5.0)
    total_reward = 0

    # Training. This will use the epsilon-greedy actions.
    for _ in range(100):
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step)
        time_step = env.step([agent_output.action])
        total_reward += time_step.rewards[0]
      agent.step(time_step)
    self.assertGreaterEqual(total_reward, -100)

    # Update the previous Q-network.
    agent.update_prev_q_network()

    # This will use the soft-max actions.
    policy = rl_agent_policy.RLAgentPolicy(game, agent, 0, False)
    probs = policy.action_probabilities(game.new_initial_state())
    self.assertAlmostEqual(probs[0], 0.54, places=2)


if __name__ == "__main__":
  absltest.main()
