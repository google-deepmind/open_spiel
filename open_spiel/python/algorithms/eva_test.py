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

"""Tests for open_spiel.python.algorithms.eva."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import eva

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()


class EVATest(parameterized.TestCase):

  @parameterized.parameters("tic_tac_toe", "kuhn_poker", "liars_dice")
  def test_run_games(self, game):
    env = rl_environment.Environment(game)
    num_players = env.num_players
    eva_agents = []
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]
    with tf.Session() as sess:
      for player in range(num_players):
        eva_agents.append(
            eva.EVAAgent(
                sess,
                env,
                player,
                state_size,
                num_actions,
                embedding_network_layers=(64, 32),
                embedding_size=12,
                learning_rate=1e-4,
                mixing_parameter=0.5,
                memory_capacity=int(1e6),
                discount_factor=1.0,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay_duration=int(1e6)))
      sess.run(tf.global_variables_initializer())
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = eva_agents[current_player]
        # 1.  Step the agent.
        # 2.  Step the Environment.
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])
      for agent in eva_agents:
        agent.step(time_step)


class QueryableFixedSizeRingBufferTest(tf.test.TestCase):

  def test_replay_buffer_add(self):
    replay_buffer = eva.QueryableFixedSizeRingBuffer(replay_buffer_capacity=10)
    self.assertEqual(len(replay_buffer), 0)
    replay_buffer.add("entry1")
    self.assertEqual(len(replay_buffer), 1)
    replay_buffer.add("entry2")
    self.assertEqual(len(replay_buffer), 2)

    self.assertIn("entry1", replay_buffer)
    self.assertIn("entry2", replay_buffer)

  def test_replay_buffer_max_capacity(self):
    replay_buffer = eva.QueryableFixedSizeRingBuffer(replay_buffer_capacity=2)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")
    replay_buffer.add("entry3")
    self.assertEqual(len(replay_buffer), 2)

    self.assertIn("entry2", replay_buffer)
    self.assertIn("entry3", replay_buffer)

  def test_replay_buffer_sample(self):
    replay_buffer = eva.QueryableFixedSizeRingBuffer(replay_buffer_capacity=3)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")
    replay_buffer.add("entry3")

    samples = replay_buffer.sample(3)

    self.assertIn("entry1", samples)
    self.assertIn("entry2", samples)
    self.assertIn("entry3", samples)

  # TODO(author6) Test knn query.


if __name__ == "__main__":
  tf.test.main()
