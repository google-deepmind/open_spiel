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
"""Tests for open_spiel.python.jax.ppo."""

from absl.testing import absltest

from open_spiel.python import rl_environment
from open_spiel.python.jax import ppo


def _self_play_episodes(env, agent, num_episodes):
  """Run self-play episodes collecting training data."""
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      output = agent.step(time_step)
      time_step = env.step([output.action])
      agent.post_step(time_step)
    agent.step(time_step)


class PPOTest(absltest.TestCase):

  def test_run_kuhn_poker(self):
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = ppo.PPO(
        player_id=0,
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_sizes=[16, 16],
        update_epochs=2,
        num_minibatches=2,
        seed=0,
    )

    _self_play_episodes(env, agent, num_episodes=20)
    self.assertGreater(agent.buffer_size, 0)

    metrics = agent.learn()
    self.assertIn("policy_loss", metrics)
    self.assertIn("value_loss", metrics)
    self.assertIn("entropy", metrics)
    self.assertEqual(agent.buffer_size, 0)

  def test_run_leduc_poker(self):
    env = rl_environment.Environment("leduc_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = ppo.PPO(
        player_id=0,
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_sizes=[16, 16],
        update_epochs=2,
        num_minibatches=2,
        seed=0,
    )

    _self_play_episodes(env, agent, num_episodes=20)
    metrics = agent.learn()
    self.assertIn("policy_loss", metrics)

  def test_self_play_both_players(self):
    """Verify the same agent acts for both players in a turn-based game."""
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = ppo.PPO(
        player_id=0,
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_sizes=[16, 16],
        seed=0,
    )

    players_seen = set()
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.current_player()
      players_seen.add(current_player)
      output = agent.step(time_step)
      time_step = env.step([output.action])
      agent.post_step(time_step)
    agent.step(time_step)

    self.assertGreaterEqual(len(players_seen), 2)
    self.assertGreater(agent.buffer_size, 0)

  def test_evaluation_mode(self):
    """Verify evaluation mode does not collect data."""
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = ppo.PPO(
        player_id=0,
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_sizes=[16, 16],
        seed=0,
    )

    time_step = env.reset()
    while not time_step.last():
      output = agent.step(time_step, is_evaluation=True)
      time_step = env.step([output.action])
    agent.step(time_step, is_evaluation=True)

    self.assertEqual(agent.buffer_size, 0)

  def test_multiple_learn_cycles(self):
    """Verify agent can collect and learn multiple times."""
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = ppo.PPO(
        player_id=0,
        info_state_size=info_state_size,
        num_actions=num_actions,
        hidden_sizes=[16, 16],
        update_epochs=2,
        num_minibatches=2,
        seed=0,
    )

    for _ in range(3):
      _self_play_episodes(env, agent, num_episodes=10)
      metrics = agent.learn()
      self.assertIn("policy_loss", metrics)
      self.assertEqual(agent.buffer_size, 0)


if __name__ == "__main__":
  absltest.main()
