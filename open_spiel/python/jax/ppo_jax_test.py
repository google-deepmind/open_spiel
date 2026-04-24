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
"""Tests for open_spiel.python.jax.ppo and ppo_utils."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from open_spiel.python import rl_environment
from open_spiel.python.jax import ppo
from open_spiel.python.jax import ppo_utils


def _self_play_episodes(env, agent, num_episodes):
  """Run self-play episodes collecting training data."""
  num_players = env.num_players
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      if time_step.is_simultaneous_move():
        actions = []
        for pid in range(num_players):
          output = agent.step(time_step, player_id=pid)
          actions.append(output.action)
        time_step = env.step(actions)
      else:
        output = agent.step(time_step)
        time_step = env.step([output.action])
      agent.post_step(time_step)
    agent.step(time_step)


class ComputeGAETest(absltest.TestCase):
  """Tests for the vectorized jax.lax.scan-based GAE."""

  def test_single_step_terminal(self):
    """Single terminal step: advantage = reward - value."""
    rewards = jnp.array([1.0])
    values = jnp.array([0.5])
    dones = jnp.array([1.0])
    advantages, returns = ppo_utils.compute_gae(
        rewards, values, dones, gamma=1.0, gae_lambda=0.95)
    np.testing.assert_allclose(advantages, [0.5], atol=1e-6)
    np.testing.assert_allclose(returns, [1.0], atol=1e-6)

  def test_two_step_trajectory(self):
    """Two steps, terminal at t=1, gamma=1, lambda=1."""
    rewards = jnp.array([0.0, 1.0])
    values = jnp.array([0.4, 0.6])
    dones = jnp.array([0.0, 1.0])
    advantages, returns = ppo_utils.compute_gae(
        rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    # At t=1 (terminal): delta = 1.0 + 0 - 0.6 = 0.4, gae = 0.4
    # At t=0: delta = 0.0 + 1.0*0.6*1 - 0.4 = 0.2, gae = 0.2 + 1*1*0.4 = 0.6
    np.testing.assert_allclose(advantages, [0.6, 0.4], atol=1e-6)
    np.testing.assert_allclose(returns, [1.0, 1.0], atol=1e-6)

  def test_discounting(self):
    """Verify gamma discounting in a 2-step trajectory."""
    rewards = jnp.array([0.0, 1.0])
    values = jnp.array([0.0, 0.0])
    dones = jnp.array([0.0, 1.0])
    advantages, returns = ppo_utils.compute_gae(
        rewards, values, dones, gamma=0.5, gae_lambda=1.0)
    # t=1: delta = 1.0, gae = 1.0
    # t=0: delta = 0 + 0.5*0*1 - 0 = 0, gae = 0 + 0.5*1*1.0 = 0.5
    np.testing.assert_allclose(advantages, [0.5, 1.0], atol=1e-6)

  def test_all_terminal(self):
    """All steps terminal (independent transitions)."""
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([0.5, 1.0, 1.5])
    dones = jnp.array([1.0, 1.0, 1.0])
    advantages, _ = ppo_utils.compute_gae(
        rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(advantages, [0.5, 1.0, 1.5], atol=1e-6)


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
    self.assertIn("approx_kl", metrics)
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

  def test_run_matrix_pd(self):
    """Test on a simultaneous-move game (Prisoner's Dilemma)."""
    env = rl_environment.Environment("matrix_pd")
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

  def test_prng_reproducibility(self):
    """Two agents with the same seed produce the same first action."""
    env = rl_environment.Environment("kuhn_poker")
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent1 = ppo.PPO(
        player_id=0, info_state_size=info_state_size,
        num_actions=num_actions, hidden_sizes=[16], seed=123)
    agent2 = ppo.PPO(
        player_id=0, info_state_size=info_state_size,
        num_actions=num_actions, hidden_sizes=[16], seed=123)

    time_step = env.reset()
    out1 = agent1.step(time_step, is_evaluation=True)
    out2 = agent2.step(time_step, is_evaluation=True)

    self.assertEqual(out1.action, out2.action)
    np.testing.assert_array_equal(out1.probs, out2.probs)


if __name__ == "__main__":
  absltest.main()
