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

"""Tests for open_spiel.python.jax.nash_pg."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import nash_pg
import pyspiel

SEED = 24984617


def _run_episode(env, agents):
  time_step = env.reset()
  while not time_step.last():
    if time_step.is_simultaneous_move():
      actions = [agent.step(time_step).action for agent in agents]
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


_SMALL = dict(hidden_layers_sizes=[16], batch_size=8, num_minibatches=2)


class HelpersTest(absltest.TestCase):

  def test_masked_kl_and_entropy_ignore_illegal_actions(self):
    mask = jnp.array([[True, False, True]])
    logits_p = jnp.array([[0.3, 100.0, -0.2]])
    logits_q = jnp.array([[-1.0, -100.0, 0.5]])

    def log_probs(logits):
      return jax.nn.log_softmax(
          jnp.where(mask, logits, nash_pg.ILLEGAL_ACTION_LOGITS_PENALTY)
      )

    lp, lq = log_probs(logits_p), log_probs(logits_q)
    p, q = np.exp(lp[0, [0, 2]]), np.exp(lq[0, [0, 2]])
    np.testing.assert_allclose(
        nash_pg.masked_kl(lp, lq, mask)[0],
        np.sum(p * np.log(p / q)),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        nash_pg.masked_entropy(lp, mask)[0], -np.sum(p * np.log(p)), rtol=1e-5
    )
    self.assertAlmostEqual(float(nash_pg.masked_kl(lp, lp, mask)[0]), 0.0)

  def test_gradients_are_finite_with_illegal_actions(self):
    mask = jnp.array([[True, False, True]])

    def objective(logits):
      lp = jax.nn.log_softmax(
          jnp.where(mask, logits, nash_pg.ILLEGAL_ACTION_LOGITS_PENALTY)
      )
      lq = jax.lax.stop_gradient(lp) - 0.1 * mask
      return (
          nash_pg.masked_kl(lp, lq, mask).sum()
          + nash_pg.masked_entropy(lp, mask).sum()
      )

    grads = jax.grad(objective)(jnp.array([[0.3, 1.0, -0.2]]))
    self.assertTrue(np.all(np.isfinite(grads)))
    self.assertEqual(float(grads[0, 1]), 0.0)

  def test_gae_matches_reference(self):
    rng = np.random.RandomState(0)
    rewards = rng.randn(7).astype(np.float32)
    values = rng.randn(7).astype(np.float32)
    dones = np.array([0, 0, 1, 0, 1, 0, 0], np.float32)
    bootstrap, gamma, lam = 0.7, 0.9, 0.8
    expected = np.zeros(7, np.float32)
    running = 0.0
    for t in reversed(range(7)):
      next_value = bootstrap if t == 6 else values[t + 1]
      delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
      running = delta + gamma * lam * (1 - dones[t]) * running
      expected[t] = running
    np.testing.assert_allclose(
        nash_pg.gae(
            jnp.asarray(rewards),
            jnp.asarray(values),
            jnp.asarray(dones),
            jnp.float32(bootstrap),
            gamma,
            lam,
        ),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


class NashPGTest(parameterized.TestCase):

  @parameterized.parameters("kuhn_poker", "leduc_poker")
  def test_runs_on_sequential_game(self, game_name):
    env = rl_environment.Environment(game_name)
    env.seed(SEED)
    agents = _agents(env, **_SMALL)
    for _ in range(10):
      _run_episode(env, agents)
    self.assertIsNotNone(agents[0].loss)
    self.assertTrue(np.all(np.isfinite(agents[0].loss)))

  @parameterized.parameters(("matrix_rps", {}), ("goofspiel", {"num_cards": 3}))
  def test_runs_on_simultaneous_game(self, game_name, params):
    env = rl_environment.Environment(pyspiel.load_game(game_name, params))
    env.seed(SEED)
    agents = _agents(env, **_SMALL)
    for _ in range(10):
      _run_episode(env, agents)
    self.assertIsNotNone(agents[0].loss)

  def test_policy_is_masked_distribution(self):
    env = rl_environment.Environment("leduc_poker")
    agent = _agents(env, **_SMALL)[0]
    time_step = env.reset()
    out = agent.step(time_step, is_evaluation=True)
    legal = time_step.observations["legal_actions"][0]
    self.assertLen(out.probs, env.action_spec()["num_actions"])
    self.assertAlmostEqual(np.sum(out.probs), 1.0, places=6)
    illegal = np.setdiff1d(np.arange(len(out.probs)), legal)
    np.testing.assert_array_equal(out.probs[illegal], 0.0)
    self.assertIn(out.action, legal)

  def test_evaluation_has_no_side_effects(self):
    env = rl_environment.Environment("kuhn_poker")
    agent = _agents(env, **_SMALL)[0]
    for _ in range(20):
      agent.step(env.reset(), is_evaluation=True)
    self.assertIsNone(agent.loss)
    self.assertEqual(agent.magnet_refreshes, 0)
    self.assertEmpty(agent._buffer)

  def test_magnet_refreshes_automatically(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    agents = _agents(env, magnet_update_period=1, **_SMALL)
    for _ in range(10):
      _run_episode(env, agents)
    self.assertGreater(agents[0].magnet_refreshes, 0)

  def test_manual_magnet_refresh_copies_params(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    agents = _agents(env, auto_update_magnet=False, **_SMALL)
    initial = jax.tree.leaves(agents[0]._magnet_params)
    for _ in range(10):
      _run_episode(env, agents)
    self.assertEqual(agents[0].magnet_refreshes, 0)
    self.assertGreater(agents[0].loss.magnet_kl, 0.0)
    # Updates must not touch the magnet.
    for a, b in zip(initial, jax.tree.leaves(agents[0]._magnet_params)):
      np.testing.assert_array_equal(a, b)
    agents[0].update_magnet()
    self.assertEqual(agents[0].magnet_refreshes, 1)
    network, _ = nn.merge(agents[0]._graphdef, agents[0]._state)
    current = jax.tree.leaves(nn.state(network, nn.Param))
    for a, b in zip(jax.tree.leaves(agents[0]._magnet_params), current):
      np.testing.assert_array_equal(a, b)

  def test_same_seed_is_deterministic(self):
    losses = []
    for _ in range(2):
      env = rl_environment.Environment("kuhn_poker")
      env.seed(SEED)
      agents = _agents(env, **_SMALL)
      for _ in range(10):
        _run_episode(env, agents)
      losses.append(agents[0].loss)
    self.assertEqual(losses[0], losses[1])

  def test_batch_size_one_skips_advantage_normalization(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    agents = _agents(
        env, hidden_layers_sizes=[16], batch_size=1, num_minibatches=1
    )
    for _ in range(3):
      _run_episode(env, agents)
    self.assertTrue(np.all(np.isfinite(agents[0].loss)))

  def test_rejects_uneven_minibatches(self):
    with self.assertRaises(ValueError):
      nash_pg.NashPG(0, 11, 2, batch_size=10, num_minibatches=4)

  def test_kuhn_exploitability_decreases(self):
    env = rl_environment.Environment("kuhn_poker")
    env.seed(SEED)
    agents = _agents(
        env,
        hidden_layers_sizes=[32],
        batch_size=128,
        num_minibatches=4,
        magnet_update_period=10,
    )
    policy = rl_agent_policy.JointRLAgentPolicy(
        env.game, {p: a for p, a in enumerate(agents)}, env.use_observation
    )
    initial = exploitability.exploitability(env.game, policy)
    for _ in range(3000):
      _run_episode(env, agents)
    final = exploitability.exploitability(env.game, policy)
    self.assertLess(final, 0.5 * initial)


if __name__ == "__main__":
  absltest.main()
