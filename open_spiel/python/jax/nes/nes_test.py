import itertools

import flax.nnx as nn
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import samplers
from open_spiel.python.jax.nes import utils

"""Tests for open_spiel.python.jax.nes.nes.py"""


class NESTest(parameterized.TestCase):
  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE),
      (2, 3),
      (2, 3, 5),
    )
  )
  def test_dummy_cubic_model(self, mode, num_players, num_actions):
    # Testing for (C)CE
    action_sizes = tuple(num_actions for _ in range(num_players))
    batch_size = 10

    batch = samplers.dummy_nes_batch(
      batch_size, num_players, action_sizes, jax.random.key(0)
    )
    
    _model = networks.NeuralEquilibriumModel(
      num_players,
      # action_sizes,
      dual_channels=128,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    _model.eval()
    duals = utils.batched_call(
      _model,
      samplers.stack(jax.vmap(samplers.broadcast)(batch)),
      batch.strat_mask_per_player,
    )

    logits, strategy, epsilon, _, _ = jax.vmap(
      nes.recover_primals, in_axes=(0, 0, None, None, None, None)
    )(duals, batch, 1, 1, 10, int(mode.value))

    self.assertEqual(logits.shape, (batch_size, *action_sizes))
    self.assertEqual(strategy.shape, (batch_size, *action_sizes))
    self.assertEqual(epsilon.shape, (batch_size, num_players))

  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE),  # mode
      (2, 3),  # num_players
      (2,),  # num_actions
      (0.0, 1.0),
      (0.5, 5),
      (0.1, 0.5),
    )
  )
  def test_gradient_correctness(
    self,
    mode,
    num_players,
    num_actions,
    welfare_coeff,
    entropy_coeff,
    epsilon_max,
  ):
    """∂L/∂α must match ε_p - E_σ[gain]."""

    data_rng, duals_rng = jax.random.split(jax.random.PRNGKey(num_players), 2)

    action_sizes = tuple(num_actions for _ in range(num_players))
    data = jax.tree.map(
      lambda x: jnp.squeeze(x, 0),
      samplers.dummy_nes_batch(1, num_players, action_sizes, data_rng)
    )

    def loss_fn(duals):
      _, _, epsilon, sum_alphas, lse = nes.recover_primals(
        duals, data, welfare_coeff, entropy_coeff, epsilon_max, mode.value
      )
      return (
        lse
        + epsilon_max * jnp.sum(sum_alphas)
        - entropy_coeff * jnp.sum(epsilon)
      )

    if mode == networks.Mode.CCE:
      duals_shape = (num_players, num_actions)
    else:
      duals_shape = (num_players, num_actions, num_actions)

    # dummy alphas, that there're under softplus
    sample_duals = jax.random.exponential(duals_rng, duals_shape)

    # Autodiff gradient
    grad_auto = jax.grad(loss_fn)(sample_duals)

    _, strategy, epsilon, _, _ = nes.recover_primals(
      sample_duals, data, welfare_coeff, entropy_coeff, epsilon_max, mode.value
    )

    grad_theory = jnp.zeros_like(sample_duals)
    indices = jnp.indices(action_sizes)  # [A1, ..., AN]

    for p in range(num_players):
      payoff = data.payoffs[p]

      if mode == networks.Mode.CCE:
        for dev in range(num_actions):
          # G_p(dev, a_{-p}) for every joint action a
          idx_dev = indices.at[p].set(dev)
          payoff_at_dev = payoff[tuple(idx_dev)]
          gain = payoff_at_dev - payoff
          expected_gain = jnp.sum(strategy * gain)
          grad_theory = grad_theory.at[p, dev].set(epsilon[p] - expected_gain)
      else:  # CE
        for dev in range(num_actions):
          idx_dev = indices.at[p].set(dev)
          payoff_at_dev = payoff[tuple(idx_dev)]
          gain = payoff_at_dev - payoff
          for rec in range(num_actions):
            mask_rec = indices[p] == rec
            expected_gain = jnp.sum(strategy * gain * mask_rec)
            grad_theory = grad_theory.at[p, dev, rec].set(
              epsilon[p] - expected_gain
            )

    self.assertTrue(
      jnp.allclose(grad_auto, grad_theory, atol=1e-4, rtol=1e-4),
      msg=f"Failed for mode={mode.name}, N={num_players}, A={num_actions}. "
      f"Max error: {jnp.max(jnp.abs(grad_auto - grad_theory))}",
    )

  @parameterized.parameters(networks.Mode.CCE, networks.Mode.CE)
  def test_it_runs(self, mode):
    game = "matrix_rps"
    network_config = dict(
      dual_channels=32,
      payoff_channel_list=[128, 64, 128],
      dual_channel_list=[32, 32],
    )
    solver = nes.NESolver(game, mode, network_config, network_train_steps=10)
    solver.solve()


if __name__ == "__main__":
  absltest.main()
