import itertools

import flax.nnx as nn
import pyspiel
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import samplers
from open_spiel.python.jax.nes import utils


class NESTest(parameterized.TestCase):
  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE),
      (2, 5),
      (2, 3, 4,),
    )
  )
  def test_dummy_cubic_model(self, mode, num_players, num_actions):
    # Testing for (C)CE
    batch_size = 10
    action_sizes = tuple(num_actions for _ in range(num_players))
    game_tensor = jnp.zeros((batch_size, 4, num_players, *action_sizes))
    mask = jnp.zeros((batch_size, *action_sizes))

    _model = networks.NeuralEquilibriumModel(
      num_players,
      dual_channels=32,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    batch = nes.Data(
      **utils.dummy_nes_batch(
        batch_size, num_players, action_sizes, jax.random.key(0)
      )
    )
    alpha = utils.batched_call(_model, game_tensor, mask)
    logits, sigma, epsilon, _, _ = jax.vmap(
      nes.recover_primals, in_axes=(0, 0, None, None, None, None, None)
    )(alpha, batch, 1, 1, 10, action_sizes, int(mode.value))

    self.assertEqual(logits.shape, (batch_size, *action_sizes))
    self.assertEqual(sigma.shape, (batch_size, *action_sizes))
    self.assertEqual(epsilon.shape, (batch_size, num_players))

  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE), #mode
      (2, 3), #num_players
      (2, 3, 4), #num_actions
      # (0, 0.1, 1.0),  # mu
      # (0, 0.8, 1.5),  # rho
      # (0, 0.1, 1.0),  # eps_plus
    )
  )
  def test_gradient_correctness(
    self, mode, num_players, num_actions,
  ):
    """∂L/∂α must match ε_p - (1/ρ)·E_σ[gain]."""
    mu, rho, eps_plus = 1, 1, 0.1

    rng = jax.random.PRNGKey(42)
    rng_G, rng_sigma, rng_eps, rng = jax.random.split(rng, 4)

    action_sizes = tuple(num_actions for _ in range(num_players))
    joint_A = utils.compute_joint_action_size(action_sizes)
    shape = (num_players,) + action_sizes

    # Random game data
    G = jax.random.uniform(rng_G, shape)
    sigma_hat = jax.random.dirichlet(rng_sigma, alpha=jnp.ones(joint_A))
    sigma_hat = sigma_hat.reshape(action_sizes)
    epsilon_hat = jax.random.uniform(
      rng_eps, (num_players,), minval=-0.5, maxval=0.5
    )
    W = jnp.sum(G, axis=0)
    mask = jnp.ones(action_sizes, dtype=jnp.bool)

    data = samplers.Data(
      reward=G,
      sigma_hat=sigma_hat,
      sigma_norm=None,  # dont' need that
      epsilon_hat=epsilon_hat,
      welfare=W,
      mask=mask,
    )

    def loss_fn(alpha):
      _, sigma, epsilon, sum_alphas, lse = nes.recover_primals(
        alpha, data, mu, rho, eps_plus, action_sizes, mode.value
      )
      return lse + eps_plus * jnp.sum(sum_alphas) - rho * jnp.sum(epsilon)

    # Random positive duals (network outputs softplus(·), so alpha >= 0)
    if mode == networks.Mode.CCE:
      alpha_shape = (1, num_players, num_actions)
    else:
      alpha_shape = (1, num_players, num_actions, num_actions)

    # dummy alphas, that there're under softplus
    alpha = jax.random.exponential(rng, alpha_shape)

    # Autodiff gradient
    grad_auto = jax.grad(loss_fn)(alpha)

    _, sigma, epsilon, _, _ = nes.recover_primals(
      alpha, data, mu, rho, eps_plus, action_sizes, mode.value
    )

    grad_theory = jnp.zeros_like(alpha)
    indices = jnp.indices(action_sizes)  # [N, A1, ..., AN]

    for p in range(num_players):
      G_p = G[p]

      if mode == networks.Mode.CCE:
        for dev in range(num_actions):
          # G_p(dev, a_{-p}) for every joint action a
          idx_dev = indices.at[p].set(dev)
          G_p_dev = G_p[tuple(idx_dev)]
          gain = G_p_dev - G_p
          expected_gain = jnp.sum(sigma * gain)
          grad_theory = grad_theory.at[0, p, dev].set(
            epsilon[p] - (1.0 / rho) * expected_gain
          )
      else:  # CE
        for dev in range(num_actions):
          idx_dev = indices.at[p].set(dev)
          G_p_dev = G_p[tuple(idx_dev)]
          gain = G_p_dev - G_p
          for rec in range(num_actions):
            mask_rec = indices[p] == rec
            expected_gain = jnp.sum(sigma * gain * mask_rec)
            grad_theory = grad_theory.at[0, p, dev, rec].set(
              epsilon[p] - (1.0 / rho) * expected_gain
            )

    self.assertTrue(
      jnp.allclose(grad_auto, grad_theory, atol=1e-4, rtol=1e-4),
      msg=f"Failed for mode={mode.name}, N={num_players}, A={num_actions}. "
      f"Max error: {jnp.max(jnp.abs(grad_auto - grad_theory))}",
    )

  @parameterized.parameters(networks.Mode.CCE, networks.Mode.CE)
  def test_it_runs(self, mode):
    game = pyspiel.load_game("matrix_rps")
    network_config = dict(
      dual_channels=32,
      payoff_channel_list=[128, 64, 128],
      dual_channel_list=[32, 32],
    )
    solver = nes.NESolver(game, mode, network_config, network_train_steps=10)
    solver.solve()


if __name__ == "__main__":
  absltest.main()
