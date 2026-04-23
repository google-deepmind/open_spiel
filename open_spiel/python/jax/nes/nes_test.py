import itertools

import flax.nnx as nn
import pyspiel
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils


class NESTest(parameterized.TestCase):
  # @parameterized.parameters(
  #   itertools.product(
  #     (networks.Mode.CCE, networks.Mode.CE),
  #     (2, 5),
  #     (2, 3, 4,),
  #   )
  # )
  # def test_dummy_cubic_model(self, mode, num_players, num_actions):
  #   # Testing for (C)CE
  #   batch_size = 10
  #   action_sizes = tuple(num_actions for _ in range(num_players))
  #   game_tensor = jnp.zeros((batch_size, 4, num_players, *action_sizes))
  #   mask = jnp.zeros((batch_size, *action_sizes))

  #   _model = networks.NeuralEquilibriumModel(
  #     num_players,
  #     dual_channels=32,
  #     mode=mode,
  #     rngs=nn.Rngs(jax.random.key(0)),
  #   )
  #   batch = nes.Data(
  #     **utils.dummy_nes_batch(batch_size, num_players, action_sizes, jax.random.key(0))
  #   )
  #   alpha = utils.batched_call(_model, game_tensor, mask)
  #   logits, sigma, epsilon, _, _ = jax.vmap(
  #     nes.recover_primals, in_axes=(0, 0, None, None, None, None, None)
  #   )(
  #       alpha, batch, 1, 1, 10, action_sizes, int(mode.value)
  #   )

  #   self.assertEqual(logits.shape, (batch_size, *action_sizes))
  #   self.assertEqual(sigma.shape, (batch_size, *action_sizes))
  #   self.assertEqual(epsilon.shape, (batch_size, num_players))

  # @parameterized.parameters(
  #   itertools.product(
  #     (networks.Mode.CCE, networks.Mode.CE),
  #     (2, 5),
  #     (2, 3, 4,),
  #   )
  # )
  # def test_gradients(
  #     self, mode, num_players, num_actions
  # ):
  #   """Verify that ∂L/∂α_p(a') = ε_p - Σ_a A_p(a', a) σ(a)."""
  #   # Testing for (C)CE
  #   batch_size = 10
  #   action_sizes = tuple(num_actions for _ in range(num_players))
  #   game_tensor = jnp.zeros((batch_size, 4, num_players, *action_sizes))
  #   mask = jnp.zeros((batch_size, *action_sizes))

  #   batch = nes.Data(
  #     **utils.dummy_nes_batch(batch_size, num_players, action_sizes, jax.random.key(0))
  #   )
  #   alpha = ...
  #   # Compute theoretical RHS: ε_p - Σ_a A_p(a', a) σ(a)
  #   _, aux = compute_loss(
  #       alpha, G, W, hat_sigma, hat_epsilon, mu, rho, epsilon_plus, mode
  #   )
  #   sigma = ...
  #   epsilon = ...

  #   # For CCE: theoretical gradient should be epsilon_p - expected_gain
  #   N = G.shape[0]
  #   theoretical = []

  #   for p in range(N):
  #     gain = compute_cce_deviation_gain(G[p], p)  # [Ap, *A]
  #     # Contract gain with sigma: sum over *A -> [Ap]
  #     expected_gain = jnp.tensordot(
  #         gain, sigma, axes=(tuple(range(1, gain.ndim)), tuple(range(sigma.ndim)))
  #     )
  #     grad_p_theory = epsilon[p] - expected_gain
  #     theoretical.append(grad_p_theory)

  #   theoretical = jnp.array(theoretical)

  @parameterized.parameters(networks.Mode.CCE, networks.Mode.CE)
  def test_it_runs(self, mode):
    game = pyspiel.load_game("matrix_rps")
    network_config = dict(
      dual_channels=32,
      payoff_channel_list=[128, 64, 128],
      dual_channel_list=[32, 32],
    )
    solver = nes.NESolver(game, mode, network_config, rho=10)
    solver.solve()


if __name__ == "__main__":
  absltest.main()
