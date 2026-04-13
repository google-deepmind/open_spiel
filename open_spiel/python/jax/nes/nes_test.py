import itertools

import flax.nnx as nn
import pyspiel
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import nes, networks, utils


class NESTest(parameterized.TestCase):
  # @parameterized.parameters(
  #   itertools.product(
  #     (networks.Mode.CCE, networks.Mode.CE),
  #     (2, 4, 6),
  #     (2, 3, 4,),
  #   )
  # )
  # def test_dummy_cubic_model(self, mode, num_players, num_actions):
  #   # Testing for (C)CE
  #   batch_size = 10
  #   action_sizes = tuple(num_actions for _ in range(num_players))
  #   game_tensor = jnp.zeros((batch_size, 4, num_players, *action_sizes))
  #   _model = networks.NeuralEquilibriumModel(
  #     num_players,
  #     # action_sizes,
  #     dual_channels=32,
  #     mode=mode,
  #     rngs=nn.Rngs(jax.random.key(0)),
  #   )
  #   batch = nes.Data(
  #     **utils.dummy_nes_batch(batch_size, num_players, action_sizes, jax.random.key(0))
  #   )
  #   alpha = utils.batched_call(_model, game_tensor)
  #   logits, sigma, epsilon, _, _ = nes.recover_primals(
  #       alpha, batch, 1, 1, 10, int(mode.value)
  #   )

  #   self.assertEqual(logits.shape, (batch_size, *action_sizes))
  #   self.assertEqual(sigma.shape, (batch_size, *action_sizes))
  #   self.assertEqual(epsilon.shape, (batch_size, num_players))

  @parameterized.parameters(
    networks.Mode.CCE, networks.Mode.CE
  )
  def test_it_runs(self, mode):
    game = pyspiel.load_game("matrix_rps")
    network_config = dict(dual_channels=32, payoff_channel_list=[128, 64, 128], dual_channel_list=[32, 32])
    solver = nes.NESSolver(game, mode, network_config, rho=10)
    solver.solve()


if __name__ == "__main__":
  absltest.main()
