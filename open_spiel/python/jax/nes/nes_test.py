import itertools

import flax.nnx as nn
import jax
import jax.numpy as jnp
import pyspiel
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import model, nes, utils


class NESTest(parameterized.TestCase):
  @parameterized.parameters(
    itertools.product(
      (model.Mode.CCE, model.Mode.CE),
      (2, 4, 6),
      (2, 3, 4),
    )
  )
  def test_dummy_model(self, mode, num_players, num_actions):
    # Testing for (C)CE
    batch_size = 10
    action_sizes = tuple(num_actions for _ in range(num_players))
    game_tensor = jnp.zeros((batch_size, 4, num_players, *action_sizes))
    _model = model.NeuralEquilibriumModel(
      num_players,
      action_sizes,
      dual_channels=32,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    batch = nes.Data(
      **utils.dummy_nes_batch(batch_size, num_players, action_sizes)
    )
    alpha = utils.batched_call(_model, game_tensor)
    logits, sigma, epsilon, _, _ = jax.vmap(
      nes.recover_primals, in_axes=(0, 0, None, None, None, None)
    )(alpha, batch, 1, 1, 10, int(mode.value))

    self.assertEqual(logits.shape, (batch_size, *action_sizes))
    self.assertEqual(sigma.shape, (batch_size, *action_sizes))
    self.assertEqual(epsilon.shape, (batch_size, num_players))






if __name__ == "__main__":
  absltest.main()
