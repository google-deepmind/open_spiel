from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils
import jax.numpy as jnp
import jax
import flax.nnx as nn
import itertools
import math

"""Tests for open_spiel.python.jax.nes.networks.py"""


class NESModelTest(parameterized.TestCase):
  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE),
      (2, 3),
      (2, 3, 5),
    )
  )
  def test_dummy_model(self, mode, num_players, num_actions):
    # Testing for (C)CE
    action_sizes = tuple(num_actions for _ in range(num_players))
    action_sizes = [2, 3, 1, 2, 5, 1][:num_players]
    max_size = 8

    tensor = jnp.zeros((num_players, *action_sizes))

    padded_tensor = utils.pad_game_tensor(tensor, max_size, 0.0)
    batch_padded = jnp.broadcast_to(
      padded_tensor[jnp.newaxis, jnp.newaxis], (1, 4, *padded_tensor.shape)
    )

    # Build strat_mask_per_player
    strat_masks = utils.make_strat_masks(action_sizes, [max_size] * num_players)
    strat_masks_batched = tuple(m[jnp.newaxis] for m in strat_masks)

    _model = networks.NeuralEquilibriumModel(
      num_players,
      # action_sizes,
      dual_channels=128,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    _model.eval()
    alpha = utils.batched_call(_model, batch_padded, strat_masks_batched)

    if mode == networks.Mode.CCE:
      self.assertEqual(alpha.shape, (1, num_players, max_size))

    if mode == networks.Mode.CE:
      self.assertEqual(
        alpha.shape, (1, num_players, max_size, max_size)
      )

    # Should it be the case?
    self.assertTrue((alpha >= 0).all())

    # Verify ALL masked actions (padded + explicitly masked) are zero
    for p in range(num_players):
      padded_actions = ~strat_masks[p]
      if mode == networks.Mode.CCE:
        self.assertTrue(jnp.all(alpha[0, p, padded_actions] == 0))
      if mode == networks.Mode.CE:
        # For CE, both dev and rec should be zero for padded actions
        self.assertTrue(jnp.all(alpha[0, p, padded_actions, :] == 0))
        self.assertTrue(jnp.all(alpha[0, p, :, padded_actions] == 0))


if __name__ == "__main__":
  absltest.main()
