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
      (2, 5),
      (2, 3, 4),
    )
  )
  def test_dummy_model(self, mode, num_players, num_actions):
    # Testing for (C)CE
    action_sizes = tuple(num_actions for _ in range(num_players))
    batch = jnp.zeros((1, 4, num_players, *action_sizes))
    mask = jnp.ones((1, *action_sizes), dtype=jnp.bool)
    _model = networks.NeuralEquilibriumModel(
      num_players,
      # action_sizes,
      dual_channels=128,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    alpha = utils.batched_call(_model, batch, mask)

    if mode == networks.Mode.CCE:
      self.assertEqual(alpha.shape, (1, num_players, action_sizes[0]))

    if mode == networks.Mode.CE:
      self.assertEqual(
        alpha.shape, (1, num_players, action_sizes[0], action_sizes[1])
      )

    # Should it be the case?
    self.assertTrue((alpha >= 0).all())

  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE),
      (2, 5),
    )
  )
  def test_dummy_noncubic_model(self, mode, num_players):
    # Testing for (C)CE
    action_sizes = [2, 3, 1, 2, 5, 1][:num_players]
    max_size = max(action_sizes)
    batch = jnp.zeros((1, 4, num_players, *action_sizes))
    mask = None
    _model = networks.NeuralEquilibriumModel(
      num_players,
      # action_sizes,
      dual_channels=128,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    alpha = utils.batched_call(_model, batch, mask)

    if mode == networks.Mode.CCE:
      self.assertEqual(alpha.shape, (1, num_players, max_size))

    if mode == networks.Mode.CE:
      self.assertEqual(alpha.shape, (1, num_players, max_size, max_size))

    # Should it be the case?
    self.assertTrue((alpha >= 0).all())

  @parameterized.parameters(
    itertools.product(
      (networks.Mode.CCE, networks.Mode.CE),
      (2,),
    )
  )
  def test_masking(self, mode, num_players):
    max_size = 8
    action_sizes = [2, 3, 1, 2, 5, 1][:num_players]
    batch = jnp.zeros((num_players, *action_sizes))

    batch_padded = utils.pad_game_tensor(batch, max_size, 0.0)
    batch_padded = jnp.broadcast_to(
      batch_padded[jnp.newaxis, jnp.newaxis], (1, 4, *batch_padded.shape)
    )

    mask = utils.joint_mask(action_sizes, max_size)[jnp.newaxis]
    self.assertEqual(mask.sum(), math.prod(action_sizes))

    _model = networks.NeuralEquilibriumModel(
      num_players,
      # action_sizes,
      dual_channels=128,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )

    alpha = utils.batched_call(_model, batch_padded, mask)

    if mode == networks.Mode.CCE:
      self.assertEqual(alpha.shape, (1, num_players, max_size))

    if mode == networks.Mode.CE:
      self.assertEqual(alpha.shape, (1, num_players, max_size, max_size))

    self.assertTrue((alpha >= 0).all())


if __name__ == "__main__":
  absltest.main()
