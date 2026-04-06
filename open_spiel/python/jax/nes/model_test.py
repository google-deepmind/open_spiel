from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import model
from open_spiel.python.jax.nes import utils
import pyspiel
import jax.numpy as jnp
import jax
import flax.nnx as nn
import itertools

"""Tests for open_spiel.python.jax.nes.model.py"""

class NESModelTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
        (model.Mode.CCE, model.Mode.CE),
        (2, 4, 6),
        (2, 3, 4),
      )
  )
  def test_dummy_model(self, mode, num_players, num_actions):
    # Testing for (C)CE
    action_sizes = tuple(num_actions for _ in range(num_players))
    batch = jnp.zeros((1, 4, num_players, *action_sizes))
    _model = model.NeuralEquilibriumModel(
      num_players, 
      action_sizes, 
      dual_channels=128, mode=mode, rngs=nn.Rngs(jax.random.key(0))
    )
    alpha = utils.batched_call(_model, batch)

    if mode == model.Mode.CCE:
      self.assertEqual(alpha.shape, (1, 1, num_players, action_sizes[0]))

    if mode == model.Mode.CE:
      self.assertEqual(alpha.shape, (1, 1, num_players, action_sizes[0], action_sizes[1]))

    # Should it be the case?
    self.assertTrue((alpha >=0).all())
    

if __name__ == "__main__":
  absltest.main()
