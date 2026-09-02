from absl.testing import absltest, parameterized


from open_spiel.python.jax.nes import generators
from open_spiel.python.jax.nes import utils
import jax.numpy as jnp
import jax
import flax.nnx as nn
import itertools
import math

"""Tests for open_spiel.python.jax.nes.generators.py"""


class GeneratorsTest(parameterized.TestCase):

  @parameterized.parameters((2, 3, 4))
  def test_inverse_equilibrium_generator(self, num_players,):
    generator = generators.InverseEquilibriumGenerator(
      num_players=num_players,
      channel_list=[32, 32]
    )

    num_actions = (3,) * num_players
    context = generator(
      target_sigma=...,
      noise=...,
      mask=...
    )

    self.assertEqual(context.shape, (num_players,) + num_actions)

    num_actions = tuple([2, 5, 4, 3, 2][:num_players])
    context = generator(
      target_sigma=...,
      noise=...,
      mask=...
    )

    self.assertEqual(context.shape, (num_players,) + num_actions)

  @parameterized.parameters(itertools.product((2, 3, 4), (1, 2, 3)))
  def test_contract_design_generator(self, num_players, o_channels):
    generator = generators.ContractDesignGenerator(
      num_players=num_players,
      po2po_channels=32,
      o2o_channels=3,
      num_po2po=4,
      num_o2o=3
    )

    num_actions = (3,) * num_players
    context = generator(
      costs = ...,
      transition = ...,
      principal_payoff = ...,
      mask = ... 
    )

    self.assertEqual(context.shape, (num_players, o_channels))
    self.assertEqual(jnp.all(context >= 0))


    num_actions = tuple([2, 5, 4, 3, 2][:num_players])
    context = generator(
      costs = ...,
      transition = ...,
      principal_payoff = ...,
      mask = ... 
    )

    self.assertEqual(context.shape, (num_players, o_channels))
    self.assertEqual(jnp.all(context >= 0))

  
if __name__ == "__main__":
  absltest.main()