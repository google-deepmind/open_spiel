from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils
import jax.numpy as jnp
import jax
import flax.nnx as nn
import itertools


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
      dual_channels=128,
      mode=mode,
      rngs=nn.Rngs(jax.random.key(0)),
    )
    _model.eval()
    duals = utils.batched_call(_model, batch_padded, strat_masks_batched)

    if mode == networks.Mode.CCE:
      self.assertEqual(duals.shape, (1, num_players, max_size))

    if mode == networks.Mode.CE:
      self.assertEqual(duals.shape, (1, num_players, max_size, max_size))

    # Should it be the case?
    self.assertTrue((duals >= 0).all())

    # Verify ALL masked actions (padded + explicitly masked) are zero
    for p in range(num_players):
      padded_actions = ~strat_masks[p]
      if mode == networks.Mode.CCE:
        self.assertTrue(jnp.all(duals[0, p, padded_actions] == 0))
      if mode == networks.Mode.CE:
        # For CE, both dev and rec should be zero for padded actions
        self.assertTrue(jnp.all(duals[0, p, padded_actions, :] == 0))
        self.assertTrue(jnp.all(duals[0, p, :, padded_actions] == 0))

  # @absltest.sk("In works")
  def test_dual_masking(self):
    """Verify duals are correctly masked for padded actions."""
    N = 2
    max_A = 4
    
    # Player 0 has 3 actions, Player 1 has 4 actions
    strat_masks = [
        jnp.array([True, True, True, False]),  # Player 0: 3 valid
        jnp.array([True, True, True, True]),   # Player 1: 4 valid
    ]
    
    model = networks.NeuralEquilibriumModel(N, 8, networks.Mode.CCE, rngs=nn.Rngs(0))
    
    # CCE duals: [C=1, N=2, A=4]
    cce_duals = jnp.ones((1, N, max_A))
    masked = model._mask_duals(cce_duals, strat_masks)
    
    # Player 0's 4th action should be zero
    self.assertEqual(masked[0, 0, 3], 0, "Padded action not masked for CCE")
    self.assertEqual(masked[0, 0, 0], 1, "Valid action incorrectly masked for CCE")
    
    # CE duals: [C=1, N=2, A=4, A=4]
    ce_duals = jnp.ones((1, N, max_A, max_A))
    model_ce = networks.NeuralEquilibriumModel(N, 8, networks.Mode.CE, rngs=nn.Rngs(0))
    masked_ce = model_ce._mask_duals(ce_duals, strat_masks)
    
    # Player 0: row 3 and col 3 should be zero; diagonal should be zero
    self.assertEqual(masked_ce[0, 0, 3, 0], 0, "Padded row not masked for CE")
    self.assertEqual(masked_ce[0, 0, 0, 3], 0, "Padded col not masked for CE")
    self.assertEqual(masked_ce[0, 0, 0, 0], 0, "Diagonal not zeroed for CE")
    self.assertEqual(masked_ce[0, 0, 1, 2], 1, "Valid off-diagonal incorrectly masked")
    

if __name__ == "__main__":
  absltest.main()
