import jax.numpy as jnp
from absl.testing import absltest

from open_spiel.python.jax.nes import utils

"""Tests for open_spiel.python.jax.nes.utils.py"""


class UtilsTest(absltest.TestCase):
  # === Mask construction ===

  def test_strat_mask_derives_joint_mask(self):
    """Verify strat_mask derivation matches old joint_mask utility."""
    action_sizes = [2, 3]
    max_size = 4

    # Old way (from utils)
    joint_mask_old = utils.joint_mask(action_sizes, max_size)

    # New way (from strat_masks)
    strat_masks = utils.make_strat_masks(action_sizes, [max_size] * 2)
    joint_mask_new = utils.make_joint_mask_from_strat_masks(strat_masks)

    self.assertTrue(jnp.all(joint_mask_old == joint_mask_new))

  def test_make_strat_masks_basic(self):
    """Basic strat mask construction."""
    masks = utils.make_strat_masks([3, 2], [4, 4])
    self.assertEqual(masks[0].tolist(), [True, True, True, False])
    self.assertEqual(masks[1].tolist(), [True, True, False, False])

  def test_make_joint_mask_from_strat_masks(self):
    """Cartesian product of strat masks."""
    strat_masks = (
      jnp.array([True, True, False]),
      jnp.array([True, False, True]),
    )
    joint = utils.make_joint_mask_from_strat_masks(strat_masks)
    expected = jnp.array(
      [
        [True, False, True],
        [True, False, True],
        [False, False, False],
      ]
    )
    self.assertTrue(jnp.all(joint == expected))

  def test_joint_mask_roundtrip(self):
    """Marginalising joint mask recovers strat masks."""

    strat_masks = (
      jnp.array([True, True, False]),
      jnp.array([True, False, True, True]),
    )
    joint = utils.make_joint_mask_from_strat_masks(strat_masks)
    marg_0 = jnp.sum(joint, axis=1) > 0
    marg_1 = jnp.sum(joint, axis=0) > 0
    self.assertTrue(jnp.all(marg_0 == strat_masks[0]))
    self.assertTrue(jnp.all(marg_1 == strat_masks[1]))


if __name__ == "__main__":
  absltest.main()
