
"""Tests for open_spiel.python.algorithms.alpha_zero.model."""

from absl.testing import absltest
import jax.numpy as jnp
import jax
import chex

from open_spiel.python.algorithms.alpha_zero.replay_buffer import Buffer, get_tree_shape_prefix

@chex.dataclass(frozen=True)
class FakeTransition:
  obs: chex.Array 
  action: chex.Array
  reward: chex.Array 



def get_fake_batch(fake_transition: chex.ArrayTree, batch_size) -> chex.ArrayTree:
    """Create a fake batch with differing values for each transition."""
    return jax.tree.map(
        lambda x: jnp.stack([x + i for i in range(batch_size)]), fake_transition
    )

def get_fake_transition():
  return FakeTransition(
    **{
      "obs": jnp.ones((5, 4), dtype=jnp.float32),
      "action": jnp.ones((2,), dtype=jnp.int32),
      "reward": jnp.zeros((), dtype=jnp.float16),
    }
  )

class FlatBufferTest(absltest.TestCase):

  def test_append(self):
    buffer = Buffer(10, force_cpu=True)
    batch = get_fake_transition()
    buffer.append(batch)

    self.assertEqual(len(buffer), 1)
    self.assertTrue(buffer) #can sample
    self.assertFalse(buffer.buffer_state.is_full.item())

  def test_extend(self):
    buffer = Buffer(10, force_cpu=True)
    batch = get_fake_transition()
    buffer.append(batch)

    for iter in range(3):
      big_batch = get_fake_batch(batch, 4)
      buffer.extend(big_batch)
      self.assertEqual(buffer.total_seen, 1 + 4 * (iter+1))
      self.assertEqual(buffer.buffer_state.current_index, (1 + 4 * (iter+1))%10)
    self.assertTrue(buffer.buffer_state.is_full.item())


     
  def test_sample(self):
    buffer = Buffer(5, force_cpu=True)
    batch = get_fake_transition()
    for _ in range(5):
      buffer.append(batch)

    bs1 = buffer.sample(2)
    self.assertEqual(get_tree_shape_prefix(bs1, 1)[0], 2)
    self.assertEqual(jax.tree.map(lambda x: x.shape[1:], bs1), jax.tree.map(lambda x: x.shape, batch))


if __name__ == "__main__":
  absltest.main()
