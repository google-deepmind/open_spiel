from typing import Generic, TypeVar

import chex
import jax
import jax.numpy as jnp

Experience = TypeVar("Experience", bound=chex.ArrayTree)

"""Educational simplified copy of a flat buffer inspired by
https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/flat_buffer.py
"""


@chex.dataclass(frozen=True)
class ReplayBufferState(Generic[Experience]):
  experience: Experience
  capacity: chex.Numeric
  entry_index: chex.Array
  is_full: bool


def init(capacity: chex.Numeric, experience: Experience) -> ReplayBufferState:
  """Initialise a replay buffer.

  Args:
    capacity (chex.Numeric, int): max size of the buffer
    experience (Experience): initial value

  Returns:
    ReplayBufferState: state of the buffer
  """
  # Set experience value to be empty.
  experience = jax.tree.map(jnp.empty_like, experience)
  # Broadcast to [add_batch_size, ...]
  experience = jax.tree.map(
    lambda x: jnp.broadcast_to(x[jnp.newaxis, ...], (capacity, *x.shape)),
    experience,
  )
  return ReplayBufferState(
    capacity=capacity,
    experience=experience,
    entry_index=jnp.array(0),
    is_full=jnp.array(False, dtype=jnp.bool),
  )


def append(
  state: ReplayBufferState,
  experience: Experience,
) -> ReplayBufferState:
  """Potentially adds `experience` to the replay buffer.
  Args:
    state: `ReplayBufferState`, current state of the buffer
    experience: data to be added to the reservoir buffer.
  Returns:
    An updated `ReplayBufferState`
  """

  chex.assert_trees_all_equal_dtypes(experience, state.experience)

  index = state.entry_index % state.capacity

  def update_leaf(buffer_leaf: Experience, exp_leaf: Experience):
    return buffer_leaf.at[index].set(exp_leaf)

  new_experience = jax.tree.map(update_leaf, state.experience, experience)

  new_entry_index = state.entry_index + 1
  new_is_full = state.is_full | (new_entry_index >= state.capacity)

  return ReplayBufferState(
    capacity=state.capacity,
    experience=new_experience,
    entry_index=new_entry_index,
    is_full=new_is_full,
  )


def sample(rng: chex.PRNGKey, state: ReplayBufferState, num_samples: int) -> Experience:
  """Returns `num_samples` uniformly sampled from the buffer.
  Args:
    rng: `chex.PRNGKey`, a random state
    state: `ReplayBufferState`, a buffer state
    num_samples: `int`, number of samples to draw.
  Returns:
    An iterable over `num_samples` random elements of the buffer.
  Raises:
    AssertionError: If there are less than `num_samples` elements in the buffer
  """

  # When full, the max time index is max_length_time_axis otherwise it is current index.
  max_size = jnp.where(state.is_full, state.capacity, state.entry_index)
  indices = jax.random.randint(rng, shape=(num_samples,), minval=0, maxval=max_size)
  return jax.tree.map(lambda x: x[indices], state.experience)


class Buffer:
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size: int, force_cpu: bool = False, seed: int = 0) -> None:
    self.max_size = max_size
    self._total_seen = jnp.array(0)

    self._rng = jax.random.PRNGKey(seed)
    # jit-compling all the methods for cpu if forced otherwise automatically
    backend = "cpu" if force_cpu else jax.default_backend()

    self._init = jax.jit(init, backend=backend, static_argnames=("capacity",))
    self._append = jax.jit(append, backend=backend, donate_argnums=(0,))
    self._sample = jax.jit(sample, backend=backend, static_argnames=("num_samples",))

    self.buffer_state = None

  @property
  def data(self):
    if self.buffer_state is not None:
      return self.buffer_state.experience

  @property
  def total_seen(self) -> int:
    return self._total_seen.item()

  def __len__(self) -> int:
    if self.buffer_state is None:
      return 0
    return self.max_size if self.buffer_state.is_full else self._total_seen

  def __bool__(self) -> bool:
    return self.buffer_state is not None

  def append(self, val: Experience) -> None:
    if self.buffer_state is None:
      self.buffer_state = self._init(self.max_size, val)
    self.buffer_state = self._append(self.buffer_state, val)
    self._total_seen = self.buffer_state.entry_index

  def shuffle(self, key: int) -> chex.Array:
    return jax.random.permutation(jax.random.key(key), jnp.arange(len(self)))

  def sample(self, num_samples: int) -> Experience:
    self._rng, rng = jax.random.split(self._rng)
    batch = self._sample(rng, self.buffer_state, num_samples)
    return batch
