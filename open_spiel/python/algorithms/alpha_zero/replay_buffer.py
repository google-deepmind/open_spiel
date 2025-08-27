import chex
import jax
import jax.numpy as jnp
from functools import partial

from typing import Callable, TypeVar, Generic, Any

Experience = TypeVar("Experience", bound=chex.ArrayTree)

"""Educational simplified copy of a flat buffer inspired by
https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/flat_buffer.py
"""

def get_tree_shape_prefix(tree: chex.ArrayTree, n_axes: int = 1) -> chex.Shape:
  """Get the shape of the leading axes (up to n_axes) of a pytree. This assumes all
  leaves have a common leading axes size (e.g. a common batch size)."""
  flat_tree, tree_def = jax.tree_util.tree_flatten(tree)
  leaf = flat_tree[0]
  leading_axis_shape = leaf.shape[0:n_axes]
  chex.assert_tree_shape_prefix(tree, leading_axis_shape)
  return leading_axis_shape


@chex.dataclass(frozen=True)
class BufferState(Generic[Experience]):
  experience: Experience
  is_full: chex.Array
  total_seen: chex.Array
  current_index: chex.Array


@chex.dataclass(frozen=True)
class BufferSample(Generic[Experience]):
  experience: Experience


def init(
    experience: Experience,
    max_size: int,
) -> BufferState:
  """Instanting a buffer state from the mock transition
  """

  # Set experience value to be empty.
  experience = jax.tree.map(jnp.empty_like, experience)
  # Broadcast to [add_batch_size, ...]
  experience = jax.tree.map(
    lambda x: jnp.broadcast_to(
        x[None, ...], (max_size, *x.shape)
    ),
    experience,
  )

  return BufferState(
    experience=experience,
    is_full=jnp.array(False, dtype=bool),
    total_seen=jnp.array(0),
    current_index=jnp.array(0),
  )


def extend(
  state: BufferState,
  batch: Experience,
) -> BufferState:
    
  """Adding a batch of experience to the buffer state.
  """

  max_size = get_tree_shape_prefix(state.experience)[0]
  batch_size = get_tree_shape_prefix(batch)[0]

  indices = (jnp.arange(batch_size) + state.current_index) % max_size
  new_experience = jax.tree.map(
    lambda exp_field, batch_field: exp_field.at[indices].set(batch_field),
    state.experience,
    batch,
  )

  new_total_seen = state.total_seen + batch_size
  new_is_full = state.is_full | (new_total_seen >= max_size)
  new_current_index = new_total_seen % max_size

  return state.replace(  # type: ignore
    experience=new_experience,
    total_seen=new_total_seen,
    is_full=new_is_full,
    current_index=new_current_index
 )


def sample(
  state: BufferState,
  rng_key: chex.PRNGKey,
  count: int
) -> BufferSample:
    
  """Sampling count examples from the buffer
  """

  # Get add_batch_size and the full size of the time axis.
  max_size = get_tree_shape_prefix(state.experience)[0]
  # When full, the max  index is the right itertator otherwise it is current index.
  max_size = jnp.where(state.is_full, max_size, state.current_index)
  # When full, the oldest valid data is current_index otherwise it is zero.
  head = jnp.where(state.is_full, state.current_index, 0)

  # Given no wrap around, the last valid starting index is:
  max_start = max_size - count
  # If max_start is negative then we cannot sample yet.
  num_valid_items = jnp.where(max_start >= 0, max_start + 1, 0)
  # (num_valid_items is the number of candidate subsequences—each starting at a
  # multiple of period that lie entirely in the valid region.)

  # Split the RNG key for sampling items and batch indices.
  rng_key, subkey_batch = jax.random.split(rng_key)
  # Sample an item index in [0, num_valid_items). (This is the index in the candidate list.)
  sampled_item_idx = jax.random.randint(
      subkey_batch, (count,), 0, num_valid_items
  )

  # Compute the logical start time index: ls = (sampled_item_idx * period).
  logical_start = sampled_item_idx
  # Map logical time to physical index in the buffer given there is wrap around.
  physical_start = (head + logical_start) % max_size

  # Create indices for the full subsequence.
  traj_time_indices = (
      physical_start + jnp.arange(count)
  ) % max_size

  batch_trajectory = jax.tree.map(lambda x: x[traj_time_indices], state.experience)

  return BufferSample(experience=batch_trajectory)


def can_sample(
  state: BufferState
) -> chex.Array:
  """Indicates whether the buffer has been filled above the minimum length, such that it
  may be sampled from."""
  return state.is_full | state.current_index


TBufferState = TypeVar("TBufferState", bound=BufferState)
TBufferSample = TypeVar("TBufferSample", bound=BufferSample)

@chex.dataclass(frozen=True)
class FlatBuffer(Generic[Experience, TBufferState, TBufferSample]):

  init: Callable[[Experience], BufferState]
  extend: Callable[
    [BufferState, Experience],
    BufferState,
  ]

  sample: Callable[
    [BufferState, chex.PRNGKey],
    BufferSample,
  ]
  can_sample: Callable[[BufferState], chex.Array]

def make_flat_buffer(max_size: int) -> FlatBuffer:
   return FlatBuffer(
      init=partial(init, max_size=max_size),
      extend=extend,
      sample=sample,
      can_sample=can_sample
   )


class Buffer:
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size: int, force_cpu: bool = False, seed: int = 0) -> None:
    self.max_size = max_size
    self.buffer = make_flat_buffer(max_size=max_size)
    self.total_seen = 0
    self._rng = jax.random.PRNGKey(seed)
    #jit-compling all the methods for cpu if forced otherwise automatically
    backend = "cpu" if force_cpu else None
    self.buffer = self.buffer.replace(
        init=jax.jit(self.buffer.init, backend=backend),
        extend=jax.jit(self.buffer.extend, backend=backend),
        sample=jax.jit(self.buffer.sample, backend=backend, static_argnames="count"),
        can_sample=jax.jit(self.buffer.can_sample, backend=backend),
    )
    self.buffer_state = None

  @property
  def data(self):
    if self.buffer_state is not None:
        return self.buffer_state.experience
  
  def __len__(self) -> int:
    if self.buffer_state is None:
       return 0
    
    max_size = get_tree_shape_prefix(self.buffer_state.experience)[0]
    return jnp.where(self.buffer_state.is_full, max_size, self.buffer_state.current_index).item()

  def __bool__(self) -> bool:
    if self.buffer_state is None:
        return False
    
    return bool(self.buffer.can_sample(self.buffer_state))

  def append(self, val: Any) -> None:
    
    if self.buffer_state is None:
       self.buffer_state = self.buffer.init(val)

    batched_val = jax.tree.map(lambda x: jnp.array(x)[None, ...], val)

    self.buffer_state = self.buffer.extend(self.buffer_state, batched_val)

  def extend(self, val: Any) -> None:
    if self.buffer_state is None:
       #the buffer has to be initialised with an unbatched transition
       unbatched_val = jax.tree.map(lambda x: x[0], val)
       self.buffer_state = self.buffer.init(unbatched_val)

    self.buffer_state = self.buffer.extend(self.buffer_state, val)
    self.total_seen = self.buffer_state.total_seen

  def sample(self, count: int) -> Any:
    if self.buffer.can_sample(self.buffer_state):
       self._rng, _ = jax.random.split(self._rng)
       return self.buffer.sample(self.buffer_state, self._rng, count).experience