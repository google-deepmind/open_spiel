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
  """A state for the buffer

    Important arguments: 
      write_index: Index where the next batch of experience data will be added to.
      read_index: Index where the next batch of experience data will be sampled from
  """
  experience: Experience
  is_full: chex.Array
  total_seen: chex.Array
  read_index: chex.Array
  write_index: chex.Array


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
    read_index=jnp.array(0),
    write_index=jnp.array(0),
  )


def can_add(
    state: BufferState[Experience],
    add_batch_size: int,
) -> chex.Array:
  """Check if the queue state can be written to.
  Fully taken from 
  https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/trajectory_queue.py
  """

  max_size = get_tree_shape_prefix(state.experience)[0]

  new_write_index = state.write_index + add_batch_size
  read_index_lt_eq_to_write = state.read_index <= state.write_index
  read_index_greater_than_write = state.read_index > state.write_index
  max_length_reached = new_write_index >= max_size

  write_index_circular_overtake = (
      read_index_lt_eq_to_write
      & max_length_reached
      & ((new_write_index % max_size) > state.read_index)
  )

  read_index_overtaken = read_index_greater_than_write & (
      new_write_index > state.read_index
  )

  will_overwrite = write_index_circular_overtake | read_index_overtaken

  do_not_write = state.is_full | will_overwrite

  return ~do_not_write

def extend(
  state: BufferState,
  batch: Experience,
) -> BufferState:
    
  """Adding a batch of experience to the buffer state.
  """

  max_size = get_tree_shape_prefix(state.experience)[0]
  batch_size = get_tree_shape_prefix(batch)[0]

  indices = (jnp.arange(batch_size) + state.write_index) % max_size
  new_experience = jax.tree.map(
    lambda exp_field, batch_field: exp_field.at[indices].set(batch_field),
    state.experience,
    batch,
  )

  new_total_seen = state.total_seen + batch_size
  new_is_full = state.is_full | (new_total_seen >= max_size)
  new_write_index = new_total_seen % max_size

  return state.replace(  # type: ignore
    experience=new_experience,
    total_seen=new_total_seen,
    is_full=new_is_full,
    write_index=new_write_index
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
 
  # The queue is circular, so we can loop back to the start
  traj_indices = (jnp.arange(count) + state.read_index) % max_size

  # Update the queue state.
  new_read_index = (state.read_index + count) % max_size
  state = state.replace(  # type: ignore
    read_index=new_read_index,
    is_full=jnp.array(False),
  )
  # Sample
  batch_trajectory = jax.tree.map(lambda x: x[traj_indices], state.experience)

  return state, BufferSample(experience=batch_trajectory), traj_indices


def can_sample(
  state: BufferState,
  batch_size: int,
) -> chex.Array:
  """Indicates whether the buffer has been filled above the minimum length, such that it
  may be sampled from.
  Fully taken from:
  https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/trajectory_queue.py
  """

  # Get the buffer's size
  max_size = get_tree_shape_prefix(state.experience)[0]

  # Calculate all the conditional expressions
  new_read_index = state.read_index + batch_size
  read_index_less_than_write = state.read_index < state.write_index
  read_index_greater_than_write = state.read_index > state.write_index
  max_length_reached = new_read_index >= max_size

  # If the read index is less than the write index and the new read index is still less than the
  # write index, then we can sample.
  can_sample_read_less = read_index_less_than_write & (
      new_read_index <= state.write_index
  )
  # If the read index is greater than the write index and the new read index has
  # wrapped around thus when taking modulo the length of the buffer, it is less
  # than the write index, then we can sample.
  can_sample_read_greater = (
      read_index_greater_than_write
      & max_length_reached
      & ((new_read_index % max_size) <= state.write_index)
  )

  # if the queue is full
  can_sample_if_full = (
      state.is_full & ~max_length_reached & (new_read_index > state.write_index)
  )
  can_sample_if_full_circular = (
      state.is_full
      & max_length_reached
      & ((new_read_index % max_size) <= state.write_index)
  )

  # We combine the previous conditions to get the final condition. Note that we
  # apply can_sample_read_greater only if the max length has been reached.
  # Otherwise, if the read index is greater than the write index and the
  # new read index is still greater than the write index without wrapping
  # around the buffer, then we can sample.
  can_sample = (
      can_sample_read_less
      | can_sample_read_greater
      | (read_index_greater_than_write & ~max_length_reached)
      | can_sample_if_full
      | can_sample_if_full_circular
  )

  return can_sample


TBufferState = TypeVar("TBufferState", bound=BufferState)
TBufferSample = TypeVar("TBufferSample", bound=BufferSample)

@chex.dataclass(frozen=True)
class FlatBuffer(Generic[Experience, TBufferState, TBufferSample]):

  init: Callable[[Experience], BufferState]
  # Adding function
  extend: Callable[
    [BufferState, Experience],
    BufferState,
  ]

  sample: Callable[
    [BufferState, chex.PRNGKey],
    tuple[BufferState, BufferSample, chex.Array],
  ]
  # Checkers
  can_sample: Callable[[BufferState, int], chex.Array]
  can_add: Callable[[BufferState, int], chex.Array]


def make_flat_buffer(max_size: int) -> FlatBuffer:
   return FlatBuffer(
      init=partial(init, max_size=max_size),
      extend=extend,
      sample=sample,
      can_sample=can_sample,
      can_add=can_add
   )


class Buffer:
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size: int, force_cpu: bool = False, seed: int = 0, sequential: bool = True) -> None:
    self.max_size = max_size
    self.buffer = make_flat_buffer(max_size=max_size)
    self.total_seen = 0
    self.sequential = sequential

    #TODO: debug `can_*` checks.
    
    self._rng = jax.random.PRNGKey(seed)
    # jit-compling all the methods for cpu if forced otherwise automatically
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
    # Queue length: add unless it's full, but keep unchanged further
    return jnp.where(self.buffer_state.is_full, max_size, self.buffer_state.write_index).item()

  def __bool__(self) -> bool:
    if self.buffer_state is None:
        return False
    
    return bool(self.buffer.can_sample(self.buffer_state, 1))

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

    #TODO: add `can_add` check
    batch_size = get_tree_shape_prefix(val)[0] # pylint: disable=possibly-unused-variable  # noqa: F841
    # if self.buffer.can_add(self.buffer_state, batch_size):
    self.buffer_state = self.buffer.extend(self.buffer_state, val)
    self.total_seen = self.buffer_state.total_seen

  def sample(self, count: int) -> Any:
    if self.buffer.can_sample(self.buffer_state, count):
      self._rng, rng = jax.random.split(self._rng)

      # indices are returned for debug purposes only
      self.buffer_state, batch, indices = self.buffer.sample( # pylint: disable=possibly-unused-variable
        self.buffer_state, self._rng, count) 

      # To keep experiences at random, let them be shuffled
      return jax.lax.cond(
        self.sequential,
        lambda x, rng: x,
        lambda x, rng: jax.tree.map(lambda y: jax.random.permutation(rng, y, axis=0), x),
        batch.experience, rng
      )