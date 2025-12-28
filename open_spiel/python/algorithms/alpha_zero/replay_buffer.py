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


def append(
  state: BufferState,
  batch: Experience,
) -> BufferState:
    
  """Adding a batch of experience to the buffer state.
  """

  max_size = get_tree_shape_prefix(state.experience)[0]

  new_experience = jax.tree.map(
    lambda exp_field, batch_field: exp_field.at[state.write_index].set(batch_field),
    state.experience,
    batch,
  )

  new_total_seen = state.total_seen + 1
  
  return state.replace(  # type: ignore
    experience=new_experience,
    total_seen=new_total_seen,
    is_full=state.is_full | (new_total_seen >= max_size),
    write_index=(new_total_seen % max_size)
 )


def sample_random(
  state: BufferState,
  rng_key: chex.PRNGKey,
  count: int,
) -> tuple[BufferState, Any, chex.Array]:
  max_size = get_tree_shape_prefix(state.experience)[0]
  max_size = jnp.where(state.is_full, max_size, state.write_index)
  
  traj_indices = jax.random.randint(rng_key, shape=(count,), minval=0, maxval=max_size)
  batch_trajectory = jax.tree.map(lambda x: x[traj_indices], state.experience)

  return state, batch_trajectory, traj_indices


TBufferState = TypeVar("TBufferState", bound=BufferState)

@chex.dataclass(frozen=True)
class FlatBuffer(Generic[Experience, TBufferState]):

  init: Callable[[Experience], BufferState]
  # Adding function
  append: Callable[
    [BufferState, Experience],
    BufferState,
  ]

  sample: Callable[
    [BufferState, chex.PRNGKey],
    tuple[BufferState, Any, chex.Array],
  ]

def make_flat_buffer(max_size: int) -> FlatBuffer:
   return FlatBuffer(
      init=partial(init, max_size=max_size),
      append=append,
      sample=sample_random,
   )


class Buffer:
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size: int, force_cpu: bool = False, seed: int = 0) -> None:
    self.max_size = max_size
    self.buffer = make_flat_buffer(max_size=max_size)
    self.total_seen = jnp.array(0)

    self._rng = jax.random.PRNGKey(seed)
    # jit-compling all the methods for cpu if forced otherwise automatically
    backend = "cpu" if force_cpu else jax.default_backend()
    self.buffer = self.buffer.replace(
      init=jax.jit(self.buffer.init, backend=backend),
      append=jax.jit(self.buffer.append, backend=backend, donate_argnums=(0,)),
      sample=jax.jit(self.buffer.sample, backend=backend, static_argnames=("count",)),
    )
    self.buffer_state = None

  @property
  def data(self):
    if self.buffer_state is not None:
        return self.buffer_state.experience
  
  def __len__(self) -> int:
    if self.buffer_state is None:
       return 0
    
    # Queue length: add unless it's full, but keep unchanged further
    return jnp.where(self.buffer_state.is_full, self.max_size, self.buffer_state.write_index).item()

  def __bool__(self) -> bool:
    return self.buffer_state is not None

  def append(self, val: Any) -> None:
    
    if self.buffer_state is None:
      self.buffer_state = self.buffer.init(val)

    # batched_val = jax.tree.map(lambda x: x[jnp.newaxis, ...], val)
    self.buffer_state = self.buffer.append(self.buffer_state, val)
    self.total_seen = self.buffer_state.total_seen

  def shuffle(self, key: int) -> chex.Array:
    return jax.random.permutation(jax.random.key(key), jnp.arange(len(self)))

  def sample(self, count: int) -> Any:
    self._rng, rng = jax.random.split(self._rng) 
    indices = jax.random.choice(rng, jnp.arange(len(self)), replace=False, shape=(count,))
    batch = jax.tree.map(lambda x: x[indices], self.buffer_state.experience)
    # indices are returned for debug purposes only
    # _, batch, indices = self.buffer.sample( # pylint: disable=possibly-unused-variable
    #   self.buffer_state, rng, count) 
    
    return batch