import chex
import jax
import jax.numpy as jnp

from typing import Callable, TypeVar, Optional, Generic

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
  current_index: chex.Array
  is_full: chex.Array

@chex.dataclass(frozen=True)
class BufferSample(Generic[Experience]):
  experience: Experience


def init(
    experience: Experience,
    add_batch_size: int,
) -> BufferState:
   
    # Set experience value to be empty.
    experience = jax.tree.map(jnp.empty_like, experience)
    # Broadcast to [add_batch_size, ...]
    experience = jax.tree.map(
        lambda x: jnp.broadcast_to(
            x[None, ...], (add_batch_size, *x.shape)
        ),
        experience,
    )

    return BufferState(
        experience=experience,
        is_full=jnp.array(False, dtype=bool),
        current_index=jnp.array(0),
    )


def add(
    state: BufferState,
    batch: Experience,
) -> BufferState:
    # Check that the batch has the correct shape and dtypes.
    chex.assert_tree_shape_prefix(batch, get_tree_shape_prefix(state.experience))
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    # Get the length of the time axis of the buffer state.
    max_length_time_axis = get_tree_shape_prefix(state.experience, n_axes=2)[1]
    # Check that the sequence length is less than or equal the maximum length of the time axis.
    chex.assert_axis_dimension_lteq(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )
    # Determine how many timesteps are in this batch.
    seq_len = get_tree_shape_prefix(batch, n_axes=2)[1]
    # Compute the time indices where the new data will be written.
    indices = (jnp.arange(seq_len) + state.current_index) % max_length_time_axis

    # Update the buffer state.
    new_experience = jax.tree.map(
        lambda exp_field, batch_field: exp_field.at[:, indices].set(batch_field),
        state.experience,
        batch,
    )

    new_current_index = state.current_index + seq_len
    new_is_full = state.is_full | (new_current_index >= max_length_time_axis)
    new_current_index = new_current_index % max_length_time_axis

    return state.replace(  # type: ignore
        experience=new_experience,
        current_index=new_current_index,
        is_full=new_is_full,
    )

def sample(
    state: BufferState,
    rng_key: chex.PRNGKey,
    batch_size: int
) -> BufferSample:
    # Get add_batch_size and the full size of the time axis.
    add_batch_size = get_tree_shape_prefix(state.experience, n_axes=1)

    rng_key, subkey_batch = jax.random.split(rng_key)

    sampled_batch_indices = jax.random.randint(
        subkey_batch, (batch_size,), 0, add_batch_size
    )

    batch_trajectory = jax.tree.map(
        lambda x: x[sampled_batch_indices[:, None]],
        state.experience,
    )

    return BufferSample(experience=batch_trajectory)



def can_sample(
    state: BufferState, min_length_time_axis: int
) -> chex.Array:
    """Indicates whether the buffer has been filled above the minimum length, such that it
    may be sampled from."""
    return state.is_full | (state.current_index >= min_length_time_axis)


TBufferState = TypeVar("TBufferState", bound=BufferState)
TBufferSample = TypeVar("TBufferSample", bound=BufferSample)

@chex.dataclass(frozen=True)
class FlatBuffer(Generic[Experience, TBufferState, TBufferSample]):


    init: Callable[[Experience], BufferState]
    add: Callable[
        [BufferState, Experience],
        BufferState,
    ]
    sample: Callable[
        [BufferState, chex.PRNGKey],
        BufferSample,
    ]
    can_sample: Callable[[BufferState], chex.Array]

def make_flat_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool = False,
    add_batch_size: Optional[int] = None,
) -> FlatBuffer:
   return FlatBuffer(...)