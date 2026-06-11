import enum
import functools
import math

import optax
import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def named_sharding(
  mesh: jax.sharding.Mesh, *names: str | None
) -> NamedSharding:
  return NamedSharding(mesh, P(*names))


@chex.dataclass(unsafe_hash=True)
class MeshRules:
  mlp: str | None = None
  data: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


mesh_rules = MeshRules(
  mlp="model",
  data="data",
)

SMALL_NUMBER = jnp.finfo(jnp.float32).eps


def lr_schedule(base_learning_rate: float = 1e-4) -> optax.Schedule:
  """Joint learning schedule from the NES paper.
  (iteration, factor) pairs of
  - (1e6, 1.0)
  - (4e6, 0.6)
  - (7e6, 0.3)
  - (10e6, 0.1)
  - (100e6, 0.06)
  - (inf, 0.03)
  where each "iteration" is the *duration* of that segment.
  """

  # Cumulative boundaries from segment durations
  segment_durations = [1_000_000, 4_000_000, 7_000_000, 10_000_000, 100_000_000]
  factors = [1.0, 0.6, 0.3, 0.1, 0.06, 0.03]

  # Convert durations to cumulative boundaries for optax
  boundaries = []
  cumulative = 0
  for duration in segment_durations[:-1]:
    cumulative += duration
    boundaries.append(cumulative)
  # boundaries = [1_000_000, 5_000_000, 12_000_000, 22_000_000]

  # Build constant schedules for each segment
  schedules = [
    optax.constant_schedule(base_learning_rate * factor) 
    for factor in factors
  ]

  return optax.join_schedules(
    schedules=schedules,
    boundaries=boundaries,
  )


@nn.vmap(in_axes=(None, 0, 0), out_axes=0, axis_name="batch")
def batched_call(
  model: nn.Module, x: chex.Array, mask: chex.Array
) -> chex.Array:
  """Batched model call."""
  return model(x, mask)


def mask_diagonal(x: chex.Array) -> chex.Array:
  """CE-related utility to compute f_hat."""
  diag = jnp.arange(x.shape[-1])
  x = x.at[..., diag, diag].set(0.0)
  return x

def compute_joint_action_size(action_shape: chex.Shape) -> int:
  """|A| = product of all players' action sizes"""
  return math.prod(action_shape)


@functools.partial(jax.jit, static_argnames=("m", "axes", "where"))
def compute_L_m_norm(
  tensor: chex.Array, m: int, axes: tuple, where: chex.Array = None
) -> chex.Array:
  if m == 2:
    # L2 norm: √(Σ x²)
    return jnp.sqrt(
      jnp.sum(jnp.square(tensor), axis=axes, keepdims=True, where=where)
    )
  elif m == 1:
    # L1 norm: Σ |x|
    return jnp.sum(jnp.abs(tensor), axis=axes, keepdims=True, where=where)
  # General L_m norm
  return jnp.power(
    jnp.sum(jnp.abs(tensor) ** m, axis=axes, keepdims=True, where=where),
    1.0 / m,
  )


def make_joint_mask_from_strat_masks(
  strat_mask_per_player: list[chex.Array],
) -> chex.Array:
  """Cartesian product: joint is valid iff all player actions are valid.

  Args:
      strat_mask_per_player: Tuple of boolean arrays [Ap].

  Returns:
      joint_mask: Boolean array [A1, ..., AN].
  """
  N = len(strat_mask_per_player)
  shape = tuple(int(m.shape[0]) for m in strat_mask_per_player)
  joint_mask = jnp.ones(shape, dtype=bool)

  for p, mask in enumerate(strat_mask_per_player):
    # Expand mask to broadcast over all non-p axes
    expand_axes = tuple(i for i in range(N) if i != p)
    joint_mask &= jnp.expand_dims(mask, axis=expand_axes)

  return joint_mask


def make_strat_masks(
  num_strats_per_player: list[int],
  max_strats_per_player: list[int],
) -> tuple[chex.Array, ...]:
  """Builds strat_mask for each player given actual vs. max actions.

  Args:
      num_strats_per_player: Actual number of valid actions per player.
      max_strats_per_player: Padded size per player (max across batch).

  Returns:
      strat_mask_per_player: Tuple of boolean arrays [max_Ap].
  """
  return tuple(
    jnp.arange(max_ap) < num_ap
    for num_ap, max_ap in zip(num_strats_per_player, max_strats_per_player)
  )


def make_masks_for_batch(
  batch_num_strats: list[list[int]],  # [B, N] list of lists
) -> tuple[tuple[chex.Array, ...], chex.Array]:
  """Builds strat and joint masks for a batch of games.

  Args:
      batch_num_strats: B games, each with N player's action counts.
          Example: [[3, 3], [2, 4], [3, 2]] for B=3, N=2.

  Returns:
      strat_masks_batch: Tuple of [B, max_Ap] boolean arrays per player.
      joint_mask_batch: [B, A1, ..., AN] boolean array.
  """
  B = len(batch_num_strats)
  N = len(batch_num_strats[0])

  # Find max actions per player across batch
  max_strats = tuple(
    max(game[p] for game in batch_num_strats) for p in range(N)
  )

  # Build per-player strat masks: [B, max_Ap]
  strat_masks_per_player = []
  for p in range(N):
    masks_p = jnp.stack(
      [jnp.arange(max_strats[p]) < game[p] for game in batch_num_strats]
    )  # shape [B, max_Ap]
    strat_masks_per_player.append(masks_p)

  # Derive joint masks: [B, A1, ..., AN]
  # Vectorize over batch dimension
  make_joint = jax.vmap(make_joint_mask_from_strat_masks, in_axes=0)
  joint_mask_batch = make_joint(
    tuple(strat_masks_per_player)
  )  # [B, A1, ..., AN]

  return tuple(strat_masks_per_player), joint_mask_batch


def joint_mask(action_sizes: chex.Shape, max_actions: int) -> chex.Array:
  """
  Create boolean mask of shape [max_A, ..., max_A] where True indicates
  all players' actions are within their valid ranges.
  """
  N = len(action_sizes)
  # indices[p, A1, A2, ..., AN] = action index for player p
  indices = jnp.indices((max_actions,) * N)
  sizes = jnp.array(action_sizes).reshape((N,) + (1,) * N)

  # valid_per_player[p, ...] = True where player p's coordinate is valid
  valid_per_player = indices < sizes

  return jnp.all(valid_per_player, axis=0)


@functools.partial(jax.jit, static_argnames=("max_actions", "pad_value"))
def pad_game_tensor(
  payoffs: chex.Array,
  max_actions: int,
  pad_value: float,
) -> chex.Array:
  """
  Pad payoff tensor [N, A1, ..., AN] to [N, max_actions, ..., max_actions].

  Args:
      payoffs: Payoff tensor with natural (possibly non-cubic) shape
      max_actions: Target size for every action dimension
      pad_value: Value to pad with

  Returns:
      Padded tensor of shape [N, max_actions] * N
  """
  action_sizes = payoffs.shape[1:]
  # Build pad_width: [(0,0)] for player dim + [(0, max-A_i)] for each action dim
  pad_width = [(0, 0)] + [(0, max_actions - a_i) for a_i in action_sizes]
  return jnp.pad(payoffs, pad_width, mode="constant", constant_values=pad_value)


def rand_choice_strat_mask_per_player(
  key: chex.PRNGKey,
  num_strats: int,
  count: int,
) -> chex.Array:
  """Returns random choice strat mask.

  Args:
    key: Integer.
    num_strats: Integer with value |A_p|.
    count: Integer.

  Returns:
    strat_mask: Array with shape [|A_p|].
  """

  array = jnp.arange(num_strats) < count
  return jax.random.permutation(key, array)


def rand_choice_strat_mask(
  key: chex.PRNGKey,
  num_strats_per_player: list[int],
  counts: chex.Array,
) -> tuple[chex.Array, ...]:
  """Returns random strat mask per player.

  Args:
    key: Integer.
    num_strats_per_player: Sequence with values [|A_1|,...,|A_N|].
    counts: Integer array with shape [N].

  Returns:
    strat_mask_per_player: Tuple of arrays with shape [[|A_p|]_p=1:N].
  """

  num_players = len(num_strats_per_player)
  keys = jax.random.split(key, num_players)
  return jax.tree_util.tree_map(
    rand_choice_strat_mask_per_player,
    tuple(keys),
    tuple(num_strats_per_player),
    tuple(counts),
  )


def _include_arrays(
  reduced_none: chex.Array,
  reduced_all: chex.Array,
  reduced_other: chex.Array,
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
) -> tuple[chex.Array, ...]:
  """Select which reductions to include."""
  arrays = []
  if include_identity:
    arrays.append(reduced_none)
  if include_all:
    arrays.append(reduced_all)
  if include_other:
    arrays.append(reduced_other)
  return tuple(arrays)


def reduce_sum(
  array: chex.Array,
  axis: tuple[int, ...],
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  broadcast: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Sum reduction with identity/all/other."""
  reduced_none = array
  reduced_all = jnp.sum(array, axis=axis, keepdims=True, where=where)
  reduced_other = reduced_all - reduced_none
  arrays = (reduced_none, reduced_all, reduced_other)
  includes = (include_identity, include_all, include_other)
  arrays = [
    jnp.broadcast_to(a, reduced_none.shape) if broadcast else a
    for a in _include_arrays(*arrays, *includes)
  ]
  return jnp.concatenate(arrays, axis=0)

def reduce_mean(
  array: chex.Array,
  axis: tuple[int, ...],
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  broadcast: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Mean reduction with NaN-safe masking."""
  reduced_none = array

  reduced_all = jnp.sum(array, axis=axis, keepdims=True, where=where)
  reduced_other = reduced_all - reduced_none

  if where is not None:
    broadcast_where = jnp.broadcast_to(where, array.shape)
    all_divisor = jnp.sum(broadcast_where, axis=axis, keepdims=True)
    all_divisor = jnp.where(all_divisor <= 2, 2.0, all_divisor)
  else:
    all_divisor = math.prod([array.shape[a] for a in axis])

  # Multiplication replaces division to be cleaner about the comp. graph
  reduced_all = reduced_all / all_divisor
  reduced_other = reduced_other / (all_divisor - 1)

  arrays = (reduced_none, reduced_all, reduced_other)
  includes = (include_identity, include_all, include_other)
  arrays = [
    jnp.broadcast_to(a, reduced_none.shape) if broadcast else a
    for a in _include_arrays(*arrays, *includes)
  ]
  return jnp.concatenate(arrays, axis=0)


def reduce_meanvar(
  array: chex.Array,
  axis: tuple[int, ...],
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  broadcast: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Meanvar reduction (variance-preserving scaling)."""
  reduced_none = array
  reduced_all = jnp.sum(array, axis=axis, keepdims=True, where=where)
  reduced_other = reduced_all - reduced_none

  if where is not None:
    all_divisor = jnp.sum(
      jnp.broadcast_to(where, array.shape), axis=axis, keepdims=True
    )
    all_divisor = jnp.where(all_divisor <= 2, 2.0, all_divisor)
  else:
    all_divisor = math.prod([array.shape[a] for a in axis])

  # Multiplication replaces division to be cleaner about the comp. graph
  reduced_all /= jnp.sqrt(all_divisor)
  reduced_other /= jnp.sqrt(all_divisor - 1)

  arrays = (reduced_none, reduced_all, reduced_other)
  includes = (include_identity, include_all, include_other)
  arrays = [
    jnp.broadcast_to(a, reduced_none.shape) if broadcast else a
    for a in _include_arrays(*arrays, *includes)
  ]
  return jnp.concatenate(arrays, axis=0)


def reduce_max(
  array: chex.Array,
  axis: tuple[int, ...],
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  broadcast: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Max reduction with top-k fallback for 'other'."""
  if where is not None:
    raise NotImplementedError("max reduction with where mask")

  reduced_none = array
  reduced_all = jnp.max(array, axis=axis, keepdims=True)

  ndim = array.ndim
  other_axis = tuple(a for a in range(ndim) if a not in axis)
  transpose_axes = (*other_axis, *axis)
  other_shape = tuple(array.shape[a] for a in other_axis)
  other_shape += (1,) * (ndim - len(other_shape))
  untranspose_axes = tuple(transpose_axes.index(a) for a in range(ndim))
  transposed = jnp.transpose(array, transpose_axes)
  reshaped = jnp.reshape(transposed, [math.prod(other_shape), -1])
  # Needed because "max over others"
  # cannot be derived from "max over all" minus identity
  top_2, _ = jax.vmap(functools.partial(jax.lax.top_k, k=2))(reshaped)
  top_2nd = top_2[:, 1]
  top_2nd = jnp.reshape(top_2nd, shape=other_shape)
  top_2nd = jnp.transpose(top_2nd, untranspose_axes)
  reduced_other = jnp.where(reduced_none == reduced_all, top_2nd, reduced_all)

  arrays = (reduced_none, reduced_all, reduced_other)
  includes = (include_identity, include_all, include_other)
  arrays = [
    jnp.broadcast_to(a, reduced_none.shape) if broadcast else a
    for a in _include_arrays(*arrays, *includes)
  ]
  return jnp.concatenate(arrays, axis=0)


def reduce_min(
  array: chex.Array,
  axis: tuple[int, ...],
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  broadcast: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Min reduction via negated max."""
  return -reduce_max(
    -array, axis, include_identity, include_all, include_other, broadcast, where
  )


class Reduction(enum.Enum):
  NONE = 0
  SUM = 1
  MEAN = 2
  MEANVAR = 3
  MAX = 4
  MIN = 5


def reduce(
  array: chex.Array,
  axis: tuple[int, ...],
  reduction: Reduction,
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  broadcast: bool = True,
  where: chex.Array | None = None,
) -> chex.Array:
  """Dispatch reduction by name."""
  if reduction == Reduction.NONE:
    return array
  elif reduction == Reduction.SUM:
    return reduce_sum(
      array,
      axis,
      include_identity,
      include_all,
      include_other,
      broadcast,
      where,
    )
  elif reduction == Reduction.MEAN:
    return reduce_mean(
      array,
      axis,
      include_identity,
      include_all,
      include_other,
      broadcast,
      where,
    )
  elif reduction == Reduction.MEANVAR:
    return reduce_meanvar(
      array,
      axis,
      include_identity,
      include_all,
      include_other,
      broadcast,
      where,
    )
  elif reduction == Reduction.MAX:
    return reduce_max(
      array,
      axis,
      include_identity,
      include_all,
      include_other,
      broadcast,
      where,
    )
  elif reduction == Reduction.MIN:
    return reduce_min(
      array,
      axis,
      include_identity,
      include_all,
      include_other,
      broadcast,
      where,
    )
  else:
    raise ValueError(f"Unrecognized reduction: {reduction}")


@functools.partial(
  jax.jit,
  static_argnames=(
    "other_player_strat_reduction",
    "player_strat_reduction",
    "player_reduction",
    "include_identity",
    "include_all",
    "include_other",
  ),
)
def calc_pooled_payoffs_chan(
  *,
  payoffs_chan: chex.Array | None = None,
  joint_chan: chex.Array | None = None,
  other_player_strat_reduction: Reduction = Reduction.SUM,
  player_strat_reduction: Reduction = Reduction.SUM,
  player_reduction: Reduction = Reduction.SUM,
  include_identity: bool = True,
  include_all: bool = True,
  include_other: bool = True,
  joint_where: chex.Array | None = None,
) -> chex.Array:
  """Returns pooled payoffs channels with permutation-equivariant reductions.

  Applies 3-way reductions (identity / all / other) over:
    1. Opponent strategy axes (-p)
    2. Player p's own strategy axis
    3. The player axis (N)

  Resulting in up to 3x3x3 = 27 channel fold increase.

  Args:
      payoffs_chan: Array with shape [N, A1, ..., AN, C].
      joint_chan: Array with shape [A1, ..., AN, C].
      other_player_strat_reduction: Reduction over opponent axes.
      player_strat_reduction: Reduction over player p's axis.
      player_reduction: Reduction over the player axis.
      include_identity: Include the unreduced (identity) channel.
      include_all: Include the "all" reduced channel.
      include_other: Include the "other" reduced channel.
      joint_where: Mask with shape [A1, ..., AN].

  Returns:
      payoffs_chan: Array with shape [N, A1, ..., AN, C'].
  """

  joint_where_chan = None
  payoffs_where_chan = None

  if joint_where is not None:
    joint_where_chan = jnp.expand_dims(joint_where, axis=-1)

  if payoffs_chan is not None:
    num_players = payoffs_chan.shape[0]
    num_strats_per_player = payoffs_chan.shape[1:-1]

    if joint_where is not None:
      payoffs_where_chan = jnp.expand_dims(joint_where_chan, axis=0)
      payoffs_where_chan = jnp.broadcast_to(
        payoffs_where_chan, (num_players, *payoffs_where_chan.shape[1:])
      )

    # Reduce over strategies per player
    payoff_chan_per_player = []
    for player, payoff_chan in enumerate(tuple(payoffs_chan)):
      axes = [
        (*range(player), *range(player + 1, num_players)),
        (player,),
      ]
      reductions = [other_player_strat_reduction, player_strat_reduction]
      for axis, reduction in zip(axes, reductions):
        assert (payoff_chan.ndim - 1) not in axis, (
          "Cannot reduce over channel axis"
        )
        payoff_chan = reduce(
          payoff_chan,
          axis,
          reduction,
          include_identity,
          include_all,
          include_other,
          where=joint_where_chan,
        )
      payoff_chan_per_player.append(payoff_chan)
    payoffs_chan = jnp.stack(payoff_chan_per_player)

    # Reduce over the player axis
    payoffs_chan = reduce(
      payoffs_chan,
      (0,),
      player_reduction,
      include_identity,
      include_all,
      include_other,
      where=payoffs_where_chan,
    )

  elif joint_chan is not None:
    num_players = joint_chan.ndim - 1  # infer from joint shape [A1,...,AN,C]
    num_strats_per_player = joint_chan.shape[:-1]

    payoffs_chan = jnp.broadcast_to(
      joint_chan, (num_players, *num_strats_per_player, joint_chan.shape[-1])
    )
    payoffs_chan = calc_pooled_payoffs_chan(
      payoffs_chan=payoffs_chan,
      other_player_strat_reduction=other_player_strat_reduction,
      player_strat_reduction=player_strat_reduction,
      player_reduction=player_reduction,
      include_identity=include_identity,
      include_all=include_all,
      include_other=include_other,
      joint_where=joint_where,
    )

  return payoffs_chan
