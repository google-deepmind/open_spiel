# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaZero: distillation of search targets into the network that searches.

AlphaZero (Silver et al. 2018) alternates self-play with a search operator
over the current network and a supervised update toward the search's visit
distributions and the game outcomes. Here each piece is a shared component:

  SearchPolicy (policies.search)   acts in the Arena; its visit distribution
      lands in TrajectoryBatch.target_policy (AUX_TARGET_POLICY).
  outcome_targets                  the value target: the seat's Monte Carlo
      outcome of the hand (sources.targets.with_mc_value_targets, gamma 1).
  PlyReplay                        flat-shuffle epochs over the single plies
      of the last `replay_buffer_size` iterations.
  BCLearner (learners.bc)          masked soft-target cross-entropy + value
      MSE (losses.distill_loss), AdamW with a cosine step size.
  the loop                         league.kinds.AZ runs AZ as a one-learner
      league: one league update is one iteration (a fresh game per row, then
      the replay epoch), checkpointed and resumed exactly through
      the train entry point.

  env, base, search_policy, arena, table, learner, replay = az.build(cfg)

Conventions:
  * the loss is a masked mean over decisions, and the policy cross-entropy
    masks illegal logits;
  * self-play runs every seat as a perspective row, so each decision is
    stored once with the seat to move, and only the first hand of each row's
    segment is kept (one game per row per iteration);
  * value rows of a hand that does not finish inside the segment are masked
    through labels[TARGET_VALUE_VALID]; with first-hand-only rows and a
    segment of at least max_len + 1 plies every kept hand finishes.

x64: counters (iteration, frames, learner step) are int64.
"""

from __future__ import annotations

import dataclasses
import io
import json
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.belief import registry as belief_registry
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.networks import policy as network_policy
from open_spiel.python.scale.policies import registry as policy_registry
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.sources import replay as replay_lib
from open_spiel.python.scale.sources import targets as targets_lib

tree_map = jax.tree_util.tree_map
F32 = jnp.float32


def jsonable(x: Any) -> Any:
  """X as plain JSON types (tuples become lists).

  E.g. a ConfigDict's to_dict().

  Args:
    x: a JSON-serialisable value.

  Returns:
    The JSON round trip of x.
  """
  return json.loads(json.dumps(x))


# -- targets -------------------------------------------------------------------
def outcome_targets(
    batch: types.TrajectoryBatch,
    first_hand_only: bool = False,
    value_scale: Optional[float] = None,
) -> types.TrajectoryBatch:
  """Monte Carlo outcome value targets (gamma 1).

  target_value = the perspective seat's Monte Carlo outcome of the hand
  (divided by value_scale when given), seat layout f32[B, T, P], and
  labels[TARGET_VALUE_VALID] = the hand ends inside the segment
  (sources.targets.with_mc_value_targets, gamma 1). first_hand_only narrows
  valid to each row's first hand; nothing else changes valid.

  Args:
    batch: the TrajectoryBatch.
    first_hand_only: narrow valid to each row's first hand.
    value_scale: divides the outcome when given.

  Returns:
    The batch with target_value and labels[TARGET_VALUE_VALID].
  """
  done = jnp.asarray(batch.done)
  assert done.dtype == jnp.bool_ and done.ndim == 2, done.shape
  if first_hand_only:
    ends_before = jnp.cumsum(done.astype(jnp.int32), axis=1) - done.astype(
        jnp.int32
    )
    batch = batch._replace(valid=jnp.asarray(batch.valid) & (ends_before == 0))
  return targets_lib.with_mc_value_targets(
      batch, gamma=1.0, value_scale=value_scale
  )


def search_value_targets(
    batch: types.TrajectoryBatch,
    *,
    lam: float,
    value_scale: float,
    first_hand_only: bool = False,
) -> types.TrajectoryBatch:
  """The expert-iteration value target with lam > 0.

  The expert-iteration value target with lam > 0 (league.kinds
  ExIt.value_lambda): sources.targets.with_search_value_targets(lam,
  value_scale, gamma=1), which replaces outcome_targets (it is never chained
  after it); first_hand_only narrows valid as outcome_targets does.

  Args:
    batch: the TrajectoryBatch.
    lam: the search value's weight, in (0, 1].
    value_scale: the operator's value scale.
    first_hand_only: narrow valid to each row's first hand.

  Returns:
    The batch with its value targets.
  """
  done = jnp.asarray(batch.done)
  assert done.dtype == jnp.bool_ and done.ndim == 2, done.shape
  if first_hand_only:
    ends_before = jnp.cumsum(done.astype(jnp.int32), axis=1) - done.astype(
        jnp.int32
    )
    batch = batch._replace(valid=jnp.asarray(batch.valid) & (ends_before == 0))
  return targets_lib.with_search_value_targets(
      batch, lam=lam, value_scale=value_scale, gamma=1.0
  )


# -- replay --------------------------------------------------------------------
def _flat_plies(batch: types.TrajectoryBatch) -> types.TrajectoryBatch:
  """[R, T, ...] -> [R*T, 1, ...].

  init_carry () and bootstrap_step = the ply.

  Args:
    batch: the TrajectoryBatch [R, T].

  Returns:
    The TrajectoryBatch [R*T, 1] of its single plies.
  """
  r, t = batch.valid.shape

  def per_ply(x):
    """Folds rows and time into plies.

    Args:
      x: A leaf, [r, t, ...].

    Returns:
      The leaf, [r * t, 1, ...].
    """
    return x.reshape((r * t, 1) + x.shape[2:])

  fields = {
      f: tree_map(per_ply, getattr(batch, f))
      for f in types.TrajectoryBatch._fields
      if f not in ("init_carry", "bootstrap_step")
  }
  steps = fields["steps"]
  return types.TrajectoryBatch(
      init_carry=(), bootstrap_step=tree_map(lambda x: x[:, 0], steps), **fields
  )


class PlyReplay:
  """Flat-shuffle epochs over the single plies of the last `num_slots` adds.

  Flat-shuffle epochs over the single plies (T = 1) of the last
  `num_slots` adds, held in a FIFO sources.replay.ReplayBuffer.

  samples_per_slot (batch_size * max_game_length per iteration) fixes the
  epoch: num_minibatches = num_slots * samples_per_slot // minibatch_size
  minibatches of minibatch_size plies, num_epochs copies. Stored decisions
  (valid plies) go first, so every one of them is in each copy as long as an
  add holds at most samples_per_slot of them (asserted); the rest is invalid
  padding. Slots not yet added are all-invalid padding, so the epoch shape is
  fixed from the first add. Needs a stateless policy: a ply carries no carry.

  Masked plies are NaN-free: in each epoch every
  non-decision ply, padding included, reads the SeatStep (steps and
  bootstrap_step) of one stored decision ply. An encoder may be undefined
  (NaN) off its domain, for example on a zero row or a finished game; this
  fill keeps such a value out of the masked loss's gradient (0 * NaN).
  Every other field of those plies is unchanged (valid False, zero targets
  and labels), so the loss and its gradient are the decision plies' alone; a
  decision ply is never touched (a NaN there is an encoder bug and stays
  loud). The fill is the epoch's: the stored plies (state_bytes) are the adds'.
  An epoch with no decision ply raises.

  Attributes:
    num_slots: the adds retained (FIFO).
    samples_per_slot: the plies each add contributes to an epoch.
    minibatch_size: plies per minibatch.
    num_epochs: copies of the replay per epochs() call.
    num_minibatches: minibatches per copy.
    usable: plies per copy, num_minibatches * minibatch_size.
  """

  def __init__(
      self,
      num_slots: int,
      samples_per_slot: int,
      minibatch_size: int,
      num_epochs: int,
  ):
    """Builds an empty replay.

    Args:
      num_slots: the adds retained (FIFO).
      samples_per_slot: the plies each add contributes to an epoch.
      minibatch_size: plies per minibatch.
      num_epochs: copies of the replay per epochs() call.
    """
    assert (
        num_slots >= 1
        and samples_per_slot >= 1
        and minibatch_size >= 1
        and num_epochs >= 1
    )
    self.num_slots, self.samples_per_slot = num_slots, samples_per_slot
    self.minibatch_size, self.num_epochs = minibatch_size, num_epochs
    self.num_minibatches = max(
        1, num_slots * samples_per_slot // minibatch_size
    )
    self.usable = self.num_minibatches * minibatch_size
    self._buffer = replay_lib.ReplayBuffer(
        capacity_iterations=num_slots, batch_size=minibatch_size
    )
    self._plies_per_add = None
    self._epoch = jax.jit(self._epoch_impl)

  @property
  def write_count(self) -> np.int64:
    """The number of adds made so far.

    Returns:
      The add count (int64).
    """
    return self._buffer.iterations_added

  def clear(self) -> None:
    """Drops every held ply.

    Drops every held ply (a generation boundary of expert iteration with a
    one-generation window, league.kinds ExIt.updates_per_generation): the next
    epoch reads only adds made after this. The add counter and the per-add
    shape stay.
    """
    self._buffer.clear()

  @property
  def num_held(self) -> int:
    """The number of adds held.

    Returns:
      The held add count.
    """
    return self._buffer.num_iterations

  def add(self, batch: types.TrajectoryBatch) -> None:
    """Stores one batch's plies as the newest slot.

    Args:
      batch: a TrajectoryBatch of a stateless policy.

    Raises:
      ValueError: the batch has a carry, holds more than samples_per_slot
        decisions, or changes the per-add shape.
    """
    if jax.tree_util.tree_leaves(batch.init_carry):
      raise ValueError(
          "PlyReplay stores single plies: the policy must be stateless "
          "(empty init_carry)"
      )
    n_valid = int(jnp.sum(batch.valid, dtype=jnp.int64))
    if n_valid > self.samples_per_slot:
      raise ValueError(
          f"{n_valid} decisions in one add > samples_per_slot "
          f"{self.samples_per_slot}: the epoch would drop some"
      )
    flat = _flat_plies(batch)
    n = flat.valid.shape[0]
    if self._plies_per_add is None:
      if self.num_slots * n < self.usable:
        raise ValueError(
            f"{self.num_slots} slots x {n} plies < one epoch ({self.usable})"
        )
      self._plies_per_add = n
    assert n == self._plies_per_add, (
        "every add must have the same shapes and dtypes",
        n,
        self._plies_per_add,
    )
    self._buffer.add(flat)  # checks tree, trailing shapes and dtypes

  def _epoch_impl(self, flat, key):
    """One epoch of minibatches drawn from the buffered plies.

    Invalid plies are filled from the first decision ply; `usable` plies, valid
    ones first, are shuffled into num_minibatches minibatches, which are tiled
    num_epochs times.

    Args:
      flat: the buffered plies.
      key: a PRNG key.

    Returns:
      the minibatches [num_epochs * num_minibatches, minibatch_size, ...].
    """
    valid = flat.valid[:, 0]
    src = jnp.argmax(
        valid
    )  # the first decision ply (epoch() checks there is one)

    def fill(x):
      """Fills invalid slots with a valid one.

      Args:
        x: A slot leaf.

      Returns:
        The filled leaf.
      """
      return jnp.where(
          valid.reshape((-1,) + (1,) * (x.ndim - 1)), x, x[src][None]
      )

    flat = flat._replace(
        steps=tree_map(fill, flat.steps),
        bootstrap_step=tree_map(fill, flat.bootstrap_step),
    )
    k1, k2 = jax.random.split(key)
    perm = jax.random.permutation(k1, valid.shape[0]).astype(jnp.int32)
    order = perm[jnp.argsort(~valid[perm], stable=True)]
    sel = order[: self.usable][jax.random.permutation(k2, self.usable)]
    m, mb, e = self.num_minibatches, self.minibatch_size, self.num_epochs
    mbs = tree_map(lambda x: x[sel].reshape((m, mb) + x.shape[1:]), flat)
    return tree_map(lambda x: jnp.tile(x, (e,) + (1,) * (x.ndim - 1)), mbs)

  def epoch(self, key) -> types.TrajectoryBatch:
    """Minibatches [num_epochs * num_minibatches, minibatch_size, 1, ...].

    Args:
      key: the PRNG key of the shuffle.

    Returns:
      Minibatches [num_epochs * num_minibatches, minibatch_size, 1, ...].

    Raises:
      ValueError: the replay holds no decision ply.
    """
    assert self._plies_per_add is not None, "epoch() before any add()"
    data = self._buffer.contents()
    pad = self.num_slots * self._plies_per_add - int(data.valid.shape[0])
    assert pad >= 0, pad
    if not bool(jnp.any(data.valid)):
      raise ValueError(
          "PlyReplay.epoch: no decision ply in the replay (nothing to train "
          "on, "
          "and nothing to fill the masked plies with)"
      )
    # iterations not yet added: invalid plies, a fixed epoch shape (their
    if pad:
      # SeatSteps are filled from a decision ply in _epoch_impl: class doc)
      data = tree_map(
          lambda x: jnp.concatenate(
              [x, jnp.zeros((pad,) + x.shape[1:], x.dtype)], axis=0
          ),
          data,
      )
    return self._epoch(data, key)

  def next(self, key) -> types.TrajectoryBatch:
    """One minibatch [minibatch_size, 1, ...] (TrajectorySource protocol).

    Args:
      key: the PRNG key.

    Returns:
      The minibatch.
    """
    return tree_map(lambda x: x[0], self.epoch(key))

  # -- persistence (resume) ----------------------------------------------------
  def state_bytes(self) -> bytes:
    """The held plies and the add counter as one .npz.

    Each add's flat leaves, oldest first; exact dtypes, so resumes restore them
    bitwise.

    Returns:
      The .npz bytes.

    Raises:
      TypeError: a leaf dtype .npz cannot store exactly.
    """
    items = self._buffer.items()
    arrays = {
        "iterations_added": np.asarray(self._buffer.iterations_added, np.int64),
        "num_items": np.asarray(len(items), np.int64),
    }
    for i, item in enumerate(items):
      for j, leaf in enumerate(jax.tree_util.tree_leaves(item)):
        a = np.asarray(leaf)
        if a.dtype.kind not in "biuf" or a.dtype.name == "bfloat16":
          raise TypeError(
              f"replay leaf {j} has dtype {a.dtype}: not storable exactly in "
              ".npz"
          )
        arrays[f"item{i:03d}_leaf{j:04d}"] = a
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()

  def load_state_bytes(self, data: bytes, like: types.TrajectoryBatch) -> None:
    """Restores state_bytes() into this (empty) replay.

    Args:
      data: state_bytes() output.
      like: a batch of the run (its first add after the resume) giving the tree
        and per-leaf trailing shapes / dtypes, which every stored leaf must
        match.

    Raises:
      ValueError: a stored leaf does not match `like`.
    """
    assert (
        int(self._buffer.iterations_added) == 0
    ), "load into an empty replay only"
    flat_like = _flat_plies(like)
    leaves_like, treedef = jax.tree_util.tree_flatten(flat_like)
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
      n = int(z["num_items"])
      added = np.int64(z["iterations_added"])
      for i in range(n):
        leaves = []
        for j, ref in enumerate(leaves_like):
          a = z[f"item{i:03d}_leaf{j:04d}"]
          if (
              a.dtype != np.asarray(ref).dtype
              or a.shape[1:] != np.shape(ref)[1:]
          ):
            raise ValueError(
                f"stored replay leaf {j}: {a.dtype}{a.shape} does not match "
                "the "
                f"run's {np.asarray(ref).dtype}{np.shape(ref)}"
            )
          leaves.append(jnp.asarray(a))
        item = jax.tree_util.tree_unflatten(treedef, leaves)
        if self._plies_per_add is None:
          self._plies_per_add = int(item.valid.shape[0])
        self._buffer.add(item)
    assert n <= self.num_slots and added >= n, (n, added)
    self._buffer.restore_iterations_added(added)


# -- config and loop -----------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AZConfig:
  """An AlphaZero run's settings.

  Self-play, search, replay and optimiser, plus the identities of the env,
  encoder and network.

  Attributes:
    env_name: the envs.registry name.
    encoder: the encoders registry name (the PolicySpec.obs_format).
    obs_shape: the encoder's observation shape.
    network: the actor_critic.build_network config.
    sampler: a belief.registry name: the search's sampler.
    num_worlds: K sampled worlds per search (PIMC when > 1).
    seed: the run's PRNG seed.
    num_simulations: search simulations per decision.
    gumbel_scale: the gumbel noise scale (gumbel search only,
      dirichlet_alpha == 0).
    temperature: the self-play slot's DecisionRule("sample", T).
    dirichlet_alpha: > 0: PUCT + Dirichlet (muzero_policy); 0: gumbel search.
    dirichlet_epsilon: the Dirichlet noise's weight in the root prior.
    temperature_drop_move: > 0: core.arena.TemperatureDrop(this, post_drop).
    temperature_post_drop: the self-play temperature after the drop.
    batch_size: self-play games per iteration.
    num_iterations: self-play + train iterations.
    num_epochs: replay epochs per iteration.
    minibatch_size: plies per minibatch.
    replay_buffer_size: iterations of self-play the replay retains.
    lr: the initial step size of the cosine schedule.
    lr_end: the step size the cosine schedule ends at.
    weight_decay: the AdamW weight decay.
    max_grad_norm: the global gradient-norm clip.
    value_coef: the value MSE's weight (0.5 * squared error, optax.l2_loss).
  """

  env_name: str
  encoder: str
  obs_shape: Sequence[int]
  network: Mapping[str, Any]
  sampler: str = "perfect_information"
  num_worlds: int = 1
  seed: int = 0
  num_simulations: int = 32
  gumbel_scale: float = 1.0
  temperature: float = 1.0
  dirichlet_alpha: float = 1.0
  dirichlet_epsilon: float = 0.25
  temperature_drop_move: int = 0
  temperature_post_drop: float = 1e-3
  batch_size: int = 256
  num_iterations: int = 100
  num_epochs: int = 10
  minibatch_size: int = 4096
  replay_buffer_size: int = 1
  lr: float = 1e-3
  lr_end: float = 1e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0
  value_coef: float = 0.5

  def __post_init__(self):
    assert (
        self.sampler in belief_registry.names()
    ), f"sampler {self.sampler!r} not in {belief_registry.names()}"
    assert self.num_worlds >= 1
    assert (
        self.batch_size >= 1
        and self.num_iterations >= 1
        and self.num_epochs >= 1
    )
    assert self.replay_buffer_size >= 1 and self.minibatch_size >= 1
    assert self.dirichlet_alpha >= 0.0
    assert self.temperature > 0 and self.temperature_post_drop > 0
    assert self.temperature_drop_move >= 0

  @classmethod
  def from_mapping(cls, m: Mapping[str, Any]) -> "AZConfig":
    """An AZConfig from a mapping of its fields.

    Args:
      m: {field: value}.

    Returns:
      The AZConfig.

    Raises:
      ValueError: m has a key that is not a field.
    """
    m = jsonable(dict(m))
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(m) - names
    if unknown:
      raise ValueError(f"unknown AZ config keys {sorted(unknown)}")
    return cls(**m)

  def slot(self, policy: types.Policy) -> "arena_lib.Slot":
    """The self-play Arena slot.

    The temperature schedule belongs to the slot (DecisionRule plus
    TemperatureDrop), not to the SearchPolicy.

    Args:
      policy: the SearchPolicy that acts.

    Returns:
      The arena Slot.
    """
    drop = None
    if self.temperature_drop_move > 0:
      drop = arena_lib.TemperatureDrop(
          self.temperature_drop_move, self.temperature_post_drop
      )
    return arena_lib.Slot(
        policy,
        1,
        types.DecisionRule("sample", self.temperature),
        temperature_drop=drop,
    )

  def mcts(self) -> search.MCTSConfig:
    """The search config.

    dirichlet_alpha > 0 selects PUCT with Dirichlet root noise; 0 selects gumbel
    search, which reads no Dirichlet alpha, so the MCTSConfig keeps its own
    default for that unused field.

    Returns:
      The search.MCTSConfig.
    """
    muzero = self.dirichlet_alpha > 0
    noise = {"dirichlet_alpha": self.dirichlet_alpha} if muzero else {}
    return search.MCTSConfig(
        num_simulations=self.num_simulations,
        search="muzero" if muzero else "gumbel",
        dirichlet_fraction=self.dirichlet_epsilon,
        gumbel_scale=self.gumbel_scale,
        num_worlds=self.num_worlds,
        **noise,
    )


def make_env(cfg: AZConfig) -> types.Env:
  """The run's env at its default options.

  Args:
    cfg: the AZConfig.

  Returns:
    The Env.
  """
  return env_registry.make(cfg.env_name)


def base_policy_kwargs(cfg: AZConfig, env: types.Env) -> dict[str, Any]:
  """The NetworkPolicy kwargs of the AZ base network named by the AZConfig.

  Encoder, obs_shape, network; the zero-sum value layout.

  Args:
    cfg: the AZConfig.
    env: the run's env.

  Returns:
    The kwargs dict.
  """
  return dict(
      game=env.name,
      obs_format=cfg.encoder,
      num_actions=env.num_actions,
      num_players=env.num_players,
      encoder=cfg.encoder,
      obs_shape=list(cfg.obs_shape),
      network=dict(cfg.network),
      value_layout="zero_sum",
  )


def num_minibatches(cfg: AZConfig, max_len: Optional[int] = None) -> int:
  """max(1, replay_buffer_size * batch_size * max_len // minibatch_size).

  Args:
    cfg: the AZConfig.
    max_len: the env's max game length (default the env's).

  Returns:
    The minibatch count of an epoch.
  """
  max_len = make_env(cfg).max_len if max_len is None else max_len
  return max(
      1, cfg.replay_buffer_size * cfg.batch_size * max_len // cfg.minibatch_size
  )


class AZBuild(NamedTuple):
  """The components of an AlphaZero run (build).

  Attributes:
    env: the env.
    base: the network trained and searched over.
    self_play: the self-play SearchPolicy; None with with_search=False.
    arena: the self-play Arena; None with with_search=False.
    table: the self-play slot table.
    learner: the BCLearner distilling the search targets.
    replay: the PlyReplay.
  """

  env: types.Env
  base: types.Policy
  self_play: Optional[search.SearchPolicy]
  arena: Optional[arena_lib.Arena]
  table: Any
  learner: bc.BCLearner
  replay: PlyReplay


def build(
    cfg: AZConfig,
    mesh: Optional[Any] = None,
    *,
    env: Optional[types.Env] = None,
    policy_arch: str = "network",
    policy_kwargs: Optional[Mapping[str, Any]] = None,
    with_search: bool = True,
) -> AZBuild:
  """The components of an AlphaZero run.

  Args:
    cfg: the run's settings.
    mesh: the ("batch",) device mesh of the Arena (default every device).
    env: the run's env (default make_env(cfg), the game at its default
      options); it must be cfg.env_name.
    policy_arch: the base network's architecture, built through
      policies.registry. The base must be a NATIVE_JAX JAX policy of the env;
      its per-seat value is the search's leaf value.
    policy_kwargs: the base network's kwargs; empty policy_kwargs builds the
      AZConfig's network (base_policy_kwargs).
    with_search: False builds no SearchPolicy and no Arena, for a caller that
      acts with another operator (league.slot_factories).

  Returns:
    The AZBuild.

  Raises:
    ValueError: env or base do not match cfg.
  """
  if env is None:
    env = make_env(cfg)
  if env.name != cfg.env_name:
    raise ValueError(
        f"az.build: env {env.name!r} != AZConfig.env_name {cfg.env_name!r}"
    )
  if policy_kwargs:
    base = policy_registry.build(policy_arch, dict(policy_kwargs))
  else:
    if policy_arch != "network":
      raise ValueError(
          f"policy_arch {policy_arch!r} without policy_kwargs: the AZConfig's "
          "network is a 'network' architecture"
      )
    base = network_policy.NetworkPolicy(**base_policy_kwargs(cfg, env))
  bs = base.spec
  if (bs.game, bs.num_actions, bs.num_players) != (
      env.name,
      env.num_actions,
      env.num_players,
  ):
    raise ValueError(
        f"the AZ base network {bs.arch_name} plays {bs.game} ({bs.num_actions} "
        f"actions, {bs.num_players} players), not the env {env.name}"
    )
  sp = arena = None
  if with_search:
    opponent = base if belief_registry.takes_opponent(cfg.sampler) else None
    sampler = belief_registry.build(cfg.sampler, env, opponent=opponent)
    sp = search.SearchPolicy(base, env, sampler, cfg.mcts())
    arena = arena_lib.Arena(env, [cfg.slot(sp)], mesh=mesh)
  table = arena_lib.self_play_table(
      cfg.batch_size, env.num_players, all_seats=True
  )
  replay = PlyReplay(
      num_slots=cfg.replay_buffer_size,
      samples_per_slot=cfg.batch_size * env.max_len,
      minibatch_size=cfg.minibatch_size,
      num_epochs=cfg.num_epochs,
  )
  assert replay.num_minibatches == num_minibatches(cfg, env.max_len)
  total = cfg.num_iterations * cfg.num_epochs * replay.num_minibatches
  learner = bc.BCLearner(
      base,
      bc.BCConfig(
          loss=losses.DistillLossConfig(
              policy="ce_target", value="mse", value_coef=cfg.value_coef
          ),
          learning_rate=cfg.lr,
          lr_end=cfg.lr_end,
          lr_schedule="cosine",
          total_steps=total,
          weight_decay=cfg.weight_decay,
          max_grad_norm=cfg.max_grad_norm,
      ),
  )
  return AZBuild(env, base, sp, arena, table, learner, replay)
