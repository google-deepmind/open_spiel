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

"""Small test policies whose carry and outputs are easy to reason about.

Tests of the Arena, the Evaluator, checkpoints and the contract checkers need
policies with known outputs under every capability. One maths, HistoryPolicy
(count the plies, add a one-hot of the actions seen), comes as an in-graph
JAX policy and as a numpy HOST_DRIVEN twin, so a host-driven rollout can be
compared with an in-graph one bit for bit.

  HistoryPolicy          NATIVE_JAX, stateful
  NumpyHistoryPolicy     HOST_DRIVEN, HistoryPolicy's maths in numpy
  CallbackTabularPolicy  CALLBACK_BATCHED: a host table behind jax.pure_callback
  NumpyRowRecorder       CALLBACK_BATCHED and RowMaskedPolicy; records the rows
                         its host function receives
  DtypeProbePolicy       params with an f32, an f64 and an i64 leaf

The architectures are registered only inside `registered()`; a test that
rebuilds a fixture policy from its spec runs inside it (the package's conftest
does this for the whole session).

  pol = fixtures.HistoryPolicy("kuhn_poker", num_actions=2, num_players=2)
  params = pol.make_params(w=np.eye(2, dtype=np.float32))
"""

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry


class HistoryPolicy(types.Policy):
  """Stateful fixture: logits from the hand's action history.

  Carry: {count: f32[B], hist: f32[B, A]} (+ hands: f32[B] when cross_hand).

  observe: per-hand state resets on new_hand (or new_match), then count += 1
    and hist += one_hot(last_action) when last_actor >= 0. With cross_hand,
    `hands` counts the hands begun in the match: it resets only on new_match
    and increments on new_hand.
  evaluate: logits = c * count + hist @ w + b (+ d * hands). With c = e_0 and
    w = b = 0, logits[:, 0] is the number of observes since the hand began.

  Attributes:
    cross_hand: Whether the history persists across hands.
    spec: The PolicySpec.
  """

  ARCH = "testing.history"

  def __init__(
      self,
      game: str,
      num_actions: int,
      num_players: int,
      cross_hand: bool = False,
  ):
    self.cross_hand = cross_hand
    self.spec = types.PolicySpec(
        game=game,
        obs_format="seatstep",
        num_actions=num_actions,
        num_players=num_players,
        stateful=True,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            game=game,
            num_actions=num_actions,
            num_players=num_players,
            cross_hand=cross_hand,
        ),
    )

  def init_params(self, key=None):
    """Returns fresh params for this architecture (one member, unstacked).

    Args:
      key: The PRNG key; unused by this fixture.

    Returns:
      The params.
    """
    del key
    return self.make_params()

  def make_params(self, c=None, w=None, b=None, d=None):
    """Returns params with the given leaves, defaults elsewhere.

    Args:
      c: f32[A], the count weights; defaults to e_0.
      w: f32[A, A], the history weights; defaults to zeros.
      b: f32[A], the bias; defaults to zeros.
      d: f32[A], the cross-hand weights; defaults to zeros.

    Returns:
      {"c", "w", "b", "d"} as f32 arrays.
    """
    a = self.spec.num_actions
    c = jnp.zeros((a,), jnp.float32).at[0].set(1.0) if c is None else c
    w = jnp.zeros((a, a), jnp.float32) if w is None else w
    b = jnp.zeros((a,), jnp.float32) if b is None else b
    d = jnp.zeros((a,), jnp.float32) if d is None else d
    return {
        "c": jnp.asarray(c, jnp.float32),
        "w": jnp.asarray(w, jnp.float32),
        "b": jnp.asarray(b, jnp.float32),
        "d": jnp.asarray(d, jnp.float32),
    }

  def encode(self, info):
    """Returns the features [B, ...] of a batched InfoState.

    The policy owns its encoding.

    Args:
      info: The batched InfoState.

    Returns:
      The features.
    """
    leaves = jax.tree_util.tree_leaves(info)
    return jnp.zeros((leaves[0].shape[0], 0), jnp.float32)

  def init_carry(self, params, batch, seat, key):
    """Returns the initial carry of `batch` rows.

    Args:
      params: The params.
      batch: The number of rows.
      seat: i32[batch], each row's seat.
      key: The PRNG key.

    Returns:
      The carry.
    """
    del params, seat, key
    c = {
        "count": jnp.zeros((batch,), jnp.float32),
        "hist": jnp.zeros((batch, self.spec.num_actions), jnp.float32),
    }
    if self.cross_hand:
      c["hands"] = jnp.zeros((batch,), jnp.float32)
    return c

  def observe(self, params, carry, step):
    """Observes one step; called for every seat at every ply.

    Resets on step.new_hand/new_match.

    Args:
      params: The params.
      carry: The carry.
      step: The SeatStep.

    Returns:
      The carry after the step.
    """
    del params
    reset = step.new_hand | step.new_match
    count = jnp.where(reset, 0.0, carry["count"]) + 1.0
    hist = jnp.where(reset[:, None], 0.0, carry["hist"])
    seen = (step.last_actor >= 0)[:, None]
    # dtype stated: one_hot defaults to float64 under x64 and would promote
    # hist.
    oh = jax.nn.one_hot(
        jnp.maximum(step.last_action, 0),
        self.spec.num_actions,
        dtype=jnp.float32,
    )
    hist = hist + jnp.where(seen & (step.last_action >= 0)[:, None], oh, 0.0)
    out = {"count": count, "hist": hist}
    if self.cross_hand:
      out["hands"] = jnp.where(
          step.new_match, 0.0, carry["hands"]
      ) + step.new_hand.astype(jnp.float32)
    return out

  def evaluate(self, params, carry, step, key):
    """Evaluates the acting rows (only those, in the Arena).

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    del step  # Unused.
    del key
    logits = (
        params["c"][None, :] * carry["count"][:, None]
        + carry["hist"] @ params["w"]
        + params["b"][None, :]
    )
    if self.cross_hand:
      logits = logits + params["d"][None, :] * carry["hands"][:, None]
    b = logits.shape[0]
    return types.PolicyOutput(
        logits=logits.astype(jnp.float32),
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )


class NumpyHistoryPolicy(types.Policy):
  """HistoryPolicy's maths in numpy, driven by the HostArena (HOST_DRIVEN).

  It receives numpy SeatSteps and raw uint32[B, 2] keys, as every host-driven
  policy does; params are one member's dict of numpy arrays {c, w, b}; the
  carry is {count: f32[B], hist: f32[B, A]}. For equal params its logits
  equal HistoryPolicy's (cross_hand=False) bit for bit.

  Attributes:
    spec: The PolicySpec.
  """

  ARCH = "testing.numpy_history"

  def __init__(self, game: str, num_actions: int, num_players: int):
    self.spec = types.PolicySpec(
        game=game,
        obs_format="seatstep",
        num_actions=num_actions,
        num_players=num_players,
        stateful=True,
        capability=types.Capability.HOST_DRIVEN,
        framework="numpy",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            game=game, num_actions=num_actions, num_players=num_players
        ),
    )

  def init_params(self, key=None):
    """Returns fresh params for this architecture (one member, unstacked).

    Args:
      key: The PRNG key; unused by this fixture.

    Returns:
      The params.
    """
    del key
    return self.make_params()

  def make_params(self, c=None, w=None, b=None):
    """Returns numpy params with the given leaves, defaults elsewhere.

    Args:
      c: f32[A], the count weights; defaults to e_0.
      w: f32[A, A], the history weights; defaults to zeros.
      b: f32[A], the bias; defaults to zeros.

    Returns:
      {"c", "w", "b"} as f32 numpy arrays.
    """
    a = self.spec.num_actions
    c = (
        np.eye(a, dtype=np.float32)[0]
        if c is None
        else np.asarray(c, np.float32)
    )
    w = np.zeros((a, a), np.float32) if w is None else np.asarray(w, np.float32)
    b = np.zeros((a,), np.float32) if b is None else np.asarray(b, np.float32)
    return {"c": c, "w": w, "b": b}

  def encode(self, info):
    """Not implemented: the policy reads SeatStep fields only.

    Args:
      info: The batched InfoState.

    Raises:
      NotImplementedError: always.
    """
    raise NotImplementedError("reads SeatStep fields only")

  def init_carry(self, params, batch, seat, key):
    """Returns the initial carry of `batch` rows.

    Args:
      params: The params.
      batch: The number of rows.
      seat: i32[batch], each row's seat.
      key: The PRNG key.

    Returns:
      The carry.
    """
    del params, seat, key
    return {
        "count": np.zeros((batch,), np.float32),
        "hist": np.zeros((batch, self.spec.num_actions), np.float32),
    }

  def observe(self, params, carry, step):
    """Observes one step; called for every seat at every ply.

    Resets on step.new_hand/new_match.

    Args:
      params: The params.
      carry: The carry.
      step: The SeatStep.

    Returns:
      The carry after the step.
    """
    del params
    reset = np.asarray(step.new_hand) | np.asarray(step.new_match)
    count = np.where(reset, np.float32(0), carry["count"]) + np.float32(1)
    hist = np.where(reset[:, None], np.float32(0), carry["hist"])
    la = np.asarray(step.last_action).astype(np.int64)
    seen = (np.asarray(step.last_actor) >= 0) & (la >= 0)
    oh = np.eye(self.spec.num_actions, dtype=np.float32)[np.maximum(la, 0)]
    hist = (hist + np.where(seen[:, None], oh, np.float32(0))).astype(
        np.float32
    )
    return {"count": count.astype(np.float32), "hist": hist}

  def evaluate(self, params, carry, step, key):
    """Evaluates the acting rows (only those, in the Arena).

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    del step  # Unused.
    del key
    logits = (
        params["c"][None, :] * carry["count"][:, None]
        + carry["hist"] @ params["w"]
        + params["b"][None, :]
    )
    b = logits.shape[0]
    return types.PolicyOutput(
        logits=logits.astype(np.float32),
        value=np.zeros((b, self.spec.num_players), np.float32),
        aux={},
    )

  def observe_seq(self, params, carry, steps):
    """Returns the carry after observing steps [B, T, ...] in order.

    The default (JAX only) is a scan of `observe`; an override must equal it
    (testing.check_policy).

    Args:
      params: The params.
      carry: The carry.
      steps: The SeatSteps [B, T, ...].

    Returns:
      The carry after the last step.
    """
    t = np.asarray(steps.seat).shape[1]
    for i in range(t):
      carry = self.observe(
          params, carry, jax.tree_util.tree_map(lambda x, i=i: x[:, i], steps)
      )
    return carry

  def evaluate_seq(self, params, carry, steps, keys):
    """Observes then evaluates at every ply of steps [B, T].

    The default (JAX only) is a scan; an override must equal it
    (testing.check_policy).

    Args:
      params: The params.
      carry: The carry.
      steps: The SeatSteps [B, T, ...].
      keys: What evaluate receives at each ply, stacked on a time axis (JAX:
        [T] keys; host: raw uint32[B, T, 2]), or None.

    Returns:
      (final carry, PolicyOutput [B, T, ...]).
    """
    t = np.asarray(steps.seat).shape[1]
    outs = []
    for i in range(t):
      st = jax.tree_util.tree_map(lambda x, i=i: x[:, i], steps)
      carry = self.observe(params, carry, st)
      outs.append(
          self.evaluate(params, carry, st, None if keys is None else keys[:, i])
      )
    return carry, types.PolicyOutput(
        logits=np.stack([o.logits for o in outs], 1),
        value=np.stack([o.value for o in outs], 1),
        aux={},
    )


class CallbackTabularPolicy(types.Policy):
  """CALLBACK_BATCHED fixture: a host (numpy) table behind a pure_callback.

  The callback is jax.pure_callback(vmap_method="broadcast_all"). The weights
  stay host-side ([K, N, A] numpy); the jax params are only
  {"member": i32 id} (stacked [K] in a slot), so each row's member id reaches
  the callback and the host indexes its own weights.

  Attributes:
    host_tables: f32[members, num_states, num_actions] the host-side logits.
    index_fn: The callable that maps an observation to its state index.
    num_states: The number of table rows.
    spec: The PolicySpec.
  """

  ARCH = "testing.callback_tabular"

  def __init__(
      self,
      game: str,
      num_states: int,
      num_actions: int,
      num_players: int,
      index_fn: str,
  ):
    self.num_states = num_states
    self.index_fn = registry.resolve_callable(index_fn)
    self.host_tables = np.zeros((1, num_states, num_actions), np.float32)
    self.spec = types.PolicySpec(
        game=game,
        obs_format="info_index",
        num_actions=num_actions,
        num_players=num_players,
        stateful=False,
        capability=types.Capability.CALLBACK_BATCHED,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            game=game,
            num_states=num_states,
            num_actions=num_actions,
            num_players=num_players,
            index_fn=index_fn,
        ),
    )

  def set_host_tables(self, tables) -> None:
    """Sets the host-side weights.

    Args:
      tables: [K, num_states, num_actions], one table per member.
    """
    tables = np.asarray(tables, np.float32)
    assert tables.ndim == 3 and tables.shape[1:] == (
        self.num_states,
        self.spec.num_actions,
    )
    self.host_tables = tables

  def init_params(self, key=None):
    """Returns fresh params for this architecture (one member, unstacked).

    Args:
      key: The PRNG key; unused by this fixture.

    Returns:
      The params.
    """
    del key
    return {"member": jnp.int32(0)}

  def encode(self, info):
    """Returns the features [B, ...] of a batched InfoState.

    The policy owns its encoding.

    Args:
      info: The batched InfoState.

    Returns:
      The features.
    """
    return self.index_fn(info)

  def init_carry(self, params, batch, seat, key):
    """Returns the initial carry of `batch` rows.

    Args:
      params: The params.
      batch: The number of rows.
      seat: i32[batch], each row's seat.
      key: The PRNG key.

    Returns:
      The carry.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """Observes one step; called for every seat at every ply.

    Resets on step.new_hand/new_match.

    Args:
      params: The params.
      carry: The carry.
      step: The SeatStep.

    Returns:
      The carry after the step.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """Evaluates the acting rows (only those, in the Arena).

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    del carry, key
    idx = self.encode(step.info).astype(jnp.int32)
    a = self.spec.num_actions

    def host_lookup(member, index):
      """Looks up each row's logits in its member's host table.

      Args:
        member: The member ids.
        index: The table rows.

      Returns:
        f32 logits [..., A].
      """
      member = np.asarray(member)
      index = np.asarray(index)
      m = np.broadcast_to(
          member.reshape(member.shape + (1,) * (index.ndim - member.ndim)),
          index.shape,
      )
      return self.host_tables[m, index].astype(np.float32)

    logits = jax.pure_callback(
        host_lookup,
        jax.ShapeDtypeStruct(idx.shape + (a,), jnp.float32),
        params["member"],
        idx,
        vmap_method="broadcast_all",
    )
    b = idx.shape[0]
    return types.PolicyOutput(
        logits=logits,
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )


class NumpyRowRecorder(CallbackTabularPolicy):
  """A CALLBACK_BATCHED table that records the rows its host function receives.

  It implements core.types.RowMaskedPolicy: `evaluate_rows(..., rows)` passes
  the real-row mask bool[B] to the host, which looks up only the real rows
  (padding rows get zeros) and appends (member i32[B], index i32[B], rows
  bool[B]) to `calls`. A test of the Arena's padding contract asserts that
  every recorded call's real rows are exactly the rows the slot owns.

  Attributes:
    calls: (member, index, real) arrays, one per callback call.
    spec: The PolicySpec.
  """

  ARCH = "testing.numpy_row_recorder"

  def __init__(
      self,
      game: str,
      num_states: int,
      num_actions: int,
      num_players: int,
      index_fn: str,
  ):
    super().__init__(game, num_states, num_actions, num_players, index_fn)
    self.spec = dataclasses.replace(self.spec, arch_name=self.ARCH)
    self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

  def evaluate(self, params, carry, step, key):
    """Evaluates the acting rows (only those, in the Arena).

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    rows = jnp.ones(step.legal_mask.shape[:1], jnp.bool_)
    return self.evaluate_rows(params, carry, step, key, rows)

  def evaluate_rows(self, params, carry, step, key, rows):
    """Evaluates the rows where `rows` is True; other rows' outputs are zeros.

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.
      rows: bool[B], True for real rows.

    Returns:
      The PolicyOutput.
    """
    del carry, key
    idx = self.encode(step.info).astype(jnp.int32)
    a = self.spec.num_actions

    def host_lookup(member, index, real):
      """Looks up and records each row's logits; padding rows get zeros.

      Args:
        member: The member ids.
        index: The table rows.
        real: True for real rows.

      Returns:
        f32 logits [..., A].
      """
      member, index, real = (
          np.asarray(member),
          np.asarray(index),
          np.asarray(real),
      )
      m = np.broadcast_to(
          member.reshape(member.shape + (1,) * (index.ndim - member.ndim)),
          index.shape,
      )
      self.calls.append((m.copy(), index.copy(), real.copy()))
      out = self.host_tables[m, index].astype(np.float32)
      return np.where(real[..., None], out, np.float32(0))

    logits = jax.pure_callback(
        host_lookup,
        jax.ShapeDtypeStruct(idx.shape + (a,), jnp.float32),
        params["member"],
        idx,
        jnp.asarray(rows, jnp.bool_),
        vmap_method="broadcast_all",
    )
    b = idx.shape[0]
    return types.PolicyOutput(
        logits=logits,
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )


class DtypeProbePolicy(types.Policy):
  """Checkpoint dtype fixture that plays uniform (zero logits).

  Params: {w32: f32[n32], w64: <w64_dtype>[n64], count: i64[]} (an f64 leaf,
  an f32 leaf and an int64 counter); it exists so the round trip of each dtype
  is tested.

  Attributes:
    n32: The number of float32 weights.
    n64: The number of 64-bit weights.
    spec: The PolicySpec.
    w64_dtype: The dtype of the 64-bit weights.
  """

  ARCH = "testing.dtype_probe"

  def __init__(
      self,
      game: str,
      num_actions: int,
      num_players: int,
      w64_dtype: str = "float64",
      n32: int = 3,
      n64: int = 2,
  ):
    assert w64_dtype in ("float64", "float32"), w64_dtype
    self.w64_dtype = jnp.dtype(w64_dtype)
    self.n32, self.n64 = n32, n64
    self.spec = types.PolicySpec(
        game=game,
        obs_format="none",
        num_actions=num_actions,
        num_players=num_players,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            game=game,
            num_actions=num_actions,
            num_players=num_players,
            w64_dtype=w64_dtype,
            n32=n32,
            n64=n64,
        ),
    )

  def init_params(self, key=None):
    """Returns fresh params for this architecture (one member, unstacked).

    Args:
      key: The PRNG key; unused by this fixture.

    Returns:
      The params.
    """
    del key
    return self.make_params()

  def make_params(self, w32=None, w64=None, count=None):
    """Returns params with the given leaves, zeros elsewhere.

    Args:
      w32: f32[n32]; defaults to zeros.
      w64: <w64_dtype>[n64]; defaults to zeros.
      count: The int64 counter; defaults to 0.

    Returns:
      {"w32", "w64", "count"}.
    """
    w32 = np.zeros((self.n32,), np.float32) if w32 is None else w32
    w64 = np.zeros((self.n64,), self.w64_dtype) if w64 is None else w64
    count = np.int64(0) if count is None else count
    out = {
        "w32": jnp.asarray(w32, jnp.float32),
        "w64": jnp.asarray(w64, self.w64_dtype),
        "count": jnp.asarray(count, jnp.int64),
    }
    assert out["w32"].shape == (self.n32,) and out["w64"].shape == (self.n64,)
    return out

  def encode(self, info):
    """Returns the features [B, ...] of a batched InfoState.

    The policy owns its encoding.

    Args:
      info: The batched InfoState.

    Returns:
      The features.
    """
    leaves = jax.tree_util.tree_leaves(info)
    return jnp.zeros((leaves[0].shape[0], 0), jnp.float32)

  def init_carry(self, params, batch, seat, key):
    """Returns the initial carry of `batch` rows.

    Args:
      params: The params.
      batch: The number of rows.
      seat: i32[batch], each row's seat.
      key: The PRNG key.

    Returns:
      The carry.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """Observes one step; called for every seat at every ply.

    Resets on step.new_hand/new_match.

    Args:
      params: The params.
      carry: The carry.
      step: The SeatStep.

    Returns:
      The carry after the step.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """Evaluates the acting rows (only those, in the Arena).

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    del params, carry, key
    b = step.legal_mask.shape[0]
    return types.PolicyOutput(
        logits=jnp.zeros((b, self.spec.num_actions), jnp.float32),
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )


#: Every fixture architecture, by architecture name.
ARCHS = {
    cls.ARCH: cls
    for cls in (
        HistoryPolicy,
        NumpyHistoryPolicy,
        CallbackTabularPolicy,
        NumpyRowRecorder,
        DtypeProbePolicy,
    )
}


@contextlib.contextmanager
def registered():
  """Registers every fixture architecture for the duration of the block.

  Artifacts of fixture policies can then be rebuilt through
  policies.registry:

    with fixtures.registered():
      art = artifact.load_artifact(ckpt_dir, step)

  Yields:
    None, inside the block.
  """
  with contextlib.ExitStack() as stack:
    for name, cls in ARCHS.items():
      stack.enter_context(registry.temporary(name, cls))
    yield
