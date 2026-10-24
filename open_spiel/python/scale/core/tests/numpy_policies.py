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

"""Host-side test policies in numpy.

a HOST_DRIVEN history policy and a CALLBACK_BATCHED row recorder.

NumpyHistoryPolicy computes testing.fixtures.HistoryPolicy's maths in numpy
on the host, so a HostArena slot of it must reproduce an Arena slot of
HistoryPolicy under the same keys. NumpyRowRecorder implements
types.RowMaskedPolicy and records every batch its host callback receives, so
a test can see exactly which rows reached the host and which were marked as
padding.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types


class NumpyHistoryPolicy(types.Policy):
  """HistoryPolicy in numpy: HOST_DRIVEN, framework "numpy".

  Receives numpy SeatSteps and raw uint32 keys; params are one member's dict
  of numpy arrays {c, w, b}; the carry is {count: f32[B], hist: f32[B, A]}.
  evaluate: logits = c * count + hist @ w + b.

  Attributes:
    spec: the policy's PolicySpec.
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
    """Default params (make_params with no overrides).

    Args:
      key: unused.

    Returns:
      the params.
    """
    del key
    return self.make_params()

  def make_params(self, c=None, w=None, b=None):
    """Params with the given tables, defaults elsewhere.

    Args:
      c: the count weights, or None.
      w: the history weights, or None.
      b: the bias, or None.

    Returns:
      the params dict {"c", "w", "b"}.
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
    """Not used: this policy reads SeatStep fields only.

    Args:
      info: the info state.

    Raises:
      NotImplementedError: always.
    """
    raise NotImplementedError("reads SeatStep fields only")

  def init_carry(self, params, batch, seat, key):
    """Zero count and history.

    Args:
      params: the policy params.
      batch: the row count.
      seat: i32[batch], each row's seat.
      key: unused.

    Returns:
      the carry {"count", "hist"}.
    """
    del params, seat, key
    return {
        "count": np.zeros((batch,), np.float32),
        "hist": np.zeros((batch, self.spec.num_actions), np.float32),
    }

  def observe(self, params, carry, step):
    """Counts the ply and records the visible action.

    Args:
      params: the policy params.
      carry: the carry.
      step: the SeatStep.

    Returns:
      the updated carry.
    """
    del params
    reset = np.asarray(step.new_hand) | np.asarray(step.new_match)
    count = np.where(reset, np.float32(0), carry["count"]).astype(
        np.float32
    ) + np.float32(1)
    hist = np.where(reset[:, None], np.float32(0), carry["hist"]).astype(
        np.float32
    )
    la = np.asarray(step.last_action)
    seen = (np.asarray(step.last_actor) >= 0) & (la >= 0)
    oh = np.eye(self.spec.num_actions, dtype=np.float32)[np.maximum(la, 0)]
    hist = hist + np.where(seen[:, None], oh, np.float32(0))
    return {"count": count, "hist": hist.astype(np.float32)}

  def evaluate(self, params, carry, step, key):
    """Logits linear in the count and history.

    Args:
      params: the policy params.
      carry: the carry.
      step: the SeatStep.
      key: unused.

    Returns:
      the PolicyOutput.
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
    """Observes each ply of steps in order.

    Args:
      params: the policy params.
      carry: the carry.
      steps: SeatSteps [B, T, ...].

    Returns:
      the carry after the last ply.
    """
    for i in range(np.asarray(steps.seat).shape[1]):
      carry = self.observe(
          params,
          carry,
          jax.tree_util.tree_map(lambda x, i=i: x[:, i], steps),
      )
    return carry

  def evaluate_seq(self, params, carry, steps, keys):
    """Observes then evaluates each ply of steps in order.

    Args:
      params: the policy params.
      carry: the carry.
      steps: SeatSteps [B, T, ...].
      keys: per-ply keys [B, T, ...] or None.

    Returns:
      (final carry, PolicyOutput [B, T, ...]).
    """
    outs = []
    for i in range(np.asarray(steps.seat).shape[1]):
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


class NumpyRowRecorder(types.Policy):
  """A CALLBACK_BATCHED policy that records the rows its host callback gets.

  Stateless. The host computes logits = [seat, last_action + 1] per real row
  and zeros on padding rows, and appends (seat, last_action, real) of every
  call to `calls`, so a test reads how many rows reached the host and which
  were marked real. evaluate marks every row real; evaluate_rows takes the
  Arena's mask (types.RowMaskedPolicy).

  Attributes:
    spec: the policy's PolicySpec.
    calls: (seat, last_action, real) of every host call.
  """

  ARCH = "testing.numpy_row_recorder"

  def __init__(self, game: str, num_actions: int, num_players: int):
    self.calls: list[tuple[Any, Any, Any]] = []
    self.spec = types.PolicySpec(
        game=game,
        obs_format="seatstep",
        num_actions=num_actions,
        num_players=num_players,
        stateful=False,
        capability=types.Capability.CALLBACK_BATCHED,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            game=game, num_actions=num_actions, num_players=num_players
        ),
    )

  def init_params(self, key=None):
    """Member 0.

    Args:
      key: unused.

    Returns:
      {"member": 0}.
    """
    del key
    return {"member": jnp.int32(0)}

  def encode(self, info):
    """No features.

    Args:
      info: a batched InfoState.

    Returns:
      f32[B, 0].
    """
    leaves = jax.tree_util.tree_leaves(info)
    return jnp.zeros((leaves[0].shape[0], 0), jnp.float32)

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: the policy params.
      batch: the row count.
      seat: i32[batch], each row's seat.
      key: unused.

    Returns:
      ().
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """No carry to update.

    Args:
      params: the policy params.
      carry: the carry.
      step: the SeatStep.

    Returns:
      the carry unchanged.
    """
    del params, step
    return carry

  def real_rows_seen(self) -> int:
    """The real rows the host callback has received.

    Returns:
      the count.
    """
    return int(sum(np.asarray(real).sum() for _, _, real in self.calls))

  def rows_seen(self) -> int:
    """The rows, padding included, the host callback has received.

    Returns:
      the count.
    """
    return int(sum(np.asarray(real).size for _, _, real in self.calls))

  def evaluate(self, params, carry, step, key):
    """evaluate_rows with every row real.

    Args:
      params: the policy params.
      carry: the carry.
      step: the SeatStep.
      key: unused.

    Returns:
      the PolicyOutput.
    """
    return self.evaluate_rows(
        params, carry, step, key, jnp.ones(jnp.shape(step.seat), jnp.bool_)
    )

  def evaluate_rows(self, params, carry, step, key, rows):
    """Records the rows on the host; logits from seat and last action.

    Args:
      params: the policy params.
      carry: the carry.
      step: the SeatStep.
      key: unused.
      rows: bool[B], True for real rows.

    Returns:
      the PolicyOutput.
    """
    del carry, key
    a = self.spec.num_actions

    def host(member, seat, last_action, real):
      """The host side: records the call and computes logits.

      Args:
        member: i32 member per row.
        seat: i32 seat per row.
        last_action: i32 last action per row.
        real: bool real-row mask.

      Returns:
        f32 logits [..., A].
      """
      del member
      shape = np.shape(seat)
      seat, last_action, real = (
          np.asarray(x).reshape(-1) for x in (seat, last_action, real)
      )
      self.calls.append((seat.copy(), last_action.copy(), real.copy()))
      out = np.zeros((seat.size, a), np.float32)
      out[:, 0] = seat
      out[:, 1 % a] += last_action + 1
      out[~real] = 0.0
      return out.reshape(shape + (a,))

    b = jnp.shape(step.seat)[0]
    member = jnp.broadcast_to(params["member"], (b,))
    logits = jax.pure_callback(
        host,
        jax.ShapeDtypeStruct((b, a), jnp.float32),
        member,
        step.seat,
        step.last_action,
        rows,
        vmap_method="broadcast_all",
    )
    return types.PolicyOutput(
        logits=logits,
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )
