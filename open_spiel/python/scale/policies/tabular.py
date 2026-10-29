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

"""A tabular policy: logits looked up by an InfoState index.

For games small enough to enumerate their information states, a table of
logits is the simplest learnable policy and an exact reference in tests.

  # kuhn.info_index: InfoState -> row
  from open_spiel.python.scale.envs import kuhn
  policy = TabularPolicy("kuhn_poker", num_states=12, num_actions=2,
                         num_players=2,
                         index_fn=f"{kuhn.__name__}:info_index")
  params = policy.params_from_logits(logits)  # f32[num_states, A]
"""

import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry

ARCH = "tabular"


class TabularPolicy(types.Policy):
  """params["logits"][index_fn(info)]. Stateless.

  Attributes:
    num_states: the table's rows.
    index_fn: the function mapping a batched InfoState to i32[B] in
      [0, num_states).
    spec: the PolicySpec.
  """

  def __init__(
      self,
      game: str,
      num_states: int,
      num_actions: int,
      num_players: int,
      index_fn: str,
      obs_format: str = "info_index",
  ):
    """Builds the table policy.

    Args:
      game: the game's name.
      num_states: the table's rows.
      num_actions: the game's number of actions.
      num_players: the game's number of players.
      index_fn: "module:attr" of a function mapping a batched InfoState to
        i32[B] in [0, num_states). An index outside that range reads NaN
        logits, which decide turns into NaN probabilities and the Arena
        rejects; it is never clamped onto another row.
      obs_format: the recorded observation format name.

    Raises:
      ValueError: num_states <= 0, or index_fn is not a string.
    """
    if num_states <= 0:
      raise ValueError(f"num_states must be > 0, got {num_states}")
    if not isinstance(index_fn, str):
      raise ValueError(
          f"index_fn is a 'module:attr' string, got {type(index_fn).__name__}"
      )
    self.num_states = num_states
    self.index_fn = registry.resolve_callable(index_fn)
    self.spec = types.PolicySpec(
        game=game,
        obs_format=obs_format,
        num_actions=num_actions,
        num_players=num_players,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=ARCH,
        arch_kwargs=dict(
            game=game,
            num_states=num_states,
            num_actions=num_actions,
            num_players=num_players,
            index_fn=index_fn,
            obs_format=obs_format,
        ),
    )

  def init_params(self, key=None):
    """Zero logits for every state.

    Args:
      key: unused.

    Returns:
      {"logits": f32[num_states, A] zeros}.
    """
    del key
    return self.params_from_logits(
        jnp.zeros((self.num_states, self.spec.num_actions), jnp.float32)
    )

  def params_from_logits(self, logits):
    """The params of a logits table.

    Args:
      logits: [num_states, A].

    Returns:
      {"logits": logits}.

    Raises:
      ValueError: the table has the wrong shape.
    """
    logits = jnp.asarray(logits, jnp.float32)
    if logits.shape != (self.num_states, self.spec.num_actions):
      raise ValueError(
          f"logits shape {logits.shape}; want "
          f"{(self.num_states, self.spec.num_actions)}"
      )
    return {"logits": logits}

  def encode(self, info):
    """The table row of each InfoState (index_fn).

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      i32[B] row indices.
    """
    return self.index_fn(info)

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """The logits row of each InfoState; NaN for rows out of the table.

    Args:
      params: the params.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput (zero values, no aux).
    """
    del carry, key
    idx = self.encode(step.info)
    in_range = (idx >= 0) & (idx < self.num_states)
    rows = params["logits"][jnp.where(in_range, idx, 0)]
    logits = jnp.where(in_range[:, None], rows, jnp.nan)
    b = logits.shape[0]
    return types.PolicyOutput(
        logits=logits.astype(jnp.float32),
        value=jnp.zeros((b, self.spec.num_players), jnp.float32),
        aux={},
    )


registry.register(ARCH, TabularPolicy)
