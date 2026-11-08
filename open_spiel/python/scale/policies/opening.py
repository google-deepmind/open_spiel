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

"""Uniformly random legal moves for the first N plies, then a base policy.

Self-play from a deterministic policy visits few openings. Randomizing the
first plies diversifies the positions a learner sees, while the base policy
still labels every ply. OpeningPolicy wraps any in-graph base policy, so
every consumer (behaviour-cloning self-play, held-out sets, AlphaZero,
evaluation) gets the same opening.

  policy = OpeningPolicy(teacher, num_plies=4)
  params = policy.make_params(teacher_params)

Ply index. Every seat observes every ply, so a per-seat count of the plies
observed this hand is the hand's ply index, equal for all seats. The carry
is {"base": the base's carry, "ply": i64[B]}, reset on new_hand and
new_match. A row is in the opening while ply <= num_plies; chance between
plies does not count.

evaluate. The base is evaluated on every row, so its aux (for example a
teacher's target distribution) labels the random plies too. On opening rows
the logits are 0 at an action drawn uniformly over the legal actions and
-inf elsewhere, so every DecisionRule plays it; value and aux are the
base's, plus aux[AUX_OPENING] (bool[B]) marking the opening rows. The
evaluate key k gives fold_in(k, 0) to the opening draw and fold_in(k, 1) to
the base.

Params {"base": base params}. arch_kwargs nest the base's (arch_name,
arch_kwargs).
"""

from typing import Any, Mapping, Union

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry

ARCH = "opening"
AUX_OPENING = "opening/random"


class OpeningPolicy(types.Policy):
  """Uniform legal moves for the first `num_plies` plies of a hand, then `base`.

  Attributes:
    base: the in-graph JAX policy after the opening.
    num_plies: the opening length, >= 1.
    spec: the PolicySpec.
  """

  def __init__(
      self, base: Union[types.Policy, Mapping[str, Any]], num_plies: int
  ):
    """Wraps `base`.

    Args:
      base: an in-graph JAX policy, or its {"arch_name", "arch_kwargs"}
        record.
      num_plies: the opening length, >= 1.

    Raises:
      ValueError: num_plies is not an int >= 1.
      NotImplementedError: a HOST_DRIVEN or non-JAX base.
    """
    self.base = registry.build_part(base)
    if (
        isinstance(num_plies, bool)
        or int(num_plies) != num_plies
        or int(num_plies) < 1
    ):
      raise ValueError(
          f"num_plies must be an int >= 1 (0: do not wrap), got {num_plies!r}"
      )
    self.num_plies = int(num_plies)
    sb = self.base.spec
    if sb.capability == types.Capability.HOST_DRIVEN or sb.framework != "jax":
      raise NotImplementedError(
          f"OpeningPolicy wraps in-graph JAX policies; {sb.arch_name} is "
          f"{sb.capability.value}/{sb.framework}"
      )
    self.spec = types.PolicySpec(
        game=sb.game,
        obs_format=sb.obs_format,
        num_actions=sb.num_actions,
        num_players=sb.num_players,
        stateful=True,
        capability=sb.capability,
        framework="jax",
        arch_name=ARCH,
        arch_kwargs=dict(
            base=dict(arch_name=sb.arch_name, arch_kwargs=dict(sb.arch_kwargs)),
            num_plies=self.num_plies,
        ),
    )

  @staticmethod
  def make_params(base_params) -> dict[str, Any]:
    """The policy's params around the base's.

    Args:
      base_params: the base policy's params.

    Returns:
      {"base": base_params}.
    """
    return {"base": base_params}

  def init_params(self, key=None):
    """The base's initial params, wrapped.

    Args:
      key: a PRNG key; None uses key 0.

    Returns:
      {"base": base params}.
    """
    return self.make_params(
        self.base.init_params(jax.random.key(0) if key is None else key)
    )

  def encode(self, info):
    """The base's encoding.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      The base's features.
    """
    return self.base.encode(info)

  def init_carry(self, params, batch, seat, key):
    """The base's carry and a ply counter.

    Args:
      params: the params.
      batch: the number of rows B.
      seat: the seat of each row.
      key: a PRNG key.

    Returns:
      {"base": the base's carry, "ply": i32[B] zeros}.
    """
    return {
        "base": self.base.init_carry(params["base"], batch, seat, key),
        "ply": jnp.zeros((batch,), jnp.int64),
    }

  def observe(self, params, carry, step):
    """Advances the base's carry and the hand's ply counter.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].

    Returns:
      The carry after observing `step`.
    """
    reset = jnp.asarray(step.new_hand) | jnp.asarray(step.new_match)
    ply = jnp.where(
        reset, jnp.int64(0), carry["ply"].astype(jnp.int64)
    ) + jnp.int64(1)
    return {
        "base": self.base.observe(params["base"], carry["base"], step),
        "ply": ply,
    }

  def in_opening(self, carry) -> jax.Array:
    """Whether the current ply (already observed) is one of the first num_plies.

    Args:
      carry: the carry.

    Returns:
      bool[B].
    """
    return carry["ply"] <= jnp.int64(self.num_plies)

  def evaluate(self, params, carry, step, key):
    """A uniform legal action in the opening; the base's logits after it.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key (the opening draw and the base's key).

    Returns:
      The PolicyOutput, with aux AUX_OPENING bool[B].

    Raises:
      ValueError: `key` is None, or the base already emits AUX_OPENING.
    """
    if key is None:
      raise ValueError("OpeningPolicy.evaluate needs a key (the opening draw)")
    k_open, k_base = jax.random.fold_in(key, 0), jax.random.fold_in(key, 1)
    out = self.base.evaluate(params["base"], carry["base"], step, k_base)
    legal = jnp.asarray(step.legal_mask, jnp.bool_)
    _, a = legal.shape
    any_legal = jnp.any(legal, axis=-1, keepdims=True)
    draw = jnp.where(
        legal | ~any_legal, jnp.float32(0.0), jnp.float32(-jnp.inf)
    )
    pick = jax.random.categorical(k_open, draw, axis=-1).astype(jnp.int32)
    hot = jnp.arange(a, dtype=jnp.int32)[None, :] == pick[:, None]
    rand_logits = jnp.where(hot, jnp.float32(0.0), jnp.float32(-jnp.inf))
    opening = self.in_opening(carry)
    logits = jnp.where(
        opening[:, None], rand_logits, out.logits.astype(jnp.float32)
    )
    aux = dict(out.aux)
    if AUX_OPENING in aux:
      raise ValueError(f"the base already emits aux {AUX_OPENING!r}")
    aux[AUX_OPENING] = opening
    return types.PolicyOutput(
        logits=logits, value=out.value.astype(jnp.float32), aux=aux
    )


registry.register(ARCH, OpeningPolicy)
