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

"""The output convention of policies that pick exactly one action per row.

A scripted policy (a rule bot, a scripted endgame, a fixed teacher) chooses a
single action rather than a distribution. `one_hot_output` expresses that
choice as logits that every decision rule plays, and poisons the rows the
policy does not cover, so a row routed to the wrong policy fails the Arena's
bad-ply check instead of playing silently.

  def evaluate(self, params, carry, step, key):
    action = my_rule(step.info)  # i32[B]
    return fixed_action.one_hot_output(action, in_domain(step.info),
                                       num_actions, 2)
"""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types


def one_hot_output(
    action, in_domain, num_actions: int, num_players: int
) -> types.PolicyOutput:
  """PolicyOutput playing `action` on every row in the policy's domain.

  Args:
    action: i32[B], the chosen action of each row.
    in_domain: bool[B], the rows the policy covers.
    num_actions: the game's number of actions A.
    num_players: the game's number of players P.

  Returns:
    logits f32[B, A]: 0 at `action` and -inf elsewhere, so argmax and sampling
    both play it; NaN on rows outside the domain. value f32[B, P]: zeros.
    aux {"action": i32[B]}.
  """
  action = jnp.asarray(action, jnp.int32)
  b = action.shape[0]
  hot = jnp.arange(num_actions, dtype=jnp.int32)[None, :] == action[:, None]
  logits = jnp.where(hot, jnp.float32(0.0), jnp.float32(-jnp.inf))
  logits = jnp.where(in_domain[:, None], logits, jnp.float32(jnp.nan))
  return types.PolicyOutput(
      logits=logits.astype(jnp.float32),
      value=jnp.zeros((b, num_players), jnp.float32),
      aux={"action": action},
  )


def empty_features(info) -> jax.Array:
  """f32[B, 0]: the encoding of a policy that reads the InfoState directly.

  Args:
    info: the InfoState, leaves [B, ...].

  Returns:
    f32[B, 0].

  Raises:
    ValueError: the InfoState has no leaves.
  """
  leaves = jax.tree_util.tree_leaves(info)
  if not leaves:
    raise ValueError("empty InfoState")
  return jnp.zeros((leaves[0].shape[0], 0), jnp.float32)
