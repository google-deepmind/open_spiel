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

"""Trace-time record of the rows every network call of a learner runs on.

Trace-time record of the rows every network call of a learner runs on
(no learner path calls the network on more rows than the configured chunk
size in one call).

network_rows(policy) intercepts, while tracing, every call of the policy's
flax ActorCritic (nn.intercept_methods: __call__, observe, evaluate_carry,
init_recurrent) and of its encoder, and records each call's leading row
count. It reads shapes only (jax.eval_shape of an update is enough), so it
checks a production-shape update without running or compiling it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import flax.linen as nn
import jax

from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import policy as network_policy

NETWORK_METHODS = ("__call__", "observe", "evaluate_carry", "init_recurrent")


def _rows(tree) -> int:
  """The leading size of the first leaf.

  Args:
    tree: The call's arguments.

  Returns:
    The row count.
  """
  leaves = jax.tree_util.tree_leaves(tree)
  assert leaves, "a network call with no array argument"
  return int(leaves[0].shape[0])


@contextlib.contextmanager
def network_rows(policy) -> Iterator[list[Any]]:
  """Records the rows of every network / encoder call while tracing.

  Args:
    policy: a NetworkPolicy.

  Yields:
    A list that fills with (what, rows) per network / encoder call.

  Raises:
    TypeError: the policy is not a NetworkPolicy.
  """
  if not isinstance(policy, network_policy.NetworkPolicy):
    raise TypeError(
        f"network_rows spies on a NetworkPolicy, got {type(policy).__name__}"
    )
  seen: list[Any] = []

  def _interceptor(next_fun, args, kwargs, context):
    """Records network calls and their row counts.

    Args:
      next_fun: The intercepted method.
      args: Its positional arguments.
      kwargs: Its keyword arguments.
      context: The flax interceptor context.

    Returns:
      The call's output.
    """
    if (
        isinstance(context.module, actor_critic.ActorCritic)
        and context.method_name in NETWORK_METHODS
    ):
      seen.append((f"network.{context.method_name}", _rows(args)))
    return next_fun(*args, **kwargs)

  # Reaches private _encoder of the code under test.
  enc = policy._encoder  # pylint: disable=protected-access

  def _spy_encoder(info):
    """Records the encoder call's row count.

    Args:
      info: The InfoStates.

    Returns:
      The encoded info.
    """
    seen.append(("encoder", _rows(info)))
    return enc(info)

  # Reaches private _encoder of the code under test.
  policy._encoder = _spy_encoder  # pylint: disable=protected-access
  try:
    with nn.intercept_methods(_interceptor):
      yield seen
  finally:
    # Reaches private _encoder of the code under test.
    policy._encoder = enc  # pylint: disable=protected-access
