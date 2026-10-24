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

"""Marking plies of a TrajectoryBatch as non-decisions.

A learner trains only on the decisions of the policy it trains. When a
RoutedPolicy hands some plies to a scripted override (for example a game's
mechanical endgame), those plies stay in the batch, so rewards and returns
flow through them, but they must not be trained on.

  batch, n_dropped = decisions.exclude_override(routed_policy, batch)

Only `valid` changes: advantages sum the excluded plies' rewards into the
previous kept decision, as if the override were part of the dynamics.
"""

import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import routed


def exclude_override(
    policy: routed.RoutedPolicy, batch: types.TrajectoryBatch
) -> tuple[types.TrajectoryBatch, np.int64]:
  """(batch with valid &= ~override rows, int64 count of valid rows dropped).

  Raises:
    TypeError: `policy` is not a RoutedPolicy.
    ValueError: the route's mask does not match batch.valid's shape.

  Args:
    policy: the RoutedPolicy that played the batch.
    batch: the TrajectoryBatch.

  Returns:
    (batch with valid &= ~override rows, int64 count of valid rows dropped).
  """
  routed_rows = np.asarray(routed.override_rows(policy, batch.steps.info))
  valid = np.asarray(batch.valid)
  if routed_rows.shape != valid.shape:
    raise ValueError(
        f"route mask {routed_rows.shape} does not match valid {valid.shape}"
    )
  drop = valid & routed_rows
  return (
      batch._replace(valid=jnp.asarray(valid & ~routed_rows)),
      np.int64(drop.sum(dtype=np.int64)),
  )
