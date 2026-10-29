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

"""Functions the policy tests name by "module:attr" (routes, index_fns).

They live in a module that is not a test file: a test file may be imported
under a name pytest derives from its path, and a recorded "module:attr" must
name a module that imports under one name wherever it is resolved.
"""

import jax.numpy as jnp


def facing_bet(info):
  """Whether the kuhn seat faces a bet ("b" or "pb").

  Args:
    info: the kuhn InfoState, leaves [B, ...].

  Returns:
    bool[B].
  """
  last = jnp.take_along_axis(
      info.bets, jnp.maximum(info.num_bets - 1, 0)[..., None], axis=-1
  )
  return (info.num_bets >= 1) & (last[..., 0] == 1)


def never(info):
  """False for every row of an array info.

  Args:
    info: an array InfoState [B, ...].

  Returns:
    bool[B] False.
  """
  return jnp.zeros(jnp.shape(info)[:1], jnp.bool_)


def tab_index(info):
  """Table row 1 where the first feature is positive, else 0.

  Args:
    info: an array InfoState [B, D].

  Returns:
    i32[B].
  """
  return (info[:, 0] > 0).astype(jnp.int32)


def first_feature_positive(info):
  """Whether the first obs feature is positive.

  Args:
    info: an ObsInfo, leaves [B, ...].

  Returns:
    bool[B].
  """
  return info.obs[:, 0] > 0


def never_obs(info):
  """False for every row of an ObsInfo.

  Args:
    info: an ObsInfo, leaves [B, ...].

  Returns:
    bool[B] False.
  """
  return jnp.zeros(info.obs.shape[:1], jnp.bool_)


def routed_rows(info):
  """Whether the first feature of info["x"] is positive.

  Args:
    info: a dict InfoState with "x" [B, D].

  Returns:
    bool[B].
  """
  return info["x"][:, 0] > 0


FACING_BET = f"{__name__}:facing_bet"
NEVER = f"{__name__}:never"
TAB_INDEX = f"{__name__}:tab_index"
FIRST_FEATURE_POSITIVE = f"{__name__}:first_feature_positive"
NEVER_OBS = f"{__name__}:never_obs"
ROUTED_ROWS = f"{__name__}:routed_rows"
