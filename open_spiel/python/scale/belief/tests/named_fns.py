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

"""Functions the belief tests name by "module:attr" (index_fns).

They live in a module that is not a test file: a test file may be imported
under a name pytest derives from its path, and a recorded "module:attr" must
name a module that imports under one name wherever it is resolved.
"""

import jax.numpy as jnp

RAISE = 2  # leduc's raise action


def leduc_index_fn(info):
  """The opponent-table row of each Leduc InfoState.

  Returns:
    i32[B]: rank(card) * 2 + (a raise in the current round's sequence).

  Args:
    info: a batch of Leduc InfoStates.
  """
  seq = jnp.where((info.round == 1)[:, None], info.round1, info.round2)
  facing = jnp.any(seq == RAISE, axis=-1)
  return (jnp.maximum(info.card, 0) // 2 * 2 + facing.astype(jnp.int32)).astype(
      jnp.int32
  )


LEDUC_INDEX_FN = f"{__name__}:leduc_index_fn"
