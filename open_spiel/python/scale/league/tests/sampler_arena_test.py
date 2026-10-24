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

"""The sampler's draw through the one Arena.

The sampler's draw through the one Arena (moved out of sampler_test, which
is jax-free since the portability change): the drawn table and the pools'
params run as is, and every opponent ply names its member.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.policies import tabular


def _pol():
  """A Kuhn tabular policy.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy(
      "kuhn_poker", 12, 2, 2, f"{kuhn.__name__}:info_index"
  )


def test_drawn_table_runs_in_the_arena_with_known_members():
  """A drawn table runs in the Arena with known members.

  Known answer through the one Arena: member m of the pool always plays
  action m (logits one-hot), so the PlyLog of every opponent ply names it.
  """
  pol = _pol()
  k = 3
  logits = np.full((k, 12, 2), -20.0, np.float32)
  logits[0, :, 0] = 20.0  # member 0 always passes
  logits[1, :, 1] = 20.0  # member 1 always bets
  logits[2, :, 0] = 20.0
  pool = sampler.Pool(
      policy=pol,
      rule=types.ARGMAX,
      names=("pass@0", "bet@0", None),
      params={"logits": jnp.asarray(logits)},
  )
  st = sampler.LeagueSlots(
      num_players=2,
      batch=16,
      pools=(pool,),
      mix=sampler.Mix(0.0, {"pass@0": 0.5, "bet@0": 0.5}),
  )
  d = sampler.sample(st, jax.random.key(1))
  env = kuhn.KuhnEnv()
  arena = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.SAMPLE),
          arena_lib.Slot(pol, k, types.ARGMAX),
      ],
  )
  learner = {"logits": jnp.zeros((1, 12, 2), jnp.float32)}
  dk, pk = arena_lib.independent_keys(jax.random.key(2), 16, 2)
  _, _, log = arena.collect([learner, *d.params], d.table, dk, pk, 8)
  slot, member, action = (
      np.asarray(x) for x in (log.slot, log.member, log.action)
  )
  opp_plies = slot == 1
  assert opp_plies.any()
  np.testing.assert_array_equal(action[opp_plies], member[opp_plies])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
