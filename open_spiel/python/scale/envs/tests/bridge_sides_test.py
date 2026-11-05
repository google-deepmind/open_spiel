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

"""Bridge's sides and duplicate pair score (envs.bridge SIDES, imps_pair_score).

* the env's sides are the two partnerships, North-South {0, 2} and
  East-West {1, 3}, zero-sum, and sides_of returns them;
* on random boards a partnership's side return is either seat's return;
* imps_pair_score is bridge.imps of the entrant's point total over the two
  tables, for every IMP band edge, and refuses fractional points;
* declaring them changes neither the rules hash nor the EnvSpec.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import bridge as game
from open_spiel.python.scale.core import sides
from open_spiel.python.scale.envs import bridge
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.testing import infoset_invariance as twins

# envs/tests/bridge_identity_test.py pins the same value.
RULES_SHA = "8f0c3f24a6542b978a082cbe7221ef7f30e85b2d4884acf24867b69139158f7b"


def test_bridge_sides_are_the_partnerships():
  """Bridge sides are the partnerships."""
  env = bridge.BridgeEnv()
  s = sides.sides_of(env)
  assert s is bridge.SIDES
  assert s.seats(0) == (0, 2) and s.seats(1) == (1, 3)
  assert s.constant_sum and s.rotatable and s.num_sides == 2
  assert sides.pair_scorer_of(env) is bridge.imps_pair_score


def test_side_return_is_the_seat_return_on_random_boards():
  """Side return is the seat return on random boards."""
  env = bridge.BridgeEnv()
  states, _, _ = twins.playouts(env, 8, env.max_len, seed=11)
  final = np.stack(
      [np.asarray(env.returns(_last(states, i))) for i in range(8)]
  )
  assert np.any(final != 0.0), "no board scored points; pick another seed"
  side = bridge.SIDES.side_returns(final.astype(np.float64))
  np.testing.assert_array_equal(side[:, 0], final[:, 0])
  np.testing.assert_array_equal(side[:, 0], final[:, 2])
  np.testing.assert_array_equal(side[:, 1], final[:, 1])
  np.testing.assert_array_equal(side.sum(-1), 0.0)


def _last(states, i):
  """Row i's last state of a [B, T] state history.

  Args:
    states: The states, leaves [B, T, ...].
    i: The row.

  Returns:
    The state.
  """
  return jax.tree_util.tree_map(lambda x: x[i, -1], states)


def test_imps_pair_score_equals_bridge_imps():
  """IMPs pair score equals bridge IMPs."""
  floors = np.asarray(game.IMP_BAND_FLOORS)
  diffs = np.unique(
      np.concatenate([
          floors,
          floors - 10,
          floors + 10,
          -floors,
          -floors + 10,
          np.arange(-5000, 5001, 10),
          [0],
      ])
  )
  r0 = diffs.astype(np.float64) - 430.0
  r1 = np.full_like(r0, 430.0)
  got = bridge.imps_pair_score(r0, r1)
  want = np.asarray(game.imps(jnp.asarray(diffs, jnp.int32)), np.float64)
  assert got.dtype == np.float64
  np.testing.assert_array_equal(got, want)
  assert got.min() == -24 and got.max() == 24


def test_imps_pair_score_is_antisymmetric_in_the_tables():
  """IMPs pair score is antisymmetric in the tables."""
  # North-South at table 1 scores 620, at table 2 the other pair stops in 170:
  # the entrant (NS at 1, EW at 2) holds +620 and -170, a 450-point swing = 10
  # IMPs.
  assert float(bridge.imps_pair_score(620.0, -170.0)) == 10.0
  assert float(bridge.imps_pair_score(-620.0, 170.0)) == -10.0
  # The same result at both tables is a push.
  assert float(bridge.imps_pair_score(620.0, -620.0)) == 0.0


def test_imps_pair_score_refuses_fractional_points():
  """IMPs pair score refuses fractional points."""
  with pytest.raises(ValueError, match="whole"):
    bridge.imps_pair_score(np.array([10.5]), np.array([0.0]))
  with pytest.raises(ValueError, match="shapes"):
    bridge.imps_pair_score(np.zeros(2), np.zeros(3))


def test_rules_identity_is_unchanged():
  """Rules identity is unchanged."""
  env = registry.make("bridge")
  assert registry.rules_sha(env) == RULES_SHA == bridge.RULES_SHA
  assert "sides" not in registry.spec_of(env).kwargs


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
