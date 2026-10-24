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

"""The Arena reproduces its recorded rollouts bit for bit (CPU).

arena_golden.json holds digests of every leaf of (final ArenaState,
TrajectoryBatch, PlyLog) of the cases in arena_golden. Lazy deals, skipped
observes, chunked slot evaluation and sharding change nothing a learner,
evaluator or test can see. The digests are XLA:CPU bits, so the test runs on
CPU only.
"""

import json
import sys

import jax
import pytest

from open_spiel.python.scale.core.tests import arena_golden


@pytest.fixture(name="golden", scope="module")
def _golden_fixture():
  """The recorded digests.

  Returns:
    the GOLDEN digests by case name.
  """
  with open(arena_golden.GOLDEN, encoding="utf-8") as f:
    return json.load(f)


@pytest.mark.parametrize("name", sorted(arena_golden.cases()))
def test_rollout_matches_recorded_digest(golden, name):
  """Case `name`'s rollout matches its recorded digest."""
  # In the test, not a skipif decorator: jax.devices() starts a backend,
  # which importing this module must not do.
  if jax.devices()[0].platform != "cpu":
    pytest.skip("digests are XLA:CPU bits")
  got = arena_golden.run_case(name)
  want = golden[name]
  assert set(got) == set(want), sorted(set(got) ^ set(want))
  bad = sorted(k for k in want if got[k] != want[k])
  assert (
      not bad
  ), f"{name}: {len(bad)} leaves differ from the recorded rollout: {bad[:10]}"


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
