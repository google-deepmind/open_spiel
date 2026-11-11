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

"""The two-player paths reproduce their recorded behaviour bit for bit (CPU).

two_player.json holds the digests of the cases in two_player (sampler tables,
deal keys, Evaluator results, short kuhn Leagues with PSRO, the promotion gate
and self-play), recorded before the N-player work. The digests of the
evaluator and league cases are XLA:CPU bits, so those run on CPU only, and
only on a host of the CPU class the file was recorded on (two_player.META):
XLA:CPU bits are not portable across CPU models, so on any other class they
are skipped with both classes named, never compared (they fail instead when
two_player.REQUIRE_CLASS_ENV is 1, for a lane that must compare them). The
sampler and key cases are numpy and threefry and run anywhere.
"""

import json
import sys

import jax
import pytest

from open_spiel.python.scale.testing.goldens import two_player


@pytest.fixture(name="golden", scope="module")
def _golden_fixture():
  """The recorded goldens.

  Returns:
    {case name: {key: value}}.
  """
  with open(two_player.GOLDEN, encoding="utf-8") as f:
    return json.load(f)


def test_every_case_is_recorded(golden):
  """Every case is recorded, and the CPU class it was recorded on."""
  recorded = set(golden) - {two_player.META}
  assert recorded == set(two_player.cases()), sorted(
      recorded ^ set(two_player.cases())
  )
  assert golden[two_player.META]["cpu_class"], golden[two_player.META]


@pytest.mark.parametrize("name", sorted(two_player.cases()))
def test_case_matches_recorded_digest(golden, name):
  """Case matches recorded digest."""
  on_cpu = jax.devices()[0].platform == "cpu"  # at run time, not import
  if name.startswith(two_player.XLA_CPU_PREFIXES):
    if not on_cpu:
      pytest.skip("digests are XLA:CPU bits")
    mismatch = two_player.class_mismatch(golden)
    if mismatch and two_player.class_required():
      pytest.fail(f"{name}: {two_player.REQUIRE_CLASS_ENV}=1 and {mismatch}")
    if mismatch:
      pytest.skip(f"{name}: {mismatch}")
  got = two_player.run_case(name)
  want = golden[name]
  assert set(got) == set(want), sorted(set(got) ^ set(want))
  bad = sorted(k for k in want if got[k] != want[k])
  assert not bad, (
      f"{name}: {len(bad)} entries differ from the recorded two-player "
      f"behaviour: {[(k, want[k], got[k]) for k in bad[:5]]}"
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
