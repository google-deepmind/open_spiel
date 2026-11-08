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
CPU only, and the network cases (arena_golden.NETWORK_CASES) only on a host of
the CPU class the file was recorded on: on another class they are skipped with
both classes named, never compared (they fail instead under
cpu_class.REQUIRE_CLASS_ENV=1, for a lane that must compare them).
"""

import json
import sys

import jax
import pytest

from open_spiel.python.scale.core.tests import arena_golden
from open_spiel.python.scale.testing import cpu_class


@pytest.fixture(name="golden", scope="module")
def _golden_fixture():
  """The recorded digests.

  Returns:
    the GOLDEN digests by case name.
  """
  with open(arena_golden.GOLDEN, encoding="utf-8") as f:
    return json.load(f)


def test_every_case_is_recorded_with_its_cpu_class(golden):
  """Every case is recorded, and so is the CPU class it was recorded on."""
  recorded = set(golden) - {arena_golden.META}
  assert recorded == set(arena_golden.cases()), sorted(
      recorded ^ set(arena_golden.cases())
  )
  assert arena_golden.NETWORK_CASES <= recorded, sorted(
      arena_golden.NETWORK_CASES - recorded
  )
  assert golden[arena_golden.META]["cpu_class"], golden[arena_golden.META]


@pytest.mark.parametrize("name", sorted(arena_golden.cases()))
def test_rollout_matches_recorded_digest(golden, name):
  """Case `name`'s rollout matches its recorded digest."""
  # In the test, not a skipif decorator: jax.devices() starts a backend,
  # which importing this module must not do.
  if jax.devices()[0].platform != "cpu":
    pytest.skip("digests are XLA:CPU bits")
  if name in arena_golden.NETWORK_CASES:
    mismatch = arena_golden.class_mismatch(golden)
    if mismatch and cpu_class.class_required():
      pytest.fail(f"{name}: {cpu_class.REQUIRE_CLASS_ENV}=1 and {mismatch}")
    if mismatch:
      pytest.skip(f"{name}: {mismatch}")
  got = arena_golden.run_case(name)
  want = golden[name]
  assert set(got) == set(want), sorted(set(got) ^ set(want))
  bad = sorted(k for k in want if got[k] != want[k])
  assert (
      not bad
  ), f"{name}: {len(bad)} leaves differ from the recorded rollout: {bad[:10]}"


CASCADE_LAKE_CLASS = (
    "x86_64 GenuineIntel family 6 model 85 [avx avx2 avx512_vnni avx512bw"
    " avx512cd avx512dq avx512f avx512vl f16c fma sse4_1 sse4_2]"
)
BROADWELL_CLASS = (
    "x86_64 GenuineIntel family 6 model 79 [avx avx2 f16c fma sse4_1 sse4_2]"
)


def test_the_recorded_class_is_cascade_lake(golden):
  """The file is stamped with the class its digests were checked on (n2)."""
  assert golden[arena_golden.META]["cpu_class"] == CASCADE_LAKE_CLASS


def test_another_class_names_the_file_and_both_classes(golden, monkeypatch):
  """On another class the reason names the golden and both classes."""
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  msg = arena_golden.class_mismatch(golden)
  assert "arena_golden.json" in msg, msg
  assert f"recorded on [{CASCADE_LAKE_CLASS}]" in msg, msg
  assert f"this host is [{BROADWELL_CLASS}]" in msg, msg
  monkeypatch.setattr(cpu_class, "host", lambda: CASCADE_LAKE_CLASS)
  assert arena_golden.class_mismatch(golden) is None


@pytest.fixture(name="other_class_golden")
def _other_class_golden_fixture(monkeypatch, tmp_path):
  """A golden file recorded on Cascade Lake, read on a Broadwell host.

  Args:
    monkeypatch: pytest's monkeypatch.
    tmp_path: pytest's tmp_path.

  Returns:
    The golden file's path.
  """
  path = tmp_path / "arena_golden.json"
  path.write_text(
      json.dumps({arena_golden.META: {"cpu_class": CASCADE_LAKE_CLASS}}),
      encoding="utf-8",
  )
  monkeypatch.setattr(arena_golden, "GOLDEN", str(path))
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  monkeypatch.delenv(cpu_class.REQUIRE_CLASS_ENV, raising=False)
  return path


def test_partial_write_on_another_class_is_refused(other_class_golden):
  """--write --only on another class is refused and the file is unchanged."""
  before = other_class_golden.read_text(encoding="utf-8")
  with pytest.raises(ValueError, match="refusing a partial --write"):
    arena_golden.main(["--write", "--only=leduc_network_self_play"])
  assert other_class_golden.read_text(encoding="utf-8") == before


def test_compare_on_another_class_fails_when_required(
    other_class_golden, monkeypatch
):
  """The compare tool refuses another class when the switch is 1."""
  del other_class_golden
  monkeypatch.setenv(cpu_class.REQUIRE_CLASS_ENV, "1")
  with pytest.raises(ValueError, match=cpu_class.REQUIRE_CLASS_ENV):
    arena_golden.main([])


def test_full_write_over_an_unstamped_file_rerecords(monkeypatch, tmp_path):
  """A full --write needs no class in the old file and stamps this host's."""
  path = tmp_path / "arena_golden.json"
  path.write_text(
      json.dumps({"a": {"k": 0}, arena_golden.META: {}}), encoding="utf-8"
  )
  monkeypatch.setattr(arena_golden, "GOLDEN", str(path))
  monkeypatch.setattr(arena_golden, "cases", lambda: {"a": None})
  monkeypatch.setattr(arena_golden, "run_case", lambda name: {"k": 1})
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  arena_golden.main(["--write"])
  out = json.loads(path.read_text(encoding="utf-8"))
  assert out["a"] == {"k": 1}, out
  assert out[arena_golden.META]["cpu_class"] == BROADWELL_CLASS, out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
