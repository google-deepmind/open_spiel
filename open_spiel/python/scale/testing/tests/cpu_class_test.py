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


"""cpu_class reads a CPU class from /proc/cpuinfo; two_player acts on it.

Fixture cpuinfo texts (x86 and Arm), the class mismatch message, the refusal
of a partial --write on another class and the opt-in switch that turns a
mismatch into a failure. No case is run: every path checked here returns or
raises before a digest is computed.
"""

import json
import sys

import pytest

from open_spiel.python.scale.testing import cpu_class
from open_spiel.python.scale.testing.goldens import two_player

CASCADE_LAKE = """\
processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 85
model name\t: Intel(R) Xeon(R) CPU @ 2.80GHz
stepping\t: 7
flags\t\t: fpu sse sse2 sse4_1 sse4_2 avx avx2 fma f16c avx512f avx512dq \
avx512cd avx512bw avx512vl avx512_vnni aes

processor\t: 1
vendor_id\t: OtherVendor
cpu family\t: 99
model\t\t: 1
flags\t\t: fpu
"""

ARM = """\
processor\t: 0
BogoMIPS\t: 2100.00
Features\t: fp asimd evtstrm aes asimdhp fphp sve bf16 i8mm cpuid
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd40
"""

CASCADE_LAKE_CLASS = (
    "x86_64 GenuineIntel family 6 model 85 [avx avx2 avx512_vnni avx512bw"
    " avx512cd avx512dq avx512f avx512vl f16c fma sse4_1 sse4_2]"
)
BROADWELL_CLASS = (
    "x86_64 GenuineIntel family 6 model 79 [avx avx2 f16c fma sse4_1 sse4_2]"
)


def test_x86_class_is_the_first_processor_and_isa_flags():
  """x86: vendor, family, model and the sorted ISA flags of processor 0."""
  assert cpu_class.from_cpuinfo(CASCADE_LAKE, "x86_64") == CASCADE_LAKE_CLASS


def test_arm_class_reads_implementer_architecture_part_and_features():
  """Arm: CPU implementer, architecture and part, and the Features flags."""
  assert cpu_class.from_cpuinfo(ARM, "aarch64") == (
      "aarch64 0x41 family 8 model 0xd40 [asimd asimdhp bf16 fphp i8mm sve]"
  )


def test_missing_fields_are_question_marks():
  """A cpuinfo with none of the fields still gives a deterministic class."""
  assert cpu_class.from_cpuinfo("processor\t: 0\n", "x86_64") == (
      "x86_64 ? family ? model ? []"
  )


def test_same_class_is_no_mismatch(monkeypatch):
  """The recorded class on its own host: no mismatch."""
  monkeypatch.setattr(cpu_class, "host", lambda: CASCADE_LAKE_CLASS)
  golden = {two_player.META: {"cpu_class": CASCADE_LAKE_CLASS}}
  assert two_player.class_mismatch(golden) is None


def test_another_class_names_both(monkeypatch):
  """Another host class: the message names the recorded and the host class."""
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  golden = {two_player.META: {"cpu_class": CASCADE_LAKE_CLASS}}
  msg = two_player.class_mismatch(golden)
  assert f"recorded on [{CASCADE_LAKE_CLASS}]" in msg, msg
  assert f"this host is [{BROADWELL_CLASS}]" in msg, msg


def test_mismatch_names_the_golden_file(monkeypatch):
  """cpu_class.mismatch, shared by every golden, names the file it is for."""
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  msg = cpu_class.mismatch(CASCADE_LAKE_CLASS, "/x/y/some_golden.json")
  assert msg.startswith("some_golden.json: "), msg
  assert cpu_class.mismatch(BROADWELL_CLASS, "/x/some_golden.json") is None
  with pytest.raises(ValueError, match="some_golden.json"):
    cpu_class.mismatch(None, "/x/some_golden.json")


def test_two_player_shares_the_one_switch():
  """two_player's switch is cpu_class's: one variable governs every golden."""
  assert two_player.REQUIRE_CLASS_ENV == cpu_class.REQUIRE_CLASS_ENV
  assert two_player.class_required is cpu_class.class_required


@pytest.mark.parametrize("golden", [{}, {two_player.META: {}}])
def test_no_recorded_class_raises(golden):
  """A golden file with no recorded class is an error, not a pass."""
  with pytest.raises(ValueError, match="cpu_class"):
    two_player.class_mismatch(golden)


@pytest.mark.parametrize(
    ("value", "want"), [(None, False), ("", False), ("0", False), ("1", True)]
)
def test_class_required(monkeypatch, value, want):
  """The opt-in switch: 1 requires the class, unset, empty or 0 do not."""
  if value is None:
    monkeypatch.delenv(two_player.REQUIRE_CLASS_ENV, raising=False)
  else:
    monkeypatch.setenv(two_player.REQUIRE_CLASS_ENV, value)
  assert two_player.class_required() is want


def test_class_required_rejects_other_values(monkeypatch):
  """Any other value of the switch is an error, not a silent skip."""
  monkeypatch.setenv(two_player.REQUIRE_CLASS_ENV, "yes")
  with pytest.raises(ValueError, match=two_player.REQUIRE_CLASS_ENV):
    two_player.class_required()


@pytest.fixture(name="other_class_golden")
def _other_class_golden_fixture(monkeypatch, tmp_path):
  """A golden file recorded on Cascade Lake, read on a Broadwell host.

  Args:
    monkeypatch: pytest's monkeypatch.
    tmp_path: pytest's tmp_path.

  Returns:
    The golden file's path.
  """
  path = tmp_path / "two_player.json"
  path.write_text(
      json.dumps({two_player.META: {"cpu_class": CASCADE_LAKE_CLASS}}),
      encoding="utf-8",
  )
  monkeypatch.setattr(two_player, "GOLDEN", str(path))
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  monkeypatch.delenv(two_player.REQUIRE_CLASS_ENV, raising=False)
  return path


def test_partial_write_on_another_class_is_refused(other_class_golden):
  """--write --only on another class is refused and the file is unchanged."""
  before = other_class_golden.read_text(encoding="utf-8")
  with pytest.raises(ValueError, match="refusing a partial --write"):
    two_player.main(["--write", "--only=league/kuhn/psro_evaluator"])
  assert other_class_golden.read_text(encoding="utf-8") == before


def test_compare_on_another_class_fails_when_required(
    other_class_golden, monkeypatch
):
  """The compare tool refuses another class when the switch is 1."""
  del other_class_golden
  monkeypatch.setenv(two_player.REQUIRE_CLASS_ENV, "1")
  with pytest.raises(ValueError, match=two_player.REQUIRE_CLASS_ENV):
    two_player.main([])


def test_full_write_over_an_unstamped_file_rerecords(monkeypatch, tmp_path):
  """A full --write needs no class in the old file and stamps this host's."""
  path = tmp_path / "two_player.json"
  path.write_text(json.dumps({"a": {"k": 0}}), encoding="utf-8")
  monkeypatch.setattr(two_player, "GOLDEN", str(path))
  monkeypatch.setattr(two_player, "cases", lambda: {"a": None})
  monkeypatch.setattr(two_player, "run_case", lambda name: {"k": 1})
  monkeypatch.setattr(cpu_class, "host", lambda: BROADWELL_CLASS)
  two_player.main(["--write"])
  out = json.loads(path.read_text(encoding="utf-8"))
  assert out["a"] == {"k": 1}, out
  assert out[two_player.META]["cpu_class"] == BROADWELL_CLASS, out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
