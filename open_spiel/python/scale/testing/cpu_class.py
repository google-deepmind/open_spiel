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

"""The host's CPU class, the thing XLA:CPU floating-point bits depend on.

XLA:CPU compiles for the host it runs on (LLVM's host CPU and its features),
so the last bits of a float result can differ between CPU models: vector
width, FMA contraction and reduction order follow the instruction set. A
golden of XLA:CPU bits recorded on an AVX-512 host (Cascade Lake) does not
reproduce on an AVX2 host (Broadwell), and nothing in the code changed. A
test that pins such bits records the class it was recorded on and is skipped,
naming both classes, on a host of any other class. It never compares bits
from another class.

The class is one deterministic line:

  "<arch> <vendor> family <family> model <model> [<isa flags>]"

read from the first processor of /proc/cpuinfo: the vendor (x86 vendor_id,
or the Arm "CPU implementer"), the family and model numbers (x86 "cpu
family" and "model", or the Arm "CPU architecture" and "CPU part"), and the
members of ISA_FLAGS the CPU lists ("flags" on x86, "Features" on Arm),
sorted. Steppings, clock speeds, cache sizes and core counts are left out:
they do not change the compiled code. A host with no /proc/cpuinfo (macOS)
gets its architecture and platform.processor() with "?" for the rest, so its
class is still deterministic, only coarser.

Every golden of XLA:CPU bits (testing.goldens.two_player, core.tests.
arena_golden) stamps its file with the recording class and asks mismatch()
before comparing; one switch, REQUIRE_CLASS_ENV, turns a mismatch into a
failure for a lane that must compare them.
"""

import os
import platform
from typing import Optional

CPUINFO = "/proc/cpuinfo"

#: Instruction-set flags that change what LLVM emits for f32/f64 math
#: (vector width, FMA, the AVX-512 subsets), in /proc/cpuinfo spelling.
ISA_FLAGS = frozenset(
    # x86
    "sse4_1 sse4_2 avx avx2 fma f16c avx512f avx512dq avx512cd avx512bw"
    " avx512vl avx512ifma avx512vbmi avx512_vnni avx512_bf16 avx512_fp16"
    " amx_tile"
    # Arm
    " asimd asimdhp fphp sve sve2 bf16 i8mm".split()
)


#: Set to "1" in a lane that must compare the XLA:CPU bits of a golden: a host
#: of another CPU class then fails instead of skipping them.
REQUIRE_CLASS_ENV = "OPEN_SPIEL_SCALE_REQUIRE_GOLDEN_CPU_CLASS"


def class_required() -> bool:
  """Whether REQUIRE_CLASS_ENV turns a CPU-class mismatch into a failure.

  Returns:
    True when it is "1", False when it is unset, empty or "0".

  Raises:
    ValueError: any other value.
  """
  value = os.environ.get(REQUIRE_CLASS_ENV, "")
  if value not in ("", "0", "1"):
    raise ValueError(f"{REQUIRE_CLASS_ENV}={value!r}: expected 1, 0 or unset")
  return value == "1"


def mismatch(recorded: Optional[str], golden: str) -> Optional[str]:
  """Why a golden's XLA:CPU bits cannot be compared on this host, or None.

  Args:
    recorded: The CPU class the golden file records (None or empty when it
      records none).
    golden: The golden file's path, for the messages.

  Returns:
    A message naming the recorded and the host CPU class when they differ,
    None when they are the same.

  Raises:
    ValueError: a golden file with no recorded CPU class.
  """
  if not recorded:
    raise ValueError(f"{golden} records no _meta.cpu_class")
  here = host()
  if here == recorded:
    return None
  return (
      f"{os.path.basename(golden)}: the XLA:CPU digests were recorded on "
      "another CPU class and are not portable across CPU models (vector "
      "width, FMA and reduction order change the last bits): recorded on "
      f"[{recorded}], this host is [{here}]. Run on a host of the recorded "
      "class (a GCE n2 of up to 80 vCPUs is Cascade Lake), or re-record "
      "every case on a known-good tree with the golden's --write."
  )


def _first_processor(text: str) -> dict[str, str]:
  """The first processor block of /proc/cpuinfo as {field: value}.

  Args:
    text: The file's contents.

  Returns:
    The fields, keys and values stripped.
  """
  out = {}
  for line in text.splitlines():
    if not line.strip():
      if out:
        break
      continue
    key, sep, value = line.partition(":")
    if sep:
      out.setdefault(key.strip(), value.strip())
  return out


def from_cpuinfo(text: str, arch: str) -> str:
  """The CPU class of a /proc/cpuinfo text.

  Args:
    text: The file's contents.
    arch: The machine architecture (platform.machine()).

  Returns:
    The class line (module doc).
  """
  f = _first_processor(text)

  def pick(*names):
    """The first processor field among `names` with a value.

    Args:
      *names: field names, in order of preference.

    Returns:
      Its value, or "?" when none has one.
    """
    for n in names:
      if f.get(n):
        return f[n]
    return "?"

  flags = set(pick("flags", "Features").split()) & ISA_FLAGS
  return (
      f"{arch} {pick('vendor_id', 'CPU implementer')}"
      f" family {pick('cpu family', 'CPU architecture')}"
      f" model {pick('model', 'CPU part')} [{' '.join(sorted(flags))}]"
  )


def host() -> str:
  """This host's CPU class (module doc).

  Returns:
    The class line.
  """
  arch = platform.machine() or "?"
  if os.path.exists(CPUINFO):
    with open(CPUINFO, encoding="utf-8") as f:
      return from_cpuinfo(f.read(), arch)
  return f"{arch} {platform.processor() or '?'} family ? model ? []"
