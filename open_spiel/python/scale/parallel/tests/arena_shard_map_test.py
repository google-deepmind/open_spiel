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

"""Arena under shard_map over 8 forced host devices (probe_arena.py)."""

import sys

import pytest

from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.testing import multidevice


def test_shard_map_arena_matches_one_device_without_collectives_or_recompiles():
  """The shard_map Arena matches 1 device, no collectives or recompiles."""
  out = multidevice.run_probe(
      subprocess_env.module("parallel.tests.probe_arena"), 8
  )
  for name in ("kuhn_self_play", "kuhn_callback", "leduc_population"):
    assert f"{name}: 8 and 2 devices == 1 device" in out, out
  assert "iterations 2-5 compiled none" in out, out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
