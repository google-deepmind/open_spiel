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

"""Arena shards the batch axis over 4 forced host devices; equals 1 device."""

import os
import sys

import pytest

from open_spiel.python.scale.launch import subprocess_env


def test_four_device_sharding_matches_one_device():
  """A 4-device sharded collect matches 1 device (sharding_probe)."""
  env = {
      "XLA_FLAGS": (
          os.environ.get("XLA_FLAGS", "")
          + " --xla_force_host_platform_device_count=4"
      ).strip(),
      "JAX_PLATFORMS": "cpu",
  }
  r = subprocess_env.run_python(
      [
          "-m",
          subprocess_env.module("core.tests.sharding_probe"),
          "--logtostderr",
      ],
      env=env,
  )
  assert r.returncode == 0, r.stderr[-4000:]
  assert "SHARDING_PROBE_OK" in r.stderr, r.stderr[-4000:]


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
