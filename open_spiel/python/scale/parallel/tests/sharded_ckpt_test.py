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

"""Orbax sharded save on 8 devices, restore and resume on 2 and 1.

The body is probe_ckpt.py.
"""

import shutil
import sys

import pytest

from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.testing import multidevice


def test_save_on_8_devices_restore_and_resume_on_2_and_1(tmp_path):
  """Save on 8 devices, restore and resume on 2 and 1."""
  saved = tmp_path / "saved"
  saved.mkdir()
  out = multidevice.run_probe(
      subprocess_env.module("parallel.tests.probe_ckpt"), 8, "save", str(saved)
  )
  assert "saved on 8 devices: update_step 4" in out, out
  for n in (2, 1):
    copy = tmp_path / f"on{n}"
    shutil.copytree(saved, copy)
    out = multidevice.run_probe(
        subprocess_env.module("parallel.tests.probe_ckpt"),
        n,
        "restore",
        str(copy),
    )
    assert f"restored on {n} devices" in out, out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
