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


"""The gin rummy MMD config trains and evaluates as a smoke run.

The gin config trains: mmd_gin_rummy, resized to a smoke, runs two updates
through the trainer's entry point and records its rule-bot evaluation.
"""

import dataclasses
import os
import sys

import pytest

from open_spiel.python.scale import metrics
from open_spiel.python.scale import train
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.league import config_common as common


def test_mmd_gin_rummy_trains_two_updates_and_evaluates(tmp_path):
  """Mmd gin rummy trains two updates and evaluates."""
  frameworks.enable_x64()
  cfg = common.with_overrides(
      common.load_config("mmd_gin_rummy"),
      num_updates=2,
      num_envs=8,
      num_steps=16,
  )
  cfg = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, deals=2))
  out = str(tmp_path / "gin")
  train.run(cfg, checkpoint_dir=out)
  assert os.path.exists(os.path.join(out, "train_state.json"))
  rows = list(
      metrics.read_jsonl(os.path.join(out, "metrics", metrics.JSONL_NAME))
  )
  keys = []
  for r in rows:
    keys.extend(r)
  evals = [k for k in keys if k.startswith("eval/") and k.endswith("/mean")]
  assert evals, sorted(set(keys))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
