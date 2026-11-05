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

"""Train your own hearts or euchre agent: the package docs' smoke command.

For each game, the trainer runs the open card move transformer config
(hearts_move_transformer_self_play, euchre_move_transformer_self_play) as the
package docs' smoke command does: the full-size network, two updates on a CPU,
written to a checkpoint folder whose step-2 artifact is checked to exist.
About 90 s per game on 8 CPU cores.
"""

import os
import sys

import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale.launch import subprocess_env

pytest.importorskip(deps.GAMES_PACKAGE)

UPDATES = 2
#: The docs' CPU smoke flags (and each config's module doc).
SMOKE_FLAGS = (
    f"--num_updates={UPDATES}",
    "--num_envs=8",
    "--num_steps=16",
    "--set=learner.learner.minibatch_size=64",
    "--set=learner.learner.sgd_steps=2",
    "--set=eval.0.deals=4",
)
CONFIGS = {
    "hearts": "hearts_move_transformer_self_play",
    "euchre": "euchre_move_transformer_self_play",
}


@pytest.mark.parametrize("game", sorted(CONFIGS))
def test_train_two_updates_writes_a_checkpoint(game, tmp_path):
  ckpt = str(tmp_path / f"{game}_ckpt")
  r = subprocess_env.run_python(
      [
          "-m",
          subprocess_env.module("train"),
          f"--config={CONFIGS[game]}",
          f"--checkpoint_dir={ckpt}",
          *SMOKE_FLAGS,
      ],
      env={"JAX_PLATFORMS": "cpu"},
      timeout=1200,
  )
  assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
  assert os.path.isdir(os.path.join(ckpt, str(UPDATES))), os.listdir(ckpt)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
