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


r"""MMD self-play on gin rummy, scored against the scripted rule bot.

recipes.mmd_self_play on standard gin (OpenSpiel's gin_rummy rules, the
envs/gin defaults) at 512 rows x 128 plies per update, 1000 updates, on the
OpenSpiel observation tensor (644 features, 241 actions). The network plays
every decision, the knock and layoff phases included. Hands run on within a
row across updates; a hand is far longer than an update's plies.

Every 100 updates (and at the first and the last) the learner plays the
gin_rummy.rule_bot reference evaluation: the scripted rule bot on 256
seat-swapped deal pairs. A smoke run on a laptop CPU:

  python -m open_spiel.python.scale.train --config=mmd_gin_rummy \
      --checkpoint_dir=/tmp/gin \
      --num_updates=2 --num_envs=8 --num_steps=16 --set=eval.deals=4
"""

import dataclasses

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.league import runner

NUM_UPDATES = 1000
EVAL_EVERY = 100
EVAL_DEALS = 256


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  cfg = recipes.mmd_self_play(
      "mmd_gin_rummy",
      "gin_rummy",
      "gin_rummy_observation_tensor",
      [644],
      241,
      num_updates=NUM_UPDATES,
      num_steps=128,
      num_envs=512,
  )
  return dataclasses.replace(
      cfg,
      eval=runner.InLoopEval(
          "gin_rummy.rule_bot", every=EVAL_EVERY, deals=EVAL_DEALS
      ),
  )
