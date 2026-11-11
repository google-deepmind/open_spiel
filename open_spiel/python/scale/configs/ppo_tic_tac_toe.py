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

"""PPO against a uniform-random seat on tic-tac-toe.

recipes.ppo_vs_uniform with a 128-wide MLP at 4096 rows x 71 plies per update,
50 updates, on the OpenSpiel observation tensor. Games run on within a row
across updates, so the segment length is free; 71 plies span several games
(at most 9 moves each) per row per update.
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.league import runner


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.ppo_vs_uniform(
      "ppo_tic_tac_toe",
      "tic_tac_toe",
      "tic_tac_toe_observation_tensor",
      [27],
      9,
      num_updates=50,
      width=128,
      num_steps=71,
  )
