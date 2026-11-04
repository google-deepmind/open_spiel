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

"""MMD self-play on Kuhn poker (the env on the JAX games package's kuhn_poker).

recipes.mmd_self_play at 1024 rows x 91 plies per update, 1000 updates, on the
OpenSpiel observation tensor. Hands run on within a row across updates, so the
segment length is free; 91 plies span tens of Kuhn hands (at most 3
decisions each) per row per update.
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.league import runner


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.mmd_self_play(
      "mmd_kuhn_poker",
      "kuhn_poker",
      "kuhn_poker_observation_tensor",
      [7],
      2,
      num_updates=1000,
      num_steps=91,
  )
