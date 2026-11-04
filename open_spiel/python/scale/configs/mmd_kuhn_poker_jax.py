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

r"""MMD self-play on Kuhn poker, with no game package needed.

The quick start: Kuhn poker is implemented inside this package (envs.kuhn_jax),
so this config runs with no game package installed. recipes.mmd_self_play at
256 rows x 16 plies per update, 200 updates.

  python -m open_spiel.python.scale.train --config=mmd_kuhn_poker_jax \
      --checkpoint_dir=/tmp/kuhn
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.league import runner


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.mmd_self_play(
      "mmd_kuhn_poker_jax",
      "kuhn_poker_jax",
      "kuhn_poker_jax_observation_tensor",
      [7],
      2,
      num_updates=200,
      num_envs=256,
      num_steps=16,
  )
