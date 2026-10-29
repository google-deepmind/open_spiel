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

"""Example training configs, one module per run.

Each module defines `get_config()`, which returns a
`league.runner.LeagueRunConfig`; the trainer runs it
(`python -m open_spiel.python.scale.train --config=<name>`). Packages registered
with `league.config_common.register_config_package` add more names.

  mmd_kuhn_poker_jax   MMD self-play on Kuhn poker; needs no game package
  mmd_kuhn_poker       MMD self-play on Kuhn poker
  mmd_leduc_poker      MMD self-play on Leduc poker
  mmd_gin_rummy        MMD self-play on gin rummy, scored against the rule bot
  ppo_tic_tac_toe      PPO against a uniform-random seat
  az_connect_four      AlphaZero
  bc_connect_four      behaviour cloning of a scripted player
  psro_connect_four    PSRO with PPO best responses
"""
