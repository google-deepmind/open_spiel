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

"""Behaviour cloning of a scripted Connect Four player.

A 256x2 MLP student learns the prefer-center scripted policy from that
policy's self-play: 512 games of up to 42 plies per update, the first 6 plies
of each game uniform at random, 500 updates.
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.networks import catalog

ENCODER = "connect_four_flat_planes"
NETWORK = "mlp_256x2"
TEACHER = "connect_four.prefer_center"


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  student = common.network_policy_kwargs(
      "connect_four", ENCODER, [126], 7, 2, catalog.get(NETWORK)
  )
  return recipes.behaviour_cloning(
      "bc_connect_four",
      "connect_four",
      student,
      refs.PolicyRef(arch_name=TEACHER),
      batch_size=512,
      num_iterations=500,
      lr=3e-4,
      lr_end=3e-5,
      weight_decay=1e-4,
      max_grad_norm=1.0,
      vf_coef=0.5,
      loss_type="argmax",
      random_opening_plies=6,
      max_game_length=42,
      checkpoint_freq=50,
  )
