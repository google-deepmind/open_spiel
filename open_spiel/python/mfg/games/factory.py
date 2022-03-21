# Copyright 2022 DeepMind Technologies Limited
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
"""Factory to create (benchmark) MFG games with different settings."""

from typing import Optional

from absl import logging

from open_spiel.python.games import dynamic_routing_data
import open_spiel.python.mfg.games as games  # pylint: disable=unused-import
from open_spiel.python.mfg.games import crowd_modelling_2d
from open_spiel.python.mfg.games import dynamic_routing
import pyspiel

# For each game, the setting with the game name, e.g. python_mfg_dynamic_routing
# for dynamic routing, denotes the default parameters. Variations are not
# prefixed by the exact game name so that they can be used with different
# implementations, e.g. Python or C++, of the same game. Empty parameters use
# the default values as specified in the game.
GAME_SETTINGS = {
    # 2D crowd modelling game.
    "crowd_modelling_2d_10x10": {},
    "crowd_modelling_2d_four_rooms": {
        **crowd_modelling_2d.FOUR_ROOMS,
        "only_distribution_reward": True,
    },
    "crowd_modelling_2d_maze": {
        **crowd_modelling_2d.MAZE,
        "only_distribution_reward": True,
    },
    # Dynamic routing game.
    "dynamic_routing_braess": {
        "max_num_time_step": 100,
        "network": "braess",
        "time_step_length": 0.05,
    },
    "dynamic_routing_line": {
        "max_num_time_step": 5,
        "network": "line",
        "time_step_length": 1.0,
    },
    "dynamic_routing_sioux_falls_dummy_demand": {
        "max_num_time_step": 81,
        "network": "sioux_falls_dummy_demand",
        "time_step_length": 0.5,
    },
    "dynamic_routing_sioux_falls": {
        # TODO(cabannes): change these values based on experiment output.
        "max_num_time_step": 81,
        "network": "sioux_falls",
        "time_step_length": 0.5,
    },
    # Predator and prey game.
    "predator_prey_5x5x3": {},
}

# Default settings for the games.
GAME_SETTINGS.update({
    "mfg_crowd_modelling_2d": GAME_SETTINGS["crowd_modelling_2d_10x10"],
    "python_mfg_dynamic_routing": GAME_SETTINGS["dynamic_routing_line"],
    "python_mfg_predator_prey": GAME_SETTINGS["predator_prey_5x5x3"],
})

_DYNAMIC_ROUTING_NETWORK = {
    "line": (dynamic_routing_data.LINE_NETWORK,
             dynamic_routing_data.LINE_NETWORK_OD_DEMAND),
    "braess": (dynamic_routing_data.BRAESS_NETWORK,
               dynamic_routing_data.BRAESS_NETWORK_OD_DEMAND),
    "sioux_falls_dummy_demand":
        (dynamic_routing_data.SIOUX_FALLS_NETWORK,
         dynamic_routing_data.SIOUX_FALLS_DUMMY_OD_DEMAND),
    "sioux_falls": (dynamic_routing_data.SIOUX_FALLS_NETWORK,
                    dynamic_routing_data.SIOUX_FALLS_OD_DEMAND)
}


def create_game_with_setting(game_name: str,
                             setting: Optional[str] = None) -> pyspiel.Game:
  """Creates an OpenSpiel game with the specified setting.

  Args:
    game_name: Name of a registered game, e.g. mfg_crowd_modelling_2d.
    setting: Name of the pre-defined setting. If None, game_name will be used
      instead. The setting should be present in the GAME_SETTINGS map above.

  Returns:
    a Game.
  """
  setting = setting or game_name
  params = GAME_SETTINGS.get(setting)
  if params is None:
    raise ValueError(f"{setting} setting does not exist for {game_name}.")

  logging.info("Creating %s game with parameters: %r", game_name, params)

  # Dynamic routing game requires setting the network and demand explicitly.
  if game_name == "python_mfg_dynamic_routing":
    # Create a copy since we modify it below removing the network key.
    params = params.copy()
    network = params.pop("network")
    network, od_demand = _DYNAMIC_ROUTING_NETWORK[network]
    return dynamic_routing.MeanFieldRoutingGame(
        params, network=network, od_demand=od_demand)

  return pyspiel.load_game(game_name, params)
