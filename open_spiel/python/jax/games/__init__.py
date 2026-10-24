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

"""Pure-JAX implementations of OpenSpiel games behind one Game interface.

This package reimplements OpenSpiel games as pure functions over fixed-shape
state pytrees, so thousands of games can run in parallel on an accelerator
inside a single compiled program. Games use OpenSpiel's names, parameters and
action ids, and the pyspiel equivalence tests check them move for move against
OpenSpiel.

  import jax, jax.numpy as jnp
  from open_spiel.python.jax import games  # this package

  game = games.load_game("connect_four(rows=6,columns=7)")
  states = jax.vmap(lambda _: game.new_initial_state())(jnp.arange(1024))
  step = jax.jit(jax.vmap(game.apply_action))
  actions = jax.vmap(lambda s: jnp.argmax(game.legal_actions_mask(s)))(states)
  states = step(states, actions)

Games are looked up by name in a registry. `register_game` adds a game,
`load_game` builds one from an OpenSpiel-style string such as
`"hex(board_size=7)"`, and `load_game_module` imports a built-in game's
module (`load_game_module("hex")`). Importing the package imports no game
module.
"""

from open_spiel.python.jax.games import game as _game
from open_spiel.python.jax.games import registry as _registry

Game = _game.Game
CHANCE_PLAYER_ID = _game.CHANCE_PLAYER_ID
TERMINAL_PLAYER_ID = _game.TERMINAL_PLAYER_ID

load_game = _registry.load_game
load_game_module = _registry.load_game_module
parse_game_string = _registry.parse_game_string
register_game = _registry.register_game
registered_names = _registry.registered_names
