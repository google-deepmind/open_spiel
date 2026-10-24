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

"""Pyspiel states to env raw histories, and the replay-reached-node check.

A serving request carries a pyspiel state: its full action history, chance
outcomes included. An env replays raw histories through its raw layer
(`root`, `apply`, `apply_chance`). For most games the two histories are the
same, because the game envs (most built on the pure-JAX OpenSpiel games that
ship beside this package) use pyspiel's chance-outcome ids. A game whose
histories differ names a translator in its GameSpec
(`pyspiel_history_to_raw`), which this module applies.

  hands = wire.pyspiel_history_to_raw("gin_rummy", state.history(), params)
  wire.check_replayed_node(env, env_state, state)

`check_replayed_node` runs on every request: after the replay, the env's
actor, legal set and terminal flag must equal pyspiel's, or the request is
refused rather than answered at the wrong node (a different rulebook, a wrong
seat, a wrong translation).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from open_spiel.python.scale import registry as path_registry
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.serve import replay


def game_params(game: Any) -> dict[str, Any]:
  """A pyspiel game's parameters as a plain dict.

  Args:
    game: the pyspiel game.

  Returns:
    The parameters.
  """
  return dict(game.get_parameters())


def _translator(game_name: str):
  """The game's pyspiel-history translator, if it has one.

  Args:
    game_name: The game.

  Returns:
    The translator, or None.
  """
  path = getattr(env_registry.game(game_name), "pyspiel_history_to_raw", None)
  if path is None:
    return None
  return path_registry.load_object(path)


def pyspiel_history_to_raw(
    game_name: str, history: Sequence[int], params: Mapping[str, Any]
) -> list[list[int]]:
  """The hands to replay for a pyspiel history of `game_name`.

  A pyspiel state is one hand, so the result holds one raw history.

  Args:
    game_name: the pyspiel game's short name.
    history: the pyspiel history.
    params: the game's parameters.

  Returns:
    [raw history].
  """
  fn = _translator(game_name)
  hist = [int(a) for a in history]
  return [hist if fn is None else [int(a) for a in fn(hist, params)]]


def hands_from_state(state: Any) -> list[list[int]]:
  """`pyspiel_history_to_raw` of a pyspiel state.

  Args:
    state: the pyspiel state.

  Returns:
    [raw history].
  """
  game = state.get_game()
  return pyspiel_history_to_raw(
      game.get_type().short_name, state.history(), game_params(game)
  )


def _node_fn(env: types.Env):
  """(actor, legal mask, terminal) of an env state, jitted once per env.

  serve.replay's cache, which lets the env go when nothing else holds it.

  Args:
    env: the env.

  Returns:
    The jitted function of an env state.
  """
  return replay.decision_node_fn(env)


def check_replayed_node(env: types.Env, env_state: Any, state: Any) -> None:
  """Checks that the replayed env node is pyspiel's node.

  Same terminal flag, same actor, same legal action set.

  Args:
    env: the env.
    env_state: the replayed env state.
    state: the pyspiel state.

  Raises:
    ValueError: the nodes differ.
  """
  if state.is_chance_node():
    raise ValueError(
        "the pyspiel state is a chance node; there is no decision to serve"
    )
  actor, legal, terminal = (np.asarray(x) for x in _node_fn(env)(env_state))
  if bool(terminal) != bool(state.is_terminal()):
    raise ValueError(
        f"replay: env terminal {bool(terminal)} != pyspiel terminal "
        f"{state.is_terminal()} after {len(state.history())} actions"
    )
  if state.is_terminal():
    return
  want_actor = int(state.current_player())
  if int(actor) != want_actor:
    raise ValueError(
        f"replay: env actor {int(actor)} != pyspiel current_player "
        f"{want_actor} "
        f"after {len(state.history())} actions"
    )
  want = np.zeros(env.num_actions, np.bool_)
  la = state.legal_actions()
  if la and max(la) >= env.num_actions:
    raise ValueError(
        f"replay: pyspiel legal action {max(la)} >= env num_actions "
        f"{env.num_actions}"
    )
  want[la] = True
  if not np.array_equal(legal.astype(np.bool_), want):
    raise ValueError(
        f"replay: env legal set {np.flatnonzero(legal).tolist()} != pyspiel "
        f"legal set {la} after {len(state.history())} actions"
    )
