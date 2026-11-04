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

"""Helpers of the remote_bot tests: an artifact, requests, games, factories.

The factory mismatch_bot is importable by name, for a bot server's
`--bot MODULE:FACTORY`.
"""

import json
import os

import numpy as np

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import remote_bot
from open_spiel.python.scale.serve.tests import helpers

GAME = "leduc_poker"
MISMATCH_FACTORY = f"{__name__}:mismatch_bot"


def write_leduc(path):
  """Writes a network leduc artifact (step 3) under `path`.

  Args:
    path: the artifact folder.

  Returns:
    The folder, as a string.
  """
  pol, params = helpers.network(games.env_for(GAME, {}), "mlp", seed=3)
  ckpt.write_artifact(
      str(path),
      3,
      pol,
      params,
      rules_sha="r",
      config_hash="c",
      algo="test",
      env_name=GAME,
  )
  return str(path)


def request_of(game, state, seat, config=None, seed=4):
  """A full_state request body as the GUI's Remote bot builds it.

  Args:
    game: the pyspiel game.
    state: the state.
    seat: the seat to move.
    config: the seat's config.
    seed: the request's seed.

  Returns:
    The body, through a JSON round trip.
  """
  body = {
      "game": str(game),
      "state": state.serialize(),
      "history": [int(a) for a in state.history()],
      "seat": seat,
      "legal_actions": [int(a) for a in state.legal_actions(seat)],
      "seed": seed,
      "game_encoding": {"num_distinct_actions": game.num_distinct_actions()},
      "config": {} if config is None else config,
  }
  return json.loads(json.dumps(body))


def _chance(state, rng):
  """A chance outcome of `state`, sampled.

  Args:
    state: a pyspiel chance state.
    rng: a numpy Generator.

  Returns:
    The action.
  """
  actions, probs = zip(*state.chance_outcomes())
  p = np.asarray(probs, np.float64)
  return int(rng.choice(actions, p=p / p.sum()))


def play(game, choose, *, bot_seat, rng):
  """One game: `choose(state, seat)` for `bot_seat`, uniform random otherwise.

  Args:
    game: the pyspiel game.
    choose: the bot's decision, (state, seat) -> (action, notice).
    bot_seat: the bot's seat.
    rng: a numpy Generator.

  Returns:
    [(history before the move, action, notice)] of the bot's moves.
  """
  s = game.new_initial_state()
  moves = []
  while not s.is_terminal():
    if s.is_chance_node():
      s.apply_action(_chance(s, rng))
      continue
    seat = s.current_player()
    if seat == bot_seat:
      action, notice = choose(s, seat)
      assert action in s.legal_actions(seat), (action, s.history())
      moves.append((list(s.history()), action, notice))
    else:
      action = int(rng.choice(s.legal_actions(seat)))
    s.apply_action(action)
  return moves


class FewerLegal:
  """A pyspiel state whose last legal action (of two or more) is dropped.

  Stands in for a node where the env's rules allow more than OpenSpiel's, so
  checkpoint_bot plays through a legal-set mismatch and says so.
  """

  def __init__(self, state):
    self._state = state

  def legal_actions(self, *unused_args):
    """The real legal actions, the last dropped when there are two or more.

    Args:
      *unused_args: pyspiel's optional player, ignored.

    Returns:
      The actions.
    """
    legal = list(self._state.legal_actions())
    return legal[:-1] if len(legal) > 1 else legal

  def __getattr__(self, name):
    return getattr(self._state, name)

  def __str__(self):
    return str(self._state)


class MismatchBot(remote_bot.RemoteCheckpointBot):
  """RemoteCheckpointBot on states with fewer legal actions (FewerLegal)."""

  def state_of(self, request):
    """The request's game and state, the state wrapped in FewerLegal.

    Args:
      request: the request's body.

    Returns:
      (game, state).
    """
    game, state = super().state_of(request)
    return game, FewerLegal(state)


def mismatch_bot():
  """The factory `--bot <this module>:mismatch_bot` names.

  Returns:
    A MismatchBot of the artifact CHECKPOINT_ENV names.
  """
  return MismatchBot(os.environ[remote_bot.CHECKPOINT_ENV])


