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

"""Pytest skips for OpenSpiel games that only an optional dependency builds.

OpenSpiel compiles some games only when an optional dependency is enabled:
universal_poker (and repeated_poker, built on it) needs ACPC, hanabi needs
the Hanabi Learning Environment. A build with DEFAULT_OPTIONAL_DEPENDENCY=OFF
registers none of them, and `pyspiel.load_game` then raises "Unknown game".
A test (or fixture) that needs such a game loads it through `load_game`,
which skips the test, with the reason, on a build without it:

  game = optional_pyspiel_games.load_game(up.GAME_STRING)

Only the games in OPTIONAL_DEPENDENCY can be skipped this way: a missing
game that every OpenSpiel build has is a broken install, and its test fails.
Without pyspiel installed at all the test skips, as pytest.importorskip does.
"""

import pytest

#: The OpenSpiel games that need an optional dependency, and that dependency.
OPTIONAL_DEPENDENCY = {
    "universal_poker": "ACPC",
    "repeated_poker": "ACPC",
    "hanabi": "Hanabi",
}


def _short_name(game_string: str) -> str:
  """The short name of an optional game's string.

  Args:
    game_string: The game string.

  Returns:
    The game's short name.

  Raises:
    ValueError: The game is not an optional dependency game.
  """
  name = game_string.split("(", 1)[0]
  if name not in OPTIONAL_DEPENDENCY:
    raise ValueError(
        f"{name!r} is not an optional OpenSpiel game; known: "
        f"{sorted(OPTIONAL_DEPENDENCY)}"
    )
  return name


def unavailable_reason(game_string: str) -> str:
  """Returns why pyspiel lacks the game, for a skip message.

  Args:
    game_string: The game, with or without parameters.

  Returns:
    The reason.
  """
  name = _short_name(game_string)
  return (
      f"pyspiel built without {OPTIONAL_DEPENDENCY[name]}; {name} unavailable"
  )


def available(game_string: str) -> bool:
  """Returns whether pyspiel registers the game of `game_string`.

  The game is an OPTIONAL_DEPENDENCY game; skips the caller without pyspiel.

  Args:
    game_string: The game, with or without parameters.

  Returns:
    True when pyspiel has the game.
  """
  name = _short_name(game_string)
  pyspiel = pytest.importorskip("pyspiel")
  return name in pyspiel.registered_names()


def require(game_string: str) -> None:
  """Skips the calling test or fixture unless pyspiel has the game.

  Args:
    game_string: The game, with or without parameters.
  """
  if not available(game_string):
    pytest.skip(unavailable_reason(game_string))


def load_game(game_string: str):
  """Returns `pyspiel.load_game(game_string)`, or skips when pyspiel lacks it.

  Args:
    game_string: The game, with or without parameters.

  Returns:
    The pyspiel game.

  Raises:
    ValueError: if the game is not in OPTIONAL_DEPENDENCY.
  """
  require(game_string)
  # Lazy: pyspiel is an optional dependency; require() skipped without it.
  import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return pyspiel.load_game(game_string)
