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

"""breakthrough / gomoku / hex / othello in the one game table.

Specs with their size options (ints only), make / spec_of round trips, rules_sha
separates sizes, and serving maps pyspiel parameters to env options and refuses
variants the envs do not play.
"""

import sys

import pytest

from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.serve import games

BOARD = {
    "breakthrough": {"rows": 8, "columns": 8},
    "gomoku": {"board_size": 15, "connect": 5},
    "hex": {"board_size": 11},
    "othello": {},
}


@pytest.mark.parametrize("name", sorted(BOARD))
def test_registered_served_with_defaults(name):
  """Registered served with defaults.

  Args:
    name: The parametrized case.
  """
  assert name in registry.builtin_names()
  assert name in games.served_games()
  assert dict(registry.game(name).default_kwargs) == BOARD[name]
  env = registry.make(name)
  assert registry.spec_of(env) == registry.spec(name)


def test_size_options_round_trip_and_change_the_rules_sha():
  """Size options round trip and change the rules sha."""
  for name, kw in (
      ("breakthrough", {"rows": 6, "columns": 5}),
      ("gomoku", {"board_size": 9, "connect": 4}),
      ("hex", {"board_size": 7}),
  ):
    s = registry.spec(name, **kw)
    env = registry.make(s)
    assert registry.spec_of(env) == s
    assert registry.EnvSpec.from_json(s.to_json()) == s
    assert registry.rules_sha(env) != registry.rules_sha(registry.make(name))


def test_options_are_ints():
  """Options are ints."""
  with pytest.raises(TypeError, match="hex board_size must be an int"):
    registry.spec("hex", board_size=7.0)
  with pytest.raises(TypeError, match="gomoku connect must be an int"):
    registry.spec("gomoku", connect=True)
  with pytest.raises(TypeError, match="unknown env option"):
    registry.spec("othello", board_size=8)
  with pytest.raises(ValueError, match="connect"):
    registry.make("gomoku", board_size=4)


def test_served_parameters():
  """Served parameters."""
  assert registry.spec_of(
      games.env_for("breakthrough", {"rows": 8, "columns": 8})
  ) == registry.spec("breakthrough")
  assert registry.spec_of(
      games.env_for("breakthrough", {"rows": 6, "columns": 5})
  ) == registry.spec("breakthrough", rows=6, columns=5)
  assert registry.spec_of(
      games.env_for(
          "gomoku",
          {"anti": False, "connect": 4, "dims": 2, "size": 7, "wrap": False},
      )
  ) == registry.spec("gomoku", board_size=7, connect=4)
  # pyspiel resolves hex(board_size=5) to num_rows = num_cols = 5, and
  # hex(num_cols=5,num_rows=5) to board_size 11 with num_rows = num_cols = 5.
  for params in (
      {
          "board_size": 5,
          "num_cols": 5,
          "num_rows": 5,
          "string_rep": "standard",
          "swap": False,
      },
      {"board_size": 11, "num_cols": 5, "num_rows": 5},
      {"board_size": 5},
  ):
    assert registry.spec_of(games.env_for("hex", params)) == registry.spec(
        "hex", board_size=5
    )
  assert registry.spec_of(games.env_for("othello", {})) == registry.spec(
      "othello"
  )


def test_served_variants_not_played_are_refused():
  """Served variants not played are refused."""
  with pytest.raises(ValueError, match="square"):
    games.env_for("hex", {"board_size": 11, "num_cols": 5, "num_rows": 11})
  with pytest.raises(ValueError, match="swap"):
    games.env_for("hex", {"swap": True})
  with pytest.raises(ValueError, match="anti"):
    games.env_for("gomoku", {"anti": True})
  with pytest.raises(ValueError, match="dims"):
    games.env_for("gomoku", {"dims": 3})
  with pytest.raises(ValueError, match="wrap"):
    games.env_for("gomoku", {"wrap": True})
  with pytest.raises(ValueError, match="not implemented"):
    games.env_for("othello", {"board_size": 6})
  with pytest.raises(ValueError, match="rows"):
    games.env_for("breakthrough", {"rows": 5, "columns": 8})


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
