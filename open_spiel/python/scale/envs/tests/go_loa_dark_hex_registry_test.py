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

"""go, lines_of_action, dark_hex, dark_hex_ir in the game table.

Defaults (go 9x9, komi 7.5; lines_of_action pyspiel's 1000-move limit),
canonical specs that round-trip through make / spec_of / JSON, refused options,
and rules identity per option.
"""

import json
import sys

import pytest

from open_spiel.python.scale.envs import registry


def test_rows_and_defaults():
  """Rows and defaults."""
  assert registry.game("go").default_kwargs == {"board_size": 9, "komi": 7.5}
  assert registry.game("lines_of_action").default_kwargs == {
      "max_game_length": 1000
  }
  for name in ("dark_hex", "dark_hex_ir"):
    assert not registry.game(name).default_kwargs
  for name in ("go", "lines_of_action", "dark_hex", "dark_hex_ir"):
    assert name in registry.builtin_names() and registry.game(name).served


@pytest.mark.parametrize(
    "name,kw",
    [
        ("go", {"board_size": 13, "komi": 6}),
        ("go", {"board_size": 19}),
        ("lines_of_action", {"max_game_length": 200}),
        ("dark_hex_ir", {}),
    ],
)
def test_non_default_specs_round_trip(name, kw):
  """Non default specs round trip.

  Args:
    name: The parametrized case.
    kw: The parametrized case.
  """
  s = registry.spec(name, **kw)
  env = registry.make(s)
  assert registry.spec_of(env) == s
  assert registry.EnvSpec.from_json(json.loads(json.dumps(s.to_json()))) == s


def test_go_komi_is_canonical_float_and_options_validated():
  """Go komi is canonical float and options validated."""
  assert registry.spec("go", komi=6).kwargs["komi"] == 6.0
  assert registry.spec("go", komi=6) == registry.spec("go", komi=6.0)
  with pytest.raises(TypeError, match="board_size"):
    registry.spec("go", board_size=9.0)
  with pytest.raises(ValueError, match="board_size"):
    registry.spec("go", board_size=1)
  with pytest.raises(TypeError, match="max_game_length"):
    registry.spec("lines_of_action", max_game_length="1000")
  with pytest.raises(ValueError, match="max_game_length"):
    registry.spec("lines_of_action", max_game_length=0)
  with pytest.raises(TypeError, match="unknown env option"):
    registry.spec("dark_hex", board_size=3)


def test_rules_sha_separates_options_and_games():
  """Rules sha separates options and games."""
  shas = {
      registry.rules_sha(registry.make(n, **kw))
      for n, kw in (
          ("go", {}),
          ("go", {"board_size": 13}),
          ("go", {"komi": 6.5}),
          ("lines_of_action", {}),
          ("lines_of_action", {"max_game_length": 200}),
          ("dark_hex", {}),
          ("dark_hex_ir", {}),
      )
  }
  assert len(shas) == 7


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
