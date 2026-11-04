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

"""The game registry: built-in games by name, loaded on first use.

`register_game` adds a game, `load_game` builds one from an OpenSpiel-style
string such as `"hex(board_size=7)"`, and `load_game_module` imports a
built-in game's module. The package re-exports all four; importing it (or
this module) imports no game module.
"""

from collections.abc import Callable
import types
from typing import Any

from open_spiel.python.jax.games import game as _game

# The package this registry belongs to, for messages.
_PACKAGE = __name__.rpartition(".")[0]

# The built-in games: (registered name, module, constructor). A game's module
# is imported the first time the game is loaded (or by `load_game_module`),
# so importing the package imports none of them.
_BUILTIN_GAMES = (
    ("bargaining", "bargaining", "Bargaining"),
    ("breakthrough", "breakthrough", "Breakthrough"),
    ("bridge", "bridge", "Bridge"),
    ("connect_four", "connect_four", "ConnectFour"),
    ("dark_hex", "dark_hex", "DarkHex"),
    ("dark_hex_ir", "dark_hex_ir", "DarkHexIR"),
    ("gin_rummy", "gin_rummy", "GinRummy"),
    ("gomoku", "gomoku", "Gomoku"),
    ("go", "go", "Go"),
    ("hex", "hex", "Hex"),
    ("kuhn_poker", "kuhn_poker", "KuhnPoker"),
    ("leduc_poker", "leduc_poker", "LeducPoker"),
    ("liars_dice", "liars_dice", "LiarsDice"),
    ("lines_of_action", "lines_of_action", "LinesOfAction"),
    ("no_limit_holdem", "no_limit_holdem", "NoLimitHoldem"),
    ("othello", "othello", "Othello"),
    ("phantom_ttt", "phantom_ttt", "PhantomTicTacToe"),
    ("phantom_ttt_ir", "phantom_ttt_ir", "PhantomTicTacToeIR"),
    ("tic_tac_toe", "tic_tac_toe", "TicTacToe"),
)


def load_game_module(module: str) -> types.ModuleType:
  """Imports one built-in game module by its name in this package.

  `load_game_module("connect_four")` is the package's `connect_four` module,
  imported on first use (the explicit form of `from <package> import
  connect_four`).

  Import statements, not a computed name, so a reader (and an import check)
  sees every module the package can load.

  Args:
    module: The module's name in this package.

  Returns:
    The module.

  Raises:
    ValueError: `module` is not a built-in game module.
  """
  # Lazy by contract, not for a cycle or an optional dependency: importing
  # the package imports no game module (lazy_registry_test checks it).
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  # (ruff PLC0415 is off for this file in ruff.toml for the same reason.)
  if module == "bargaining":
    from open_spiel.python.jax.games import bargaining as mod
  elif module == "breakthrough":
    from open_spiel.python.jax.games import breakthrough as mod
  elif module == "bridge":
    from open_spiel.python.jax.games import bridge as mod
  elif module == "connect_four":
    from open_spiel.python.jax.games import connect_four as mod
  elif module == "dark_hex":
    from open_spiel.python.jax.games import dark_hex as mod
  elif module == "dark_hex_ir":
    from open_spiel.python.jax.games import dark_hex_ir as mod
  elif module == "gin_rummy":
    from open_spiel.python.jax.games import gin_rummy as mod
  elif module == "gomoku":
    from open_spiel.python.jax.games import gomoku as mod
  elif module == "go":
    from open_spiel.python.jax.games import go as mod
  elif module == "hex":
    from open_spiel.python.jax.games import hex as mod
  elif module == "kuhn_poker":
    from open_spiel.python.jax.games import kuhn_poker as mod
  elif module == "leduc_poker":
    from open_spiel.python.jax.games import leduc_poker as mod
  elif module == "liars_dice":
    from open_spiel.python.jax.games import liars_dice as mod
  elif module == "lines_of_action":
    from open_spiel.python.jax.games import lines_of_action as mod
  elif module == "no_limit_holdem":
    from open_spiel.python.jax.games import no_limit_holdem as mod
  elif module == "othello":
    from open_spiel.python.jax.games import othello as mod
  elif module == "phantom_ttt":
    from open_spiel.python.jax.games import phantom_ttt as mod
  elif module == "phantom_ttt_ir":
    from open_spiel.python.jax.games import phantom_ttt_ir as mod
  elif module == "tic_tac_toe":
    from open_spiel.python.jax.games import tic_tac_toe as mod
  else:
    raise ValueError(
        f"no built-in game module {module!r}; built-in game modules: "
        f"{sorted({m for _, m, _ in _BUILTIN_GAMES})}"
    )
  return mod


class _LazyCtor:
  """A built-in game's constructor, imported on first call.

  Attributes:
    attr: The name of the constructor in the module.
    module: The game module's name in this package.
  """

  def __init__(self, module: str, attr: str):
    self.module = module
    self.attr = attr

  def resolve(self) -> Callable[..., _game.Game]:
    """Imports the game module and returns its constructor.

    Returns:
      The game constructor.
    """
    return getattr(load_game_module(self.module), self.attr)

  def __call__(self, **kwargs: Any) -> _game.Game:
    return self.resolve()(**kwargs)

  def __repr__(self) -> str:
    return f"<lazy {_PACKAGE}.{self.module}.{self.attr}>"


_GAMES: dict[str, Callable[..., _game.Game]] = {}


def register_game(
    name: str, ctor: Callable[..., _game.Game], *, replace: bool = False
) -> None:
  """Registers a game constructor under `name`.

  Args:
    name: The short name `load_game` accepts, for example `"kuhn_poker"`.
    ctor: Called with the game's keyword parameters; returns a `Game`.
    replace: Allow replacing an existing registration.

  Raises:
    ValueError: if `name` is already registered and `replace` is False, or
      if `name` contains characters `load_game` cannot parse.
  """
  if not name or any(c in name for c in "(),= "):
    raise ValueError(f"invalid game name {name!r}")
  if name in _GAMES and not replace:
    raise ValueError(f"game {name!r} is already registered")
  _GAMES[name] = ctor


def _parse_value(text: str) -> Any:
  """A game-string value as an int, float, bool or the text itself.

  Args:
    text: The value text.

  Returns:
    The parsed value.
  """
  for cast in (int, float):
    try:
      return cast(text)
    except ValueError:
      pass
  if text.lower() == "true":
    return True
  if text.lower() == "false":
    return False
  return text


def parse_game_string(game_string: str) -> tuple[str, dict[str, Any]]:
  """Splits an OpenSpiel-style game string into a name and parameters.

  `"connect_four(rows=7,columns=8)"` gives `("connect_four", {"rows": 7,
  "columns": 8})`. Values parse as int, then float, then `true`/`false`
  (any case) as bool, else stay strings.

  Args:
    game_string: An OpenSpiel-style game string.

  Returns:
    (name, parameters).

  Raises:
    ValueError: if the string is malformed.
  """
  if "(" not in game_string:
    return game_string, {}
  name, _, rest = game_string.partition("(")
  if not rest.endswith(")"):
    raise ValueError(f"unbalanced parentheses in {game_string!r}")
  body = rest[:-1]
  params = {}
  if body:
    for item in body.split(","):
      key, sep, value = item.partition("=")
      if not sep or not key:
        raise ValueError(f"expected key=value, got {item!r} in {game_string!r}")
      params[key] = _parse_value(value)
  return name, params


def load_game(game_string: str, **kwargs: Any) -> _game.Game:
  """Builds a registered game.

  Args:
    game_string: A name, optionally with parameters: `"hex"` or
      `"hex(board_size=7)"`.
    **kwargs: Parameters, as an alternative to listing them in the string.

  Returns:
    The game.

  Raises:
    ValueError: if the game is unknown, or parameters are given both in the
      string and as keyword arguments.
  """
  name, params = parse_game_string(game_string)
  if name not in _GAMES:
    raise ValueError(
        f"unknown game {name!r}; registered games: {registered_names()}"
    )
  if params and kwargs:
    raise ValueError(
        "game parameters given both in the game string and as keyword arguments"
    )
  return _GAMES[name](**(params or kwargs))


def registered_names() -> list[str]:
  """Returns the registered game names in registration order.

  Returns:
    The names.
  """
  return list(_GAMES)


for _name, _module, _attr in _BUILTIN_GAMES:
  register_game(_name, _LazyCtor(_module, _attr))
del _name, _module, _attr
