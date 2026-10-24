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
"""Importing the package imports no game module and leaves jax's config alone.

The registry names every built-in game without importing it; a game's module
is imported when the game is first loaded or by `load_game_module`. Each
check imports a fresh copy of the package in this process: the package and
its modules are taken out of `sys.modules`, imported again, and put back
afterwards, so the rest of the process keeps the modules it had.
"""

import contextlib
import sys

from absl.testing import absltest
import jax

from open_spiel.python.jax.games import registry

# The package's import name, from its registry module's.
_PKG = registry.__name__.rpartition(".")[0]


def _package_modules():
  """The loaded modules of the package.

  Returns:
    {name: module} for the package and its submodules.
  """
  return {
      k: v
      for k, v in sys.modules.items()
      if k == _PKG or k.startswith(_PKG + ".")
  }


@contextlib.contextmanager
def _fresh_package():
  """Yields a freshly imported copy of the package; restores the old one.

  Yields:
    The fresh package module.
  """
  saved = _package_modules()
  parent, _, leaf = _PKG.rpartition(".")
  for k in saved:
    del sys.modules[k]
  try:
    # An import statement, not a computed name: with the package out of
    # sys.modules it imports the package afresh. Inside the function because
    # it must run after that.
    from open_spiel.python.jax.games import game as fresh_game  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    yield sys.modules[fresh_game.__name__.rpartition(".")[0]]
  finally:
    for k in _package_modules():
      del sys.modules[k]
    sys.modules.update(saved)
    if parent and _PKG in saved:
      setattr(sys.modules[parent], leaf, saved[_PKG])


def _game_modules(reg):
  """The built-in game modules a registry module names.

  Args:
    reg: A registry module (this package's, or a fresh copy's).

  Returns:
    The sorted module names.
  """
  # Reaches private _BUILTIN_GAMES of the code under test.
  # pylint: disable=protected-access
  return sorted({m for _, m, _ in reg._BUILTIN_GAMES})


def _loaded(reg):
  """The game modules of `reg` already imported.

  Args:
    reg: The registry module.

  Returns:
    The imported game module names.
  """
  return [m for m in _game_modules(reg) if f"{_PKG}.{m}" in sys.modules]


class LazyRegistryTest(absltest.TestCase):
  """The game registry imports game modules lazily."""

  def test_importing_the_package_imports_no_game_module(self):
    """Importing the package imports no game module."""
    # Reaches private _BUILTIN_GAMES of the code under test.
    # pylint: disable=protected-access
    with _fresh_package() as pkg:
      self.assertGreater(len(_game_modules(pkg.registry)), 10)
      self.assertEqual(_loaded(pkg.registry), [])
      self.assertEqual(
          [n for n, _, _ in pkg.registry._BUILTIN_GAMES],
          pkg.registered_names(),
      )
      self.assertEqual(_loaded(pkg.registry), [])

  def test_loading_a_game_imports_only_its_module(self):
    """Loading a game, or its module, imports only that module."""
    with _fresh_package() as pkg:
      g = pkg.load_game("tic_tac_toe")
      self.assertEqual(type(g).__module__, f"{_PKG}.tic_tac_toe")
      self.assertEqual(_loaded(pkg.registry), ["tic_tac_toe"])
      hex_module = pkg.load_game_module("hex")
      self.assertIs(hex_module, sys.modules[f"{_PKG}.hex"])
      self.assertIs(pkg.load_game_module("hex"), hex_module)
      self.assertEqual(_loaded(pkg.registry), ["hex", "tic_tac_toe"])

  def test_every_builtin_module_loads(self):
    """load_game_module imports every built-in game module by its name."""
    for module in _game_modules(registry):
      self.assertEqual(
          registry.load_game_module(module).__name__, f"{_PKG}.{module}"
      )

  def test_unknown_modules_raise(self):
    """An unknown game module raises ValueError."""
    with self.assertRaisesRegex(ValueError, "no built-in game module"):
      registry.load_game_module("no_such_game_module")

  def test_importing_the_package_leaves_x64_unchanged(self):
    """Importing the package and loading a game leave x64 as it was."""
    before = jax.config.jax_enable_x64
    with _fresh_package() as pkg:
      pkg.load_game("kuhn_poker")
      self.assertEqual(jax.config.jax_enable_x64, before)


if __name__ == "__main__":
  absltest.main()
