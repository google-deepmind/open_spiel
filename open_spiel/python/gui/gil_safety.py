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


"""Whether OpenSpiel's C++ search can run on games implemented in Python.

Some C++ algorithms, MCTS among them, release Python's interpreter lock while
they search. On a game implemented in Python the search calls back into
Python, which is safe only if the bindings take the lock again on each call.
Older builds of pyspiel do not, and crash the interpreter on the first call.
"""

from collections.abc import Mapping
import functools
import subprocess
import sys
import threading

import pyspiel

# Runs one MCTS search on a Python game. It crashes the interpreter exactly
# when the installed pyspiel calls Python without holding the lock.
_PROBE = """
import pyspiel
from open_spiel.python.games import tic_tac_toe

game = pyspiel.load_game("python_tic_tac_toe")
bot = pyspiel.MCTSBot(
    game=game,
    evaluator=pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=0),
    uct_c=2.0,
    max_simulations=10,
    max_memory_mb=10,
    solve=False,
    seed=0,
    verbose=False,
)
bot.step(game.new_initial_state())
"""

_PROBE_TIMEOUT_SECONDS = 120

# Lets one caller run the probe while any others wait for its verdict.
_PROBE_LOCK = threading.Lock()


def _implemented_in_python(game: pyspiel.Game) -> bool:
  """Whether `game` itself, not any game it wraps, is implemented in Python."""
  module = type(game).__module__
  return module != pyspiel.__name__ and not module.startswith(
      pyspiel.__name__ + "."
  )


def _nested_games(parameters: Mapping[str, object]) -> list[pyspiel.Game]:
  """Loads every game named by a game-valued parameter in `parameters`."""
  games = []
  for value in parameters.values():
    if isinstance(value, Mapping):  # A game: its name and its parameters.
      nested = dict(value)
      games.append(pyspiel.load_game(str(nested.pop("name")), nested))
  return games


def uses_python(game: pyspiel.Game) -> bool:
  """Whether `game`, or any game it wraps, is implemented in Python.

  A C++ wrapper or transform around a Python game is a C++ object, so its own
  type does not tell; the games it wraps are read from its game-valued
  parameters, at any depth.

  Args:
    game: A loaded game.

  Returns:
    True if `game`, or any game it wraps, is implemented in Python.
  """
  if _implemented_in_python(game):
    return True
  return any(
      uses_python(nested) for nested in _nested_games(game.get_parameters())
  )


@functools.cache
def _probe() -> bool:
  """Runs the probe; see `python_games_are_safe`."""
  try:
    result = subprocess.run(
        # -P: the server's working directory is not put on the module path,
        # so no stray pyspiel.py there can decide the verdict.
        [sys.executable, "-P", "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_SECONDS,
        check=False,
    )
  except (OSError, TypeError):
    return False  # No interpreter could be started to run the probe in.
  except subprocess.TimeoutExpired:
    return False  # A search that never ends is as unsafe as one that crashes.
  if result.returncode == 0:
    return True
  if result.returncode < 0:  # Killed by a signal.
    return False
  raise RuntimeError(
      f"Could not probe pyspiel (exit status {result.returncode}):\n"
      f"{result.stderr}"
  )


def python_games_are_safe() -> bool:
  """Whether the installed pyspiel can search games implemented in Python.

  Runs one search in a child interpreter, once per process, so that a crash
  is observed rather than suffered. The first call blocks while the child
  runs, for up to two minutes; concurrent first callers wait for that one
  child, and every later call returns the cached verdict.

  Returns:
    True if the search completes; False if it crashes the interpreter, or
    has not finished within two minutes, as a search that deadlocks would not.

  Raises:
    RuntimeError: If the child interpreter fails for another reason. That
      failure is not cached, so the next call probes again.
  """
  with _PROBE_LOCK:
    return _probe()
