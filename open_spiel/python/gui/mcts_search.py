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


"""The MCTS controller's search: OpenSpiel's C++ search, or its Python one.

A game implemented in C++ is searched by OpenSpiel's C++ MCTS
(`pyspiel.MCTSBot`, `cpp_search`), with its tree capped at `MAX_MEMORY_MB`
megabytes. That search cannot be stopped once started, and its only bounds
are its simulation count and a wall-clock time. Some pyspiel builds stop it
at whichever of the two comes first; others, OpenSpiel's releases so far
among them, let a wall-clock time replace the count. The server tells them
apart once, at startup (`probe_limits`, cached by `limits`), and
`mcts_controller` sets the search's limits by the answer: the count with a
wall-clock backstop, or a capped count and no clock.

A game implemented in Python, or a C++ game that wraps one
(`implemented_in_python`), is searched by OpenSpiel's Python MCTS
(`open_spiel.python.algorithms.mcts`, `python_search`): the C++ search
releases Python's interpreter lock while it runs, and calling back into a
Python game without it is unsafe. Its evaluator, `Rollouts`, plays each
leaf's random rollout in C++ and checks the move budget (`move_budget.check`)
first, so a search that overruns its budget stops at its next simulation and
the thread it ran on exits. Its tree is bounded too: a simulation that
reaches an unexpanded node adds one node per legal action (or chance outcome)
there, and a search stops before its tree would hold more than `NODE_BUDGET`
nodes (about 180 MB of Python objects) and plays the best move found so far.
The budget counts nodes, not bytes or seconds, so the move stays reproducible
from the seed on any machine.
"""

from collections.abc import Mapping
import dataclasses
import functools
import random
import time

import numpy as np
import pyspiel

from open_spiel.python.algorithms import mcts
from open_spiel.python.gui import contract
from open_spiel.python.gui import move_budget

_SEED_BOUND = 2**31

# The most nodes a Python search tree may hold; see the module docstring. A
# node costs about 180 bytes.
NODE_BUDGET = 1_000_000

# The most memory a C++ search tree may take, in megabytes.
MAX_MEMORY_MB = 1000

# The probe of the C++ search's limits (`probe_limits`): a search of this
# game's initial position, given both this many simulations and this
# wall-clock time. A search that honours the count stops after it, in well
# under a millisecond; one whose clock replaces the count runs the full time,
# many thousands of simulations.
PROBE_GAME = "tic_tac_toe"
PROBE_SIMULATIONS = 5
PROBE_SECONDS = 0.2


class _NodeBudgetReachedError(Exception):
  """The search tree is full: one more expansion would exceed its budget."""


class Rollouts:
  """An MCTS evaluator: one uniformly random rollout per leaf, run in C++.

  Both calls check the move budget first, and the search calls one or the
  other in every simulation that does not end at a terminal state. The
  rollouts are played by `pyspiel.evaluate_bots`, which holds Python's
  interpreter lock, so a game implemented in Python is called back safely.

  `prior` is called once per node the search expands, and the search makes
  one child node per outcome it returns, so it also counts the tree's nodes
  and refuses to grow the tree past its node budget.
  """

  def __init__(self, num_players: int, seed: int, node_budget: int) -> None:
    if node_budget < 1:
      raise ValueError(f"The node budget must be positive, not {node_budget}.")
    self._num_players = num_players
    self._rng = random.Random(seed)
    self._node_budget = node_budget
    self.nodes = 1  # The root.

  def evaluate(self, state: pyspiel.State) -> list[float]:
    """Returns every player's return from one random rollout of `state`.

    Raises:
      move_budget.StoppedError: If the move budget has expired.
    """
    move_budget.check()
    bots = [
        pyspiel.make_uniform_random_bot(player, self._seed())
        for player in range(self._num_players)
    ]
    return list(pyspiel.evaluate_bots(state.clone(), bots, self._seed()))

  def prior(self, state: pyspiel.State) -> list[tuple[int, float]]:
    """Returns the chance outcomes at `state`, or its legal actions, uniform.

    The root is always expanded, so that the search has a move to play.

    Raises:
      move_budget.StoppedError: If the move budget has expired.
      _NodeBudgetReachedError: If the children of `state` would take the tree
        past its node budget.
    """
    move_budget.check()
    if state.is_chance_node():
      outcomes = list(state.chance_outcomes())
    else:
      legal = state.legal_actions(state.current_player())
      outcomes = [(action, 1.0 / len(legal)) for action in legal]
    root = self.nodes == 1
    if not root and self.nodes + len(outcomes) > self._node_budget:
      raise _NodeBudgetReachedError()
    self.nodes += len(outcomes)
    return outcomes

  def _seed(self) -> int:
    """Returns the next seed of a rollout's random choices."""
    return self._rng.randrange(_SEED_BOUND)


# OpenSpiel's Python MCTS is untyped.
class _Bot(mcts.MCTSBot):  # type: ignore[misc]
  """OpenSpiel's MCTS bot, keeping the root of its latest search."""

  root: mcts.SearchNode | None = None

  def _apply_tree_policy(
      self, root: mcts.SearchNode, state: pyspiel.State
  ) -> tuple[list[mcts.SearchNode], pyspiel.State]:
    """Descends the tree from `root`, as OpenSpiel's MCTS does."""
    self.root = root
    visit_path, working_state = super()._apply_tree_policy(root, state)
    return visit_path, working_state


def _nested_games(parameters: Mapping[str, object]) -> list[pyspiel.Game]:
  """Loads every game named by a game-valued parameter in `parameters`."""
  games = []
  for value in parameters.values():
    if isinstance(value, Mapping):  # A game: its name and its parameters.
      nested = dict(value)
      games.append(pyspiel.load_game(str(nested.pop("name")), nested))
  return games


def implemented_in_python(game: pyspiel.Game) -> bool:
  """Whether `game`, or any game it wraps, is implemented in Python.

  A C++ wrapper or transform around a Python game is a C++ object, so its own
  type does not tell; the games it wraps are read from its game-valued
  parameters, at any depth.
  """
  if type(game).__module__.partition(".")[0] != "pyspiel":
    return True
  return any(
      implemented_in_python(nested)
      for nested in _nested_games(game.get_parameters())
  )


@dataclasses.dataclass(frozen=True)
class LimitsProbe:
  """What the probe of the C++ search's limits found.

  Attributes:
    limits: How the search bounds itself.
    simulations: How many simulations the probe's search ran.
    seconds: How long it took.
  """

  limits: contract.MctsLimits
  simulations: int
  seconds: float


def probe_limits() -> LimitsProbe:
  """Finds out whether the C++ search stops at its count with a clock set.

  Searches `PROBE_GAME` with `PROBE_SIMULATIONS` simulations and a
  `PROBE_SECONDS` wall-clock time, and counts the simulations it ran from the
  root's visit count.

  Returns:
    `SIMULATIONS_OR_CLOCK` when the search ran its simulations and stopped
    before the clock, `SIMULATIONS_ONLY` when it ran more than its
    simulations.

  Raises:
    RuntimeError: If the search ran no more than its simulations but took
      the full wall-clock time, which either behaviour could explain.
  """
  game = pyspiel.load_game(PROBE_GAME)
  bot = pyspiel.MCTSBot(
      game=game,
      evaluator=pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=0),
      uct_c=2.0,
      max_simulations=PROBE_SIMULATIONS,
      max_memory_mb=MAX_MEMORY_MB,
      solve=False,
      seed=0,
      verbose=False,
      max_wall_clock_time=PROBE_SECONDS,
  )
  start = time.monotonic()
  root = bot.mcts_search(game.new_initial_state())
  seconds = time.monotonic() - start
  simulations = int(root.explore_count)
  if simulations > PROBE_SIMULATIONS:
    found = contract.MctsLimits.SIMULATIONS_ONLY
  elif simulations == PROBE_SIMULATIONS and seconds < PROBE_SECONDS:
    found = contract.MctsLimits.SIMULATIONS_OR_CLOCK
  else:
    raise RuntimeError(
        f"pyspiel.MCTSBot, given {PROBE_SIMULATIONS} simulations and "
        f"{PROBE_SECONDS:g} s, ran {simulations} simulations in "
        f"{seconds:.3f} s, so whether it honours its simulation count with a "
        "wall-clock time set is unknown."
    )
  return LimitsProbe(found, simulations, seconds)


@functools.cache
def limits() -> LimitsProbe:
  """Returns `probe_limits`'s finding, probing on the first call only."""
  return probe_limits()


def cpp_bot(
    game: pyspiel.Game,
    *,
    seed: int,
    max_simulations: int,
    uct_c: float,
    solve: bool,
    max_wall_clock_time: float | None = None,
) -> pyspiel.MCTSBot:
  """Returns OpenSpiel's C++ MCTS bot for one move of `game`.

  Its tree is capped at `MAX_MEMORY_MB`.

  Args:
    game: The game to search.
    seed: Seeds every random choice of the search.
    max_simulations: How many simulations to run, unless solved sooner.
    uct_c: The exploration constant of the UCT selection rule.
    solve: Whether proven wins and losses are backed up the search tree.
    max_wall_clock_time: The search's wall-clock time in seconds, or None
      for none. Only for a pyspiel whose search stops at whichever limit
      comes first (`SIMULATIONS_OR_CLOCK`); in any other the clock replaces
      `max_simulations`.

  Raises:
    ValueError: If `game` is, or wraps, a game implemented in Python, or
      `max_wall_clock_time` is not positive.
  """
  if implemented_in_python(game):
    raise ValueError(
        f"{game.get_type().short_name} is, or wraps, a game implemented in "
        "Python, which the C++ search cannot call safely."
    )
  clock: dict[str, float] = {}
  if max_wall_clock_time is not None:
    if max_wall_clock_time <= 0:
      raise ValueError(
          f"The wall-clock time must be positive, not {max_wall_clock_time}."
      )
    clock["max_wall_clock_time"] = max_wall_clock_time
  return pyspiel.MCTSBot(
      game=game,
      evaluator=pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=seed),
      uct_c=uct_c,
      max_simulations=max_simulations,
      max_memory_mb=MAX_MEMORY_MB,
      solve=solve,
      seed=seed,
      verbose=False,
      **clock,
  )


def cpp_search(
    game: pyspiel.Game,
    state: pyspiel.State,
    *,
    seed: int,
    max_simulations: int,
    uct_c: float,
    solve: bool,
    max_wall_clock_time: float | None = None,
) -> int:
  """Returns the action OpenSpiel's C++ MCTS prefers at `state`.

  The search runs `max_simulations` simulations, unless it is solved sooner
  or its wall-clock time runs out first, and cannot be stopped once started;
  see `cpp_bot`.
  """
  bot = cpp_bot(
      game,
      seed=seed,
      max_simulations=max_simulations,
      uct_c=uct_c,
      solve=solve,
      max_wall_clock_time=max_wall_clock_time,
  )
  return int(bot.step(state))


def python_search(
    game: pyspiel.Game,
    state: pyspiel.State,
    *,
    seed: int,
    max_simulations: int,
    uct_c: float,
    solve: bool,
    node_budget: int | None = None,
) -> int:
  """Returns the action OpenSpiel's Python MCTS prefers at `state`.

  Args:
    game: The game `state` is a position of.
    state: The position, where a player is to move.
    seed: Seeds every random choice of the search, so the move is
      reproducible.
    max_simulations: How many simulations to run, unless the search is
      solved, stopped or its tree full sooner.
    uct_c: The exploration constant of the UCT selection rule.
    solve: Whether proven wins and losses are backed up the search tree.
    node_budget: The most nodes the search tree may hold, `NODE_BUDGET` if
      None. A search whose next expansion would exceed it stops there and
      plays the best move of the simulations it completed.

  Raises:
    move_budget.StoppedError: If the move budget expired during the search.
  """
  bot = _Bot(
      game=game,
      uct_c=uct_c,
      max_simulations=max_simulations,
      evaluator=Rollouts(
          game.num_players(),
          seed,
          NODE_BUDGET if node_budget is None else node_budget,
      ),
      solve=solve,
      # Pylint cannot see numpy's compiled random module.
      random_state=np.random.RandomState(seed),  # pylint: disable=no-member
  )
  try:
    return int(bot.step(state))
  except _NodeBudgetReachedError:
    # The simulation that would have grown the tree is dropped; every earlier
    # one is backed up, and the root has been expanded.
    assert bot.root is not None and bot.root.children
    return int(bot.root.best_child().action)
