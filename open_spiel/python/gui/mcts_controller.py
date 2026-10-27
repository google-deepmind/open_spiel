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

"""The MCTS controller: OpenSpiel's Monte Carlo tree search plays a seat.

A game implemented in C++ is searched by OpenSpiel's C++ MCTS
(`pyspiel.MCTSBot`), its tree capped at `mcts_search.MAX_MEMORY_MB`
megabytes. It cannot be stopped once started, so its limits must fit the
move budget, and they follow how the served pyspiel bounds that search
(`contract.MctsLimits`), which the server finds out once at startup
(`mcts_search.limits`) and sets with `configure`:

- `SIMULATIONS_OR_CLOCK`, the search stops at whichever of its simulation
  count and its wall-clock time comes first: `max_simulations` is at most
  10000, 1000 by default, and the wall-clock time is a backstop a little
  inside the move budget (`backstop_seconds`). A move whose simulations
  finish first is reproducible from the seed; one the backstop stops
  depends on the machine's speed.
- `SIMULATIONS_ONLY`, a wall-clock time replaces the count: the search is
  given none, and `max_simulations` is at most 250, its default, which fits
  the move budget on the slowest catalogued game (`CONTRACT.md`,
  Controllers, has the measurements). Every move is reproducible.

Until `configure` says otherwise, MCTS plays `SIMULATIONS_ONLY`'s limits,
which fit the budget with either behaviour.

A game implemented in Python, or a C++ game wrapping one, is searched by
OpenSpiel's Python MCTS with C++ rollouts, which the C++ search cannot call
safely, under the same `max_simulations` setting. It stops between
simulations once the move budget has expired, and its tree holds at most
`mcts_search.NODE_BUDGET` nodes, after which it plays the best move found so
far (`mcts_search`).

Each search ends every simulation in a uniformly random rollout and is
seeded afresh for the ply it moves at.
"""

from collections.abc import Mapping
import dataclasses

import pyspiel

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_registry
from open_spiel.python.gui import controller_settings
from open_spiel.python.gui import mcts_search
from open_spiel.python.gui import move_budget

# The name MCTS is registered under.
NAME = "mcts"


@dataclasses.dataclass(frozen=True)
class Simulations:
  """The `max_simulations` setting's default and maximum.

  Attributes:
    default: The simulations a move runs unless the client says otherwise.
    maximum: The most simulations a move may run.
  """

  default: int
  maximum: int


# `max_simulations` under each behaviour of the C++ search. With
# `mcts_search.MAX_MEMORY_MB` and `NODE_BUDGET` the maximum bounds a search's
# memory. `SIMULATIONS_ONLY`'s cap fits a move of the slowest catalogued C++
# game, hive, in the 10 second move budget with room to spare, also on a
# host half as fast as a typical server (CONTRACT.md).
SIMULATIONS = {
    contract.MctsLimits.SIMULATIONS_OR_CLOCK: Simulations(1000, 10_000),
    contract.MctsLimits.SIMULATIONS_ONLY: Simulations(250, 250),
}

# How far inside the move budget the C++ search's wall-clock backstop ends
# (`backstop_seconds`): room for building the bot, choosing the move from the
# tree and freeing it.
BACKSTOP_MARGIN_SECONDS = 1.0


def backstop_seconds(budget: float) -> float:
  """Returns the C++ search's wall-clock time under a `budget` second budget.

  `BACKSTOP_MARGIN_SECONDS` inside the budget, or half the budget when that
  is shorter than twice the margin.
  """
  if budget <= 0:
    raise ValueError(f"The move budget must be positive, not {budget}.")
  return budget - min(BACKSTOP_MARGIN_SECONDS, budget / 2)


class Mcts:
  """Plays the move preferred by OpenSpiel's Monte Carlo tree search.

  See the module docstring for the two searches and their limits.
  """

  def __init__(
      self,
      game: pyspiel.Game,
      seed: int,
      *,
      limits: contract.MctsLimits,
      max_simulations: int,
      uct_c: float,
      solve: bool,
  ) -> None:
    self._game = game
    self._seed = seed
    self._limits = limits
    self._max_simulations = max_simulations
    self._uct_c = uct_c
    self._solve = solve
    self._python = mcts_search.implemented_in_python(game)

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the action the search prefers for `seat`."""
    del seat  # The search plays for the player to move, which is `seat`.
    seed = controller_registry.choice_seed(self._seed, state)
    if self._python:
      return mcts_search.python_search(
          self._game,
          state,
          seed=seed,
          max_simulations=self._max_simulations,
          uct_c=self._uct_c,
          solve=self._solve,
      )
    clock = None
    if self._limits == contract.MctsLimits.SIMULATIONS_OR_CLOCK:
      clock = backstop_seconds(move_budget.seconds())
    return mcts_search.cpp_search(
        self._game,
        state,
        seed=seed,
        max_simulations=self._max_simulations,
        uct_c=self._uct_c,
        solve=self._solve,
        max_wall_clock_time=clock,
    )


def refusal(game: pyspiel.Game) -> str | None:
  """Returns why MCTS cannot play `game`, or None if it can.

  The search plays forward from a clone of the full state, so it reads the
  full state (see `Kind.reads_full_state`), and in a game whose state samples
  its own chance outcomes it would foresee them.
  """
  game_type = game.get_type()
  name = game_type.short_name
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    return f"MCTS needs a sequential game, and {name} is not one."
  if game_type.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC:
    return (
        f"MCTS would foresee the chance outcomes of {name}, whose state "
        "samples them itself."
    )
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    return (
        f"MCTS needs rewards at the end of the game only, and {name} does not "
        "give them so."
    )
  return None


def settings(
    limits: contract.MctsLimits,
) -> tuple[controller_settings.Setting, ...]:
  """Returns MCTS's settings under `limits`."""
  simulations = SIMULATIONS[limits]
  return (
      controller_settings.Setting(
          "max_simulations",
          int,
          simulations.default,
          "Simulations per move, each ending in a random rollout.",
          minimum=1,
          maximum=simulations.maximum,
      ),
      controller_settings.Setting(
          "uct_c",
          float,
          2.0,
          "The exploration constant of the UCT selection rule.",
          minimum=0.0,
          maximum=100.0,
      ),
      controller_settings.Setting(
          "solve",
          bool,
          True,
          "Whether proven wins and losses are backed up the search tree.",
      ),
  )


def kind(limits: contract.MctsLimits) -> controller_registry.Kind:
  """Returns MCTS as it is registered under `limits`."""

  def build(
      game: pyspiel.Game,
      config: Mapping[str, contract.ParameterValue],
      seed: int,
  ) -> controller_registry.Controller:
    """Builds the MCTS controller from its parsed settings."""
    return Mcts(
        game,
        seed,
        limits=limits,
        max_simulations=int(config["max_simulations"]),
        uct_c=float(config["uct_c"]),
        solve=bool(config["solve"]),
    )

  return controller_registry.Kind(
      "MCTS",
      settings(limits),
      reads_full_state=True,
      build=build,
      refusal=refusal,
  )


@dataclasses.dataclass
class _Configured:
  """The limits MCTS is registered with; see `configure`."""

  limits: contract.MctsLimits = contract.MctsLimits.SIMULATIONS_ONLY


_CONFIGURED = _Configured()


def configure(limits: contract.MctsLimits) -> None:
  """Registers MCTS again, with the limits `limits` calls for.

  Call it before serving, as `controller_registry.register` says.

  Raises:
    ValueError: If MCTS is not registered.
  """
  controller_registry.replace(NAME, kind(limits))
  _CONFIGURED.limits = limits


def configured_limits() -> contract.MctsLimits:
  """Returns the limits MCTS is registered with."""
  return _CONFIGURED.limits
