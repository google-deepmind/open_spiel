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


"""Plies: actions applied to a state, recorded so a position can be rebuilt.

OpenSpiel states cannot be rewound, so every earlier position is rebuilt by
applying the history to a new initial state. A game whose rebuilt plies
differ from the recorded ones has drifted, and cannot be replayed.
"""

from collections.abc import Sequence
import dataclasses
import random

from open_spiel.python.gui import contract
from open_spiel.python.gui import errors
import pyspiel


class IllegalPlyError(errors.IllegalActionError):
  """Raised when an action is not legal at the position it is applied to.

  Attributes:
    ply: The index of the action in the history.
    seat: The seat to move, or None at a chance node.
  """

  def __init__(self, message: str, *, ply: int, seat: int | None) -> None:
    super().__init__(message, seat=seat, ply=ply)


class DriftError(RuntimeError):
  """Raised when rebuilding a position does not reproduce its record.

  The message shows the plies involved, hidden ones included, so it is for
  logs and never for a viewer.

  Attributes:
    ply: The index of the ply that was not reproduced.
  """

  def __init__(self, message: str, *, ply: int) -> None:
    super().__init__(message)
    self.ply = ply


@dataclasses.dataclass(frozen=True)
class Ply:
  """One action applied to a state.

  Attributes:
    seat: The seat that acted, or None for a chance outcome.
    action: The OpenSpiel action id.
    label: The game's text for the action, at the position it was applied.
    rewards: Each seat's reward for the action, in seat order.
  """

  seat: int | None
  action: int
  label: str
  rewards: tuple[float, ...]


def to_move(state: pyspiel.State) -> contract.ToMove:
  """Returns who acts at `state`."""
  if state.is_terminal():
    return contract.ToMove(kind=contract.MoverKind.TERMINAL)
  if state.is_chance_node():
    return contract.ToMove(kind=contract.MoverKind.CHANCE)
  player = int(state.current_player())
  if player < 0:
    raise ValueError(f"No single seat acts at player id {player}.")
  return contract.ToMove(kind=contract.MoverKind.SEAT, seat=player)


def legal_actions(state: pyspiel.State) -> tuple[int, ...]:
  """Returns the actions or chance outcomes that may be applied to `state`."""
  if state.is_terminal():
    return ()
  if state.is_chance_node():
    return tuple(
        int(action)
        for action, probability in state.chance_outcomes()
        if probability > 0
    )
  return tuple(int(action) for action in state.legal_actions())


def sample_chance(state: pyspiel.State, rng: random.Random) -> int:
  """Samples an outcome at a chance node from its distribution."""
  outcomes, probabilities = zip(*state.chance_outcomes())
  return int(rng.choices(outcomes, weights=probabilities)[0])


def label(state: pyspiel.State, action: int) -> str:
  """Returns the game's text for `action` at `state`."""
  return str(state.action_to_string(state.current_player(), action))


def apply(state: pyspiel.State, action: int, *, ply: int) -> Ply:
  """Applies a legal action or chance outcome to `state`, recording it.

  Args:
    state: The state to change.
    action: An action or chance outcome legal at `state`.
    ply: The index the action takes in the history.

  Returns:
    The record of the action.

  Raises:
    IllegalPlyError: If `action` is not legal at `state`.
  """
  seat = to_move(state).seat
  if action not in legal_actions(state):
    raise IllegalPlyError(
        f"Action {action} is not legal at ply {ply}.", ply=ply, seat=seat
    )
  text = label(state, action)
  state.apply_action(action)
  return Ply(
      seat=seat,
      action=action,
      label=text,
      rewards=tuple(float(reward) for reward in state.rewards()),
  )


def rebuild(
    game: pyspiel.Game,
    actions: Sequence[int],
    *,
    expected: Sequence[Ply] = (),
) -> tuple[pyspiel.State, tuple[Ply, ...]]:
  """Rebuilds a position by applying `actions` to a new initial state.

  Args:
    game: The game the actions were played in.
    actions: The history to apply, chance outcomes included.
    expected: The recorded plies of `actions`, or empty to skip the check.

  Returns:
    The rebuilt state and the plies that reached it.

  Raises:
    IllegalPlyError: If an action is not legal where it is applied.
    DriftError: If a rebuilt ply differs from its record.
  """
  if expected and len(expected) != len(actions):
    raise ValueError("Every action has exactly one recorded ply.")
  state = game.new_initial_state()
  plies = []
  for index, action in enumerate(actions):
    try:
      ply = apply(state, action, ply=index)
    except IllegalPlyError as error:
      if expected:
        raise DriftError(
            f"Rebuilding ply {index} refused it: {error}", ply=index
        ) from error
      raise
    if expected and ply != expected[index]:
      raise DriftError(
          f"Rebuilding ply {index} gave {ply}, not the recorded "
          f"{expected[index]}.",
          ply=index,
      )
    plies.append(ply)
  return state, tuple(plies)
