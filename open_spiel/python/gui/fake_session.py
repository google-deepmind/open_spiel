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

"""A stand-in session for testing code that drives the session API.

`FakeSession` plays a sequential pyspiel game with seeded uniformly random
moves under the `auto` chance policy, and builds the frames the contract
prescribes. A `Fault` makes it misbehave in one chosen way.
"""

import enum
import random

from open_spiel.python.gui import contract
from open_spiel.python.gui import errors
from open_spiel.python.gui import frame as frame_lib
import pyspiel

_HUMAN = contract.Controller(contract.HUMAN, "Human", {})
_RANDOM = contract.Controller("random", "Random", {})


class Fault(enum.Enum):
  """One way for the session to misbehave.

  Attributes:
    NONE: The session behaves.
    STALL: Advancing applies no action.
    PAUSE: Advancing pauses seat 0, as if its controller had failed.
    RAISE_WITHOUT_PAUSE: Advancing raises as if seat 0's controller had
      failed, but pauses nothing.
    BROKEN_FRAME: Building any live frame raises.
    SWAPPED_OBSERVATION: Each seat is given the next seat's observation.
    REPLAY_LEAKS_HISTORY: Every replay shows every viewer the whole history.
  """

  NONE = enum.auto()
  STALL = enum.auto()
  PAUSE = enum.auto()
  RAISE_WITHOUT_PAUSE = enum.auto()
  BROKEN_FRAME = enum.auto()
  SWAPPED_OBSERVATION = enum.auto()
  REPLAY_LEAKS_HISTORY = enum.auto()


class FakeSession:
  """A session of a sequential game, every seat played at random."""

  def __init__(
      self, game_name: str, *, seed: int = 0, fault: Fault = Fault.NONE
  ) -> None:
    """Opens a session of `game_name` with its default parameters.

    Args:
      game_name: The short name of a registered sequential game.
      seed: Seeds the random actions and chance outcomes.
      fault: How the session misbehaves; `Fault.NONE` for not at all.

    Raises:
      ValueError: If the game is not sequential.
    """
    self._game = pyspiel.load_game(game_name)
    game_type = self._game.get_type()
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError(f"{game_name} is not a sequential game.")
    self._game_name = game_name
    self._hides_information = (
        game_type.information
        != pyspiel.GameType.Information.PERFECT_INFORMATION
    )
    self._seed = seed
    self._rng = random.Random(seed)
    self._fault = fault
    self._paused = False
    self._state = self._game.new_initial_state()
    self._sample_chance()

  @property
  def history(self) -> tuple[int, ...]:
    """Every action of the live game, chance outcomes included."""
    return tuple(self._state.history())

  @property
  def is_terminal(self) -> bool:
    """Whether the live game is over."""
    return bool(self._state.is_terminal())

  def frame(self, viewer: contract.Viewer) -> frame_lib.Frame:
    """Returns the live position, built for `viewer`."""
    if self._fault == Fault.BROKEN_FRAME:
      raise RuntimeError("The frame cannot be built.")
    return self._build(viewer, self._state)

  def replay_frame(self, viewer: contract.Viewer, step: int) -> frame_lib.Frame:
    """Returns the position after `step` plies, built for `viewer`."""
    state = self._game.new_initial_state()
    for action in self.history[:step]:
      state.apply_action(action)
    leak = self._fault == Fault.REPLAY_LEAKS_HISTORY
    return self._build(viewer, state, leak_history=leak)

  def advance(
      self, viewer: contract.Viewer, *, max_plies: int | None = 1000
  ) -> frame_lib.Frame:
    """Plays at most `max_plies` (None: no limit) random actions.

    Args:
      viewer: The viewer to build the returned frame for.
      max_plies: The most seat actions to play, chance outcomes not counted;
        None for no limit.

    Returns:
      The live position after play, built for `viewer`.

    Raises:
      errors.ControllerFailedError: If the session is paused, or seat 0's
        controller fails, as a session's would.
    """
    if self._fault == Fault.RAISE_WITHOUT_PAUSE:
      raise errors.ControllerFailedError(seat=0)
    played = 0
    while max_plies is None or played < max_plies:
      if self._paused or self._fault == Fault.PAUSE:
        self._paused = True
        raise errors.ControllerFailedError(seat=0)
      if self._state.is_terminal() or self._fault == Fault.STALL:
        break
      played += 1
      self._state.apply_action(self._rng.choice(self._state.legal_actions()))
      self._sample_chance()
    return self.frame(viewer)

  def _sample_chance(self) -> None:
    """Samples chance outcomes until a seat acts or the game ends."""
    while self._state.is_chance_node():
      outcomes, probabilities = zip(*self._state.chance_outcomes())
      self._state.apply_action(
          self._rng.choices(outcomes, weights=probabilities)[0]
      )

  def _build(
      self,
      viewer: contract.Viewer,
      state: pyspiel.State,
      *,
      leak_history: bool = False,
  ) -> frame_lib.Frame:
    """Builds the frame of `state` for `viewer`; `leak_history` shows all."""
    omniscient = (
        viewer.kind == contract.ViewerKind.ORACLE or not self._hides_information
    )
    to_move = _to_move(state)
    num_plies = len(self._state.history())
    # A seat acts at the live position; an omniscient viewer may also play
    # from a replay.
    offered = to_move.kind == contract.MoverKind.SEAT and (
        len(state.history()) == num_plies or omniscient
    )
    if offered:
      actions = tuple(
          contract.Action(action, state.action_to_string(action))
          for action in state.legal_actions()
      )
    else:
      actions = ()
    shown = omniscient or viewer.seat == to_move.seat
    history = self._history(viewer, omniscient or leak_history)
    ply = len(state.history())
    return frame_lib.Frame(
        session=self._info(viewer),
        viewer=viewer,
        omniscient=omniscient,
        status=self._status(),
        seats=self._seats(omniscient),
        ply=ply,
        num_plies=num_plies,
        to_move=to_move,
        legal_actions=actions if shown else None,
        chance_outcomes=() if omniscient else None,
        history=history,
        rewards=history[ply - 1].rewards if ply else None,
        returns=tuple(state.returns()) if state.is_terminal() else None,
        state_text=str(state) if omniscient else None,
        observation_text=self._observation(viewer, state),
        information_state_text=self._information_state(viewer, state),
        struct=None,
    )

  def _info(self, viewer: contract.Viewer) -> contract.SessionInfo:
    """Returns the session's fixed properties, as `viewer` may see them."""
    return contract.SessionInfo(
        id="fake",
        game=self._game_name,
        parameters={},
        num_seats=self._game.num_players(),
        chance_policy=contract.ChancePolicy.AUTO,
        turn_based=False,
        hides_information=self._hides_information,
        seed=self._seed if viewer.kind == contract.ViewerKind.ORACLE else None,
        controllers=(_HUMAN, _RANDOM),
    )

  def _status(self) -> contract.SessionStatus:
    """Returns the state of the live game."""
    if self._paused:
      return contract.SessionStatus.PAUSED
    if self._state.is_terminal():
      return contract.SessionStatus.TERMINAL
    return contract.SessionStatus.PLAYING

  def _seats(self, omniscient: bool) -> tuple[contract.SeatView, ...]:
    """Returns every seat; seat 0 has failed when the session is paused."""
    failure = contract.ControllerFailure(
        "RuntimeError", "The controller stopped." if omniscient else None
    )
    return tuple(
        contract.SeatView(
            seat=seat,
            controller=_RANDOM.name,
            config={},
            autoplay=False,
            failure=failure if self._paused and seat == 0 else None,
        )
        for seat in range(self._game.num_players())
    )

  def _history(
      self, viewer: contract.Viewer, omniscient: bool
  ) -> tuple[contract.HistoryEntry, ...]:
    """Returns the live history, withholding what `viewer` may not see."""
    entries = []
    state = self._game.new_initial_state()
    for action in self._state.history():
      seat = None if state.is_chance_node() else state.current_player()
      label = state.action_to_string(state.current_player(), action)
      state.apply_action(action)
      if omniscient or (seat is not None and seat == viewer.seat):
        rewards = tuple(state.rewards())
        entries.append(contract.HistoryEntry(seat, action, label, rewards))
      else:
        entries.append(contract.HistoryEntry(seat, None, None, None))
    return tuple(entries)

  def _observed_seat(self, viewer: contract.Viewer) -> int | None:
    """Returns whose private texts `viewer` is given, if anyone's."""
    if viewer.seat is None:
      return None
    if self._fault == Fault.SWAPPED_OBSERVATION:
      return int((viewer.seat + 1) % self._game.num_players())
    return viewer.seat

  def _observation(
      self, viewer: contract.Viewer, state: pyspiel.State
  ) -> str | None:
    """Returns `viewer`'s observation of `state`, if it is given one."""
    seat = self._observed_seat(viewer)
    if seat is None or not self._game.get_type().provides_observation_string:
      return None
    return str(state.observation_string(seat))

  def _information_state(
      self, viewer: contract.Viewer, state: pyspiel.State
  ) -> str | None:
    """Returns `viewer`'s information state at `state`, if it is given one."""
    provided = self._game.get_type().provides_information_state_string
    seat = self._observed_seat(viewer)
    if seat is None or not provided:
      return None
    return str(state.information_state_string(seat))


def _to_move(state: pyspiel.State) -> contract.ToMove:
  """Returns who acts at `state`."""
  if state.is_terminal():
    return contract.ToMove(contract.MoverKind.TERMINAL)
  if state.is_chance_node():
    return contract.ToMove(contract.MoverKind.CHANCE)
  return contract.ToMove(contract.MoverKind.SEAT, state.current_player())
