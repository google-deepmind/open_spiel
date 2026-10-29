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


"""Building a frame for one viewer, withholding what the viewer may not see.

The rules are those of CONTRACT.md: an omniscient viewer sees everything; a
seat sees its own observation and its own moves; a spectator sees only who
moved. `frame.Frame` checks the result, so a leak fails to build.
"""

from collections.abc import Sequence
import dataclasses
import json

from open_spiel.python.gui import contract
from open_spiel.python.gui import frame
from open_spiel.python.gui import loader
from open_spiel.python.gui import plies
from open_spiel.python.gui import presenters
import pyspiel


@dataclasses.dataclass(frozen=True)
class Position:
  """A position of a session, before it is shown to anyone.

  Attributes:
    info: The session's fixed properties, seed included.
    loaded: The session's game.
    status: The state of the session's live game.
    seats: Every seat, failure messages included.
    state: The state at this position.
    ply: The number of plies applied to reach this position.
    history: Every ply of the live game.
  """

  info: contract.SessionInfo
  loaded: loader.LoadedGame
  status: contract.SessionStatus
  seats: tuple[contract.SeatView, ...]
  state: pyspiel.State
  ply: int
  history: Sequence[plies.Ply]

  @property
  def live(self) -> bool:
    """Whether this is the session's current position."""
    return self.ply == len(self.history)


def is_omniscient(loaded: loader.LoadedGame, viewer: contract.Viewer) -> bool:
  """Whether `viewer` may see the full state of `loaded`."""
  if viewer.kind == contract.ViewerKind.ORACLE:
    return True
  return not loaded.hides_information


def status(
    live: pyspiel.State, seats: Sequence[contract.SeatView]
) -> contract.SessionStatus:
  """Returns the status of a session's live game."""
  if live.is_terminal():
    return contract.SessionStatus.TERMINAL
  if any(seat.failure is not None for seat in seats):
    return contract.SessionStatus.PAUSED
  return contract.SessionStatus.PLAYING


def build(position: Position, viewer: contract.Viewer) -> frame.Frame:
  """Builds the frame of `position` for `viewer`.

  Args:
    position: The position to show.
    viewer: Who the frame is for. A seat viewer must be a seat of the game,
      and the game must have seat views.

  Returns:
    The frame, holding nothing `viewer` may not see.

  Raises:
    ValueError: If `viewer` is a seat the game does not have, or the game has
      no seat views. Callers refuse such a viewer first, as a request error.
  """
  if viewer.seat is not None:
    loaded = position.loaded
    if not 0 <= viewer.seat < loaded.num_seats:
      raise ValueError(f"{loaded.name} has no seat {viewer.seat}.")
    if not loaded.has_seat_views:
      raise ValueError(f"{loaded.name} has no seat views.")
  omniscient = is_omniscient(position.loaded, viewer)
  state = position.state
  mover = plies.to_move(state)
  history = tuple(
      _history_entry(ply, viewer, omniscient) for ply in position.history
  )
  legal_actions = _legal_actions(position, viewer, omniscient, mover)
  return frame.Frame(
      session=(
          position.info
          if viewer.kind == contract.ViewerKind.ORACLE
          else dataclasses.replace(position.info, seed=None)
      ),
      viewer=viewer,
      omniscient=omniscient,
      status=position.status,
      seats=tuple(_seat(seat, viewer, omniscient) for seat in position.seats),
      ply=position.ply,
      num_plies=len(position.history),
      to_move=mover,
      legal_actions=legal_actions,
      chance_outcomes=_chance_outcomes(position, omniscient, mover),
      history=history,
      rewards=history[position.ply - 1].rewards if position.ply else None,
      returns=(
          tuple(float(value) for value in state.returns())
          if state.is_terminal()
          else None
      ),
      state_text=str(state) if omniscient else None,
      observation_text=_seat_text(position, viewer, observation=True),
      information_state_text=_seat_text(position, viewer, observation=False),
      struct=(
          json.loads(state.to_json())
          if omniscient and position.loaded.provides_struct
          else None
      ),
      observation_struct=_observation_struct(position, viewer),
      presentation=presenters.present(
          position.loaded.name,
          state,
          presenters.Viewpoint(seat=viewer.seat, omniscient=omniscient),
          (
              None
              if legal_actions is None
              else [action.id for action in legal_actions]
          ),
      ),
  )


def _history_entry(
    ply: plies.Ply, viewer: contract.Viewer, omniscient: bool
) -> contract.HistoryEntry:
  """Shows one ply: in full to its own seat and to an omniscient viewer.

  Any other viewer reads only the ply's public label, if its game gives one.
  """
  if omniscient or (ply.seat is not None and ply.seat == viewer.seat):
    return contract.HistoryEntry(
        seat=ply.seat, action=ply.action, label=ply.label, rewards=ply.rewards
    )
  return contract.HistoryEntry(
      seat=ply.seat,
      action=None,
      label=None,
      rewards=None,
      public_label=ply.public_label,
  )


def _seat(
    seat: contract.SeatView, viewer: contract.Viewer, omniscient: bool
) -> contract.SeatView:
  """Shows a seat, withholding its failure message and notice from most.

  A failure message reaches only an omniscient viewer; a notice, only an
  omniscient viewer and the seat's own.
  """
  if omniscient:
    return seat
  return dataclasses.replace(
      seat,
      failure=(
          None
          if seat.failure is None
          else dataclasses.replace(seat.failure, message=None)
      ),
      notice=seat.notice if viewer.seat == seat.seat else None,
  )


def _legal_actions(
    position: Position,
    viewer: contract.Viewer,
    omniscient: bool,
    mover: contract.ToMove,
) -> tuple[contract.Action, ...] | None:
  """Lists the acting seat's actions, for viewers that may see them."""
  if not omniscient and (mover.seat is None or mover.seat != viewer.seat):
    return None
  # An omniscient viewer may play from an earlier position, so a replay
  # offers it the actions there.
  if mover.kind != contract.MoverKind.SEAT or not (position.live or omniscient):
    return ()
  state = position.state
  return tuple(
      contract.Action(id=action, label=plies.label(state, action))
      for action in plies.legal_actions(state)
  )


def _chance_outcomes(
    position: Position, omniscient: bool, mover: contract.ToMove
) -> tuple[contract.ChanceOutcome, ...] | None:
  """Lists the chance outcomes a client may choose, to omniscient viewers."""
  if not omniscient:
    return None
  if (
      mover.kind != contract.MoverKind.CHANCE
      or position.info.chance_policy != contract.ChancePolicy.EXPOSE
  ):
    return ()
  state = position.state
  return tuple(
      contract.ChanceOutcome(
          id=int(action),
          label=plies.label(state, int(action)),
          probability=float(probability),
      )
      for action, probability in state.chance_outcomes()
      if probability > 0
  )


def _seat_text(
    position: Position, viewer: contract.Viewer, *, observation: bool
) -> str | None:
  """Returns a seat viewer's observation or information state text."""
  loaded = position.loaded
  provided = (
      loaded.provides_observation
      if observation
      else loaded.provides_information_state
  )
  if viewer.seat is None or not provided:
    return None
  if observation:
    return str(position.state.observation_string(viewer.seat))
  return str(position.state.information_state_string(viewer.seat))


def _observation_struct(
    position: Position, viewer: contract.Viewer
) -> contract.Json:
  """Returns a seat viewer's observation as the game's structured JSON.

  The game builds the struct for the seat, so it holds only what the seat may
  observe. The oracle reads `struct` instead, and a spectator, who holds no
  seat, is given none: OpenSpiel defines observations per player only. A
  hearts seat is given none while the hand is passing (the glue below).
  """
  if viewer.seat is None or not position.loaded.provides_observation_struct:
    return None
  struct = position.state.to_observation_struct(viewer.seat)
  observed = json.loads(struct.to_json())
  # STRUCT-GLUE(hearts ObservationStruct): withheld during passing until the
  # engine hides received cards before the seat's own pass. Until then a seat
  # gets only the hearts adapter's presentation in that phase.
  if position.loaded.name == "hearts" and observed["phase"] == "Pass":
    return None
  return observed
