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

"""Checks a session's frames against its history, replayed independently.

`Replay` replays a session's history in a fresh pyspiel state. The checks
compare each frame the session builds with that state, so a frame that shows
another position, offers the wrong actions, or gives a viewer something the
contract withholds from it raises `InvariantError`.
"""

from collections.abc import Sequence
import dataclasses
import json
import math

from open_spiel.python.gui import contract
from open_spiel.python.gui import frame as frame_lib
import pyspiel

_FRAME_FIELDS = frozenset(
    field.name for field in dataclasses.fields(frame_lib.Frame)
)


class InvariantError(Exception):
  """A frame broke one of the checks."""


def require(condition: bool, message: str) -> None:
  """Raises `InvariantError` with `message` unless `condition` holds."""
  if not condition:
    raise InvariantError(message)


@dataclasses.dataclass(frozen=True)
class Move:
  """One replayed action.

  Attributes:
    seat: The seat that played it, or None for a chance outcome.
    action: The OpenSpiel action id.
    rewards: Each seat's reward for it, in seat order.
  """

  seat: int | None
  action: int
  rewards: tuple[float, ...]


class Replay:
  """A session's history, replayed independently in a fresh pyspiel state.

  The game is loaded with its default parameters; a simultaneous game is
  loaded through the turn-based wrapper, as the server serves it.

  Attributes:
    num_seats: The game's number of seats.
    hides_information: Whether the game hides anything from some seat: it
      does not declare perfect information, or it is simultaneous, so the
      wrapper withholds each seat's choice from the seats after it.
    provides_observation: Whether the game provides observation strings.
    provides_information_state: Whether the game provides information-state
      strings.
    state: The position the replayed history reaches.
    moves: Each replayed action.
  """

  def __init__(self, game_name: str) -> None:
    game = pyspiel.load_game(game_name)
    game_type = game.get_type()
    simultaneous = game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS
    if simultaneous:
      game = pyspiel.load_game_as_turn_based(game_name)
    self._game = game
    self.num_seats: int = game.num_players()
    self.hides_information: bool = simultaneous or (
        game_type.information
        != pyspiel.GameType.Information.PERFECT_INFORMATION
    )
    self.provides_observation: bool = (
        game.get_type().provides_observation_string
    )
    self.provides_information_state: bool = (
        game.get_type().provides_information_state_string
    )
    self.state: pyspiel.State = game.new_initial_state()
    self.moves: list[Move] = []

  @property
  def num_plies(self) -> int:
    """The plies replayed."""
    return len(self.moves)

  def follow(self, history: Sequence[int]) -> None:
    """Replays the actions of `history` not yet replayed."""
    replayed = [move.action for move in self.moves]
    require(
        list(history[: len(replayed)]) == replayed,
        "The session changed actions already played.",
    )
    for action in history[len(replayed) :]:
      self._apply(action)

  def _apply(self, action: int) -> None:
    """Applies `action`, which must be legal, to the state."""
    state = self.state
    require(not state.is_terminal(), f"Action {action} follows the end.")
    if state.is_chance_node():
      legal = [outcome for outcome, _ in state.chance_outcomes()]
      seat = None
    else:
      legal = state.legal_actions()
      seat = state.current_player()
    require(
        action in legal, f"Action {action} at ply {self.num_plies} is illegal."
    )
    state.apply_action(action)
    self.moves.append(Move(seat, action, tuple(state.rewards())))

  def state_at(self, step: int) -> pyspiel.State:
    """Returns the state the first `step` replayed actions reach."""
    state = self._game.new_initial_state()
    for move in self.moves[:step]:
      state.apply_action(move.action)
    return state

  def to_move(self) -> contract.ToMove:
    """Returns who acts at the replayed position."""
    if self.state.is_terminal():
      return contract.ToMove(contract.MoverKind.TERMINAL)
    if self.state.is_chance_node():
      return contract.ToMove(contract.MoverKind.CHANCE)
    return contract.ToMove(contract.MoverKind.SEAT, self.state.current_player())


def check_live_frame(
    frame: frame_lib.Frame, viewer: contract.Viewer, replay: Replay
) -> None:
  """Checks a live frame against the position the history reaches.

  Args:
    frame: The frame the session built for `viewer`.
    viewer: The viewer the frame was requested for.
    replay: The session's history, replayed.

  Raises:
    InvariantError: If the frame breaks a check.
  """
  _check_wire(frame)
  # The server may treat more games as hiding information than the
  # declaration does (a game can keep private state it does not declare),
  # never fewer.
  require(
      frame.session.hides_information or not replay.hides_information,
      "The session says the game hides nothing, but it hides information.",
  )
  _check_position(frame, viewer, replay)
  if viewer.kind == contract.ViewerKind.ORACLE:
    require(
        frame.omniscient and frame.state_text == str(replay.state),
        "The oracle's state text is not the position the history reaches.",
    )
  elif replay.hides_information:
    _check_withheld(frame, viewer)
  if viewer.seat is not None:
    _check_seat_texts(frame, viewer.seat, replay, replay.state)


def check_replay_frame(
    frame: frame_lib.Frame,
    viewer: contract.Viewer,
    step: int,
    replay: Replay,
) -> None:
  """Checks a viewer's replay of the position after `step` plies.

  A replay carries the whole live history, so it is held to the same
  withholding as a live frame: the oracle sees the position, and in a game
  that hides information a seat sees only its own moves and its own texts of
  the replayed position, and the spectator only who moved.

  Args:
    frame: The frame of the position after `step` plies, built for `viewer`.
    viewer: The viewer the replay was requested for.
    step: The plies the frame was requested at.
    replay: The session's history, replayed.

  Raises:
    InvariantError: If the frame is not that position, or gives `viewer`
      something the contract withholds from it.
  """
  _check_wire(frame)
  require(
      frame.viewer == viewer,
      f"The replay requested for {_name(viewer)} is built for another viewer.",
  )
  require(
      (frame.ply, frame.num_plies) == (step, replay.num_plies),
      f"The replay of ply {step} of {replay.num_plies} is of ply {frame.ply} "
      f"of {frame.num_plies}.",
  )
  _check_history(frame, replay)
  state = replay.state_at(step)
  # An omniscient viewer may play from a replay; no other viewer may.
  if frame.omniscient:
    _check_legal_actions(frame, viewer, state)
  else:
    require(
        not frame.legal_actions,
        f"The replay of ply {step} offers legal actions to {_name(viewer)}.",
    )
  if viewer.kind == contract.ViewerKind.ORACLE:
    require(
        frame.omniscient and frame.state_text == str(state),
        f"The replay of ply {step} is not the position the history reaches.",
    )
  elif replay.hides_information:
    _check_withheld(frame, viewer)
  if viewer.seat is not None:
    _check_seat_texts(frame, viewer.seat, replay, state)


def _check_wire(frame: frame_lib.Frame) -> None:
  """Checks that the frame is strict JSON carrying exactly its fields."""
  wire = contract.to_json(frame)
  try:
    decoded = json.loads(json.dumps(wire, allow_nan=False))
  except ValueError as e:
    raise InvariantError(f"The frame is not strict JSON: {e}") from e
  require(decoded == wire, "The frame changes in a JSON round trip.")
  require(
      isinstance(decoded, dict) and set(decoded) == _FRAME_FIELDS,
      "The frame does not carry exactly the frame's fields.",
  )


def _check_position(
    frame: frame_lib.Frame, viewer: contract.Viewer, replay: Replay
) -> None:
  """Checks the frame shows the live position of the replayed history."""
  require(
      frame.viewer == viewer,
      f"The frame requested for {_name(viewer)} is built for another viewer.",
  )
  require(
      frame.live and frame.num_plies == replay.num_plies,
      f"The live frame is at ply {frame.ply} of {frame.num_plies}, not "
      f"{replay.num_plies}.",
  )
  _check_history(frame, replay)
  to_move = replay.to_move()
  require(
      frame.to_move == to_move,
      f"The frame has {frame.to_move.kind} {frame.to_move.seat} to move, "
      f"not {to_move.kind} {to_move.seat}.",
  )
  require(
      to_move.kind != contract.MoverKind.CHANCE,
      "Under the auto chance policy, the live position is a chance node.",
  )
  _check_legal_actions(frame, viewer, replay.state)
  if to_move.kind == contract.MoverKind.TERMINAL:
    _check_returns(frame, replay)


def _check_history(frame: frame_lib.Frame, replay: Replay) -> None:
  """Checks each shown history entry is the replayed move."""
  for ply, (entry, move) in enumerate(zip(frame.history, replay.moves)):
    require(
        entry.seat == move.seat
        and (
            entry.withheld
            or (
                entry.action == move.action
                and _same_rewards(entry.rewards, move)
            )
        ),
        f"History entry {ply} is not seat {move.seat} playing {move.action} "
        f"for rewards {move.rewards}.",
    )


def _check_legal_actions(
    frame: frame_lib.Frame, viewer: contract.Viewer, state: pyspiel.State
) -> None:
  """Checks the seat to move is offered the legal actions, each labelled."""
  acting = frame.to_move.seat
  if acting is None:
    require(not frame.legal_actions, "Legal actions are offered off turn.")
    return
  if frame.legal_actions is None:
    require(
        viewer.kind == contract.ViewerKind.SPECTATOR
        or (viewer.seat is not None and viewer.seat != acting),
        f"The legal actions are withheld from {_name(viewer)}.",
    )
    return
  offered = [action.id for action in frame.legal_actions]
  legal = list(state.legal_actions())
  require(offered == legal, f"Seat {acting} is offered {offered}, not {legal}.")
  require(
      all(action.label.strip() for action in frame.legal_actions),
      f"A legal action of seat {acting} has no label.",
  )


def _check_returns(frame: frame_lib.Frame, replay: Replay) -> None:
  """Checks the terminal frame gives every seat's return."""
  expected = list(replay.state.returns())
  require(
      frame.returns is not None
      and len(frame.returns) == len(expected)
      and all(math.isfinite(value) for value in frame.returns)
      and all(map(math.isclose, frame.returns, expected)),
      f"The returns {frame.returns} are not {expected}.",
  )


def _check_withheld(frame: frame_lib.Frame, viewer: contract.Viewer) -> None:
  """Checks a frame of a game that hides information withholds it.

  A frame that is not omniscient cannot be built with the full state, the
  seed, chance outcomes, failure messages or another seat's legal actions.
  This checks the `omniscient` flag against the game, and that the history
  shows the viewer exactly its own actions.

  Args:
    frame: The frame built for `viewer`.
    viewer: The viewer the frame was built for.
  """
  name = _name(viewer)
  require(
      not frame.omniscient,
      f"The full state of a game that hides information is shown to {name}.",
  )
  for ply, entry in enumerate(frame.history):
    own = viewer.seat is not None and entry.seat == viewer.seat
    require(
        entry.withheld != own,
        f"History entry {ply}, by seat {entry.seat}, is "
        f"{'withheld from' if own else 'shown to'} {name}.",
    )


def _check_seat_texts(
    frame: frame_lib.Frame, seat: int, replay: Replay, state: pyspiel.State
) -> None:
  """Checks a seat is given exactly its own texts of `state`.

  The seat gets its observation exactly when the game provides observations,
  and its information state exactly when the game provides those. A session
  that treats the game as hiding more than the game declares (its
  information state can print what it hides) may withhold the information
  state.

  Args:
    frame: The frame built for a viewer of `seat`.
    seat: The seat the frame was built for.
    replay: The replayed history, which tells what the game provides and whether
      it hides information.
    state: The position the frame shows.
  """
  require(
      frame.omniscient
      or frame.observation_text is not None
      or frame.information_state_text is not None,
      f"Seat {seat} of a game that hides information gets no observation.",
  )
  observation = (
      str(state.observation_string(seat))
      if replay.provides_observation
      else None
  )
  require(
      frame.observation_text == observation,
      f"Seat {seat} is not given its own observation"
      + ("." if replay.provides_observation else ": the game provides none."),
  )
  information_state = (
      str(state.information_state_string(seat))
      if replay.provides_information_state
      else None
  )
  stricter = frame.session.hides_information and not replay.hides_information
  require(
      frame.information_state_text == information_state
      or (stricter and frame.information_state_text is None),
      f"Seat {seat} is not given its own information state"
      + (
          "."
          if replay.provides_information_state
          else ": the game provides none."
      ),
  )


def _same_rewards(rewards: tuple[float, ...] | None, move: Move) -> bool:
  """Whether `rewards` are the rewards of `move`."""
  return (
      rewards is not None
      and len(rewards) == len(move.rewards)
      and all(map(math.isclose, rewards, move.rewards))
  )


def _name(viewer: contract.Viewer) -> str:
  """Names a viewer in a message."""
  if viewer.seat is not None:
    return f"seat {viewer.seat}"
  return f"the {viewer.kind}"
