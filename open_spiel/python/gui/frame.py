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

"""The frame: one position of a session, built for one viewer.

A field the viewer may not see is null; an empty list means there is nothing
to see.
"""

import dataclasses

from open_spiel.python.gui import contract


@dataclasses.dataclass(frozen=True)
class Frame:
  """A position of a session, built for one viewer.

  A frame is live when `ply == num_plies`, and a replay of an earlier position
  otherwise. A seat acts at the live position; an omniscient viewer may also
  play from a replay, which takes back the plies after it.

  Attributes:
    contract_version: The contract version the frame was built against.
    session: The session's fixed properties.
    viewer: The audience the frame is built for.
    omniscient: Whether the viewer sees the full state: exactly when the
      viewer is the oracle or the game hides nothing
      (`session.hides_information` is false).
    status: The state of the session's live game.
    seats: Every seat, in seat order.
    ply: The number of actions, chance outcomes included, applied to reach
      this position.
    num_plies: The number of actions in the live game.
    to_move: Who acts at this position.
    legal_actions: The actions the acting seat may choose; empty where no
      seat acts and, for a viewer that is not omniscient, on a replay; null
      when withheld from the viewer.
    chance_outcomes: The outcomes the client may choose at a chance node under
      the `expose` policy; empty otherwise, null when withheld from the viewer.
    history: Every action of the live game, in order, as the viewer may see
      it.
    rewards: Each seat's reward for the action that reached this position, in
      seat order; null at the initial position or when that action is
      withheld from the viewer.
    returns: Each seat's return, in seat order, at a terminal position;
      otherwise null.
    state_text: The full state as text; null unless `omniscient`.
    observation_text: The viewer's observation as text; null unless the viewer
      holds a seat and the game provides observations.
    information_state_text: The viewer's information state as text; null
      unless the viewer holds a seat and the game provides information states.
    struct: The full state as the game's structured JSON; null unless
      `omniscient` and the game provides it.
    observation_struct: The viewer's observation as the game's structured
      JSON; null unless the viewer holds a seat and the game provides
      observation structs.
    presentation: The position as the game's adapter presents it to the
      viewer; null when no adapter is registered for the game.
  """

  contract_version: contract.ContractVersion = dataclasses.field(
      default=contract.CONTRACT_VERSION, kw_only=True
  )
  session: contract.SessionInfo
  viewer: contract.Viewer
  omniscient: bool
  status: contract.SessionStatus
  seats: tuple[contract.SeatView, ...]
  ply: int
  num_plies: int
  to_move: contract.ToMove
  legal_actions: tuple[contract.Action, ...] | None
  chance_outcomes: tuple[contract.ChanceOutcome, ...] | None
  history: tuple[contract.HistoryEntry, ...]
  rewards: tuple[float, ...] | None
  returns: tuple[float, ...] | None
  state_text: str | None
  observation_text: str | None
  information_state_text: str | None
  struct: contract.Json
  observation_struct: contract.Json
  presentation: contract.Presentation | None

  @property
  def live(self) -> bool:
    """Whether the frame is the session's current position."""
    return self.ply == self.num_plies

  def __post_init__(self) -> None:
    contract.require(
        self.contract_version == contract.CONTRACT_VERSION,
        f"Contract version {self.contract_version} is not "
        f"{contract.CONTRACT_VERSION}.",
    )
    self._check_seats()
    self._check_history()
    self._check_position()
    self._check_audience()
    self._check_presentation()

  def _check_seats(self) -> None:
    """Checks the seats against the session and its status."""
    contract.require(
        tuple(seat.seat for seat in self.seats)
        == tuple(range(self.session.num_seats)),
        "Every seat is listed once, in seat order.",
    )
    offered = {
        controller.name
        for controller in self.session.controllers
        if controller.unavailable_reason is None
    }
    for seat in self.seats:
      contract.require(
          seat.controller in offered,
          f"Seat {seat.seat} is played by {seat.controller!r}, which the "
          "session does not offer.",
      )
    # A seat's controller fails only in play, which stops before the end; a
    # build fails whenever it fails, so an over game may show one.
    contract.require(
        (self.status == contract.SessionStatus.PAUSED)
        == (
            self.status != contract.SessionStatus.TERMINAL
            and any(seat.failure is not None for seat in self.seats)
        ),
        "A session in play is paused exactly when a seat's controller has "
        "failed.",
    )

  def _check_history(self) -> None:
    """Checks the history and the rewards against the session."""
    num_seats = self.session.num_seats
    contract.require(
        0 <= self.ply <= self.num_plies == len(self.history),
        f"Ply {self.ply} of {self.num_plies} plies is inconsistent with "
        f"{len(self.history)} history entries.",
    )
    for entry in self.history:
      contract.require(
          entry.seat is None or 0 <= entry.seat < num_seats,
          f"History seat {entry.seat} is not one of {num_seats} seats.",
      )
      contract.require(
          entry.rewards is None or len(entry.rewards) == num_seats,
          "History rewards list every seat.",
      )
    last = self.history[self.ply - 1] if self.ply else None
    contract.require(
        self.rewards == (last.rewards if last else None),
        "The frame's rewards are those of the action that reached it.",
    )

  def _check_position(self) -> None:
    """Checks who acts at the position against what is offered."""
    num_seats = self.session.num_seats
    contract.require(
        self.to_move.seat is None or 0 <= self.to_move.seat < num_seats,
        f"Seat {self.to_move.seat} to move is not one of {num_seats} seats.",
    )
    terminal = self.to_move.kind == contract.MoverKind.TERMINAL
    contract.require(
        not self.live
        or (self.status == contract.SessionStatus.TERMINAL) == terminal,
        "The live position is terminal exactly when the session is.",
    )
    contract.require(
        (self.returns is not None) == terminal,
        "Returns are served exactly at a terminal position.",
    )
    contract.require(
        self.returns is None or len(self.returns) == num_seats,
        "Returns list every seat.",
    )
    contract.require(
        not self.legal_actions
        or (
            self.to_move.kind == contract.MoverKind.SEAT
            and (self.live or self.omniscient)
        ),
        "Legal actions are served only where a seat acts, and on a replay "
        "only to an omniscient viewer.",
    )
    contract.require(
        not self.chance_outcomes
        or (
            self.to_move.kind == contract.MoverKind.CHANCE
            and self.session.chance_policy == contract.ChancePolicy.EXPOSE
        ),
        "Chance outcomes are served only at a chance node under the expose "
        "policy.",
    )

  def _check_audience(self) -> None:
    """Checks that the frame holds nothing its viewer may not see."""
    viewer = self.viewer
    is_oracle = viewer.kind == contract.ViewerKind.ORACLE
    is_seat = viewer.kind == contract.ViewerKind.SEAT
    contract.require(
        viewer.seat is None or 0 <= viewer.seat < self.session.num_seats,
        f"Viewer seat {viewer.seat} is not a seat of the session.",
    )
    contract.require(
        not is_oracle or self.omniscient, "The oracle is omniscient."
    )
    contract.require(
        self.omniscient == (is_oracle or not self.session.hides_information),
        "A viewer is omniscient exactly when it is the oracle or the game "
        "hides nothing.",
    )
    contract.require(
        self.omniscient == (self.state_text is not None),
        "The state text is served exactly to an omniscient viewer.",
    )
    contract.require(
        self.omniscient or self.struct is None,
        "The state struct is served only to an omniscient viewer.",
    )
    contract.require(
        is_oracle or self.session.seed is None,
        "The seed is served only to the oracle.",
    )
    contract.require(
        self.omniscient
        or all(
            seat.failure is None or seat.failure.message is None
            for seat in self.seats
        ),
        "Failure messages are served only to an omniscient viewer.",
    )
    contract.require(
        self.omniscient
        or all(
            seat.notice is None or seat.seat == self.viewer.seat
            for seat in self.seats
        ),
        "A seat's notice is served only to that seat and to an omniscient "
        "viewer.",
    )
    contract.require(
        is_seat
        or (
            self.observation_text is None
            and self.information_state_text is None
        ),
        "Per-seat texts are served only to a seat.",
    )
    contract.require(
        is_seat or self.observation_struct is None,
        "The observation struct is served only to a seat.",
    )
    if self.omniscient:
      contract.require(
          self.legal_actions is not None
          and self.chance_outcomes is not None
          and not any(entry.withheld for entry in self.history),
          "Nothing is withheld from an omniscient viewer.",
      )
      return
    contract.require(
        not self.chance_outcomes,
        "Chance outcomes are served only to an omniscient viewer.",
    )
    contract.require(
        not self.legal_actions or self.to_move.seat == viewer.seat,
        "Legal actions are served only to the acting seat.",
    )
    contract.require(
        not is_seat
        or not any(
            entry.withheld and entry.seat == viewer.seat
            for entry in self.history
        ),
        "A seat sees its own actions.",
    )

  def _check_presentation(self) -> None:
    """Checks the adapter's hints against the legal actions offered."""
    if self.presentation is None:
      return
    hints = self.presentation.actions
    contract.require(
        (hints is None) == (self.legal_actions is None),
        "Action hints are served exactly when legal actions are.",
    )
    if hints is None or self.legal_actions is None:
      return
    offered = {action.id for action in self.legal_actions}
    hinted = [hint.id for hint in hints]
    contract.require(
        len(set(hinted)) == len(hinted) and set(hinted) <= offered,
        "Each action hint names a legal action of the frame, once.",
    )
