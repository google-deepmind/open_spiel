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

"""A session: one live game, its seats, and every frame built from it.

A session owns the live state and the history that reached it. Every earlier
position, and every position after an undo or a reset, is rebuilt from the
history. Viewers act through the session, which enforces whose turn it is and
what each viewer may see; controllers play the seats the client does not.

A session is safe to share between threads: every method holds its lock.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import random
import secrets
import threading

from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers as controllers_lib
from open_spiel.python.gui import errors
from open_spiel.python.gui import frame as frame_lib
from open_spiel.python.gui import loader
from open_spiel.python.gui import plies
from open_spiel.python.gui import request
from open_spiel.python.gui import seats
from open_spiel.python.gui import views
import pyspiel


class ReplayUnsupportedError(errors.GameUnavailableError):
  """The game does not reproduce its positions from its history."""


class Session:
  """One live game, played by human and controller seats.

  Every refusal raises an `errors.RequestError`. Every public method that
  reads or changes the game takes the viewer it acts for, and raises
  `NotFoundError` for a seat the game does not have, and
  `ViewerUnavailableError` for a seat of a game that gives its players no view
  of their own.

  Under the `auto` policy, the n-th chance outcome the session samples is
  drawn from a seed derived from the session's seed and n. The count never
  rewinds: after an undo or a reset, a chance node is drawn afresh, so a seat
  cannot see an outcome, take back its action, and choose again knowing it.

  Attributes:
    id: The session's identifier.
    game_name: The game's short name.
  """

  def __init__(
      self,
      loaded: loader.LoadedGame,
      *,
      seed: int,
      chance_policy: contract.ChancePolicy,
  ) -> None:
    """Starts a session with every seat human; use `create` instead."""
    self.id = secrets.token_hex(8)
    self.game_name = loaded.name
    self._loaded = loaded
    self._seed = seed
    self._chance_policy = chance_policy
    self._info = contract.SessionInfo(
        id=self.id,
        game=loaded.name,
        parameters=dict(loaded.parameters),
        num_seats=loaded.num_seats,
        chance_policy=chance_policy,
        turn_based=loaded.turn_based,
        hides_information=loaded.hides_information,
        seed=seed,
        controllers=tuple(
            dataclasses.replace(
                descriptor.controller,
                unavailable_reason=descriptor.unavailable_reason,
            )
            for descriptor in controllers_lib.descriptors(loaded.game)
        ),
    )
    self._seats = seats.Seats(
        loaded.num_seats, loaded.game, seed=seed, on_built=self._built
    )
    self._state = loaded.game.new_initial_state()
    self._plies: list[plies.Ply] = []
    # How many chance outcomes the session has sampled; never rewound.
    self._draws = 0
    self._lock = threading.RLock()

  @classmethod
  def create(
      cls,
      game_name: str,
      parameters: Mapping[str, contract.ParameterValue],
      *,
      seed: int | None = None,
      chance_policy: contract.ChancePolicy,
      controllers: Sequence[controllers_lib.Spec],
      autoplay: Sequence[bool] = (),
  ) -> "Session":
    """Starts a session of a game.

    Args:
      game_name: A registered short name.
      parameters: Game parameters; omitted ones take their defaults.
      seed: Seeds every chance outcome and controller; None: a random seed.
      chance_policy: How chance nodes are resolved.
      controllers: Who plays each seat, in seat order.
      autoplay: Whether each seat's controller plays as soon as it is to
        move, in seat order; empty for none.

    Returns:
      The session, at its first position that needs a choice.

    Raises:
      NotFoundError: If no game is registered under `game_name`.
      GameUnavailableError: If the server cannot serve the game.
      BadRequestError: If the game refuses `parameters`, or the seats do not
        match the game's.
      ControllerUnavailableError: If a controller cannot play the game.
    """
    loaded = loader.load(game_name, parameters)
    num_seats = loaded.num_seats
    if len(controllers) != num_seats or len(autoplay) not in (0, num_seats):
      raise errors.BadRequestError(f"{game_name} has {num_seats} seats.")
    session = cls(
        loaded,
        seed=seats.random_seed() if seed is None else seed,
        chance_policy=chance_policy,
    )
    with session._lock:  # A build may finish on its own thread meanwhile.
      session._seats.assign(
          dict(enumerate(zip(controllers, autoplay or [False] * num_seats)))
      )
      session._settle(request.MAX_ADVANCE_PLIES, autoplay_only=True)
    return session

  @property
  def history(self) -> tuple[int, ...]:
    """Every action of the live game, chance outcomes included."""
    with self._lock:
      return tuple(ply.action for ply in self._plies)

  @property
  def is_terminal(self) -> bool:
    """Whether the live game is over."""
    with self._lock:
      return bool(self._state.is_terminal())

  def frame(self, viewer: contract.Viewer) -> frame_lib.Frame:
    """Returns the live position, as `viewer` may see it."""
    with self._lock:
      self._check_viewer(viewer)
      return self._build(self._state, len(self._plies), viewer)

  def replay_frame(self, viewer: contract.Viewer, step: int) -> frame_lib.Frame:
    """Returns the position after `step` plies, as `viewer` may see it.

    The position is rebuilt from the history; the session does not change.

    Args:
      viewer: The viewer to build the frame for.
      step: The number of plies of the live game to replay.

    Raises:
      NotFoundError: If `step` is not a ply of the live game.
      ReplayUnsupportedError: If the rebuilt position differs from play.
    """
    with self._lock:
      self._check_viewer(viewer)
      if not 0 <= step <= len(self._plies):
        raise errors.NotFoundError(f"There is no ply {step}.", ply=step)
      state, _ = self._rebuild(step)
      return self._build(state, step, viewer)

  def apply_action(
      self, viewer: contract.Viewer, action: int, *, ply: int | None = None
  ) -> frame_lib.Frame:
    """Plays `action` for the human seat to move, on behalf of `viewer`.

    A seat viewer plays its own seat; the oracle plays any human seat.
    Autoplaying seats then move. `ply` is where to play: None, or the plies
    played, for the live position; an earlier ply plays from there, taking
    back every later ply, as only an omniscient viewer may.

    Args:
      viewer: The viewer on whose behalf the action is played.
      action: The OpenSpiel action id to play.
      ply: Where to play, as described above; None for the live position.

    Returns:
      The live position after play, as `viewer` may see it.

    Raises:
      ControllerFailedError: If the session is paused, or an autoplaying
        seat's controller fails; the actions applied stand.
      NotFoundError: If `ply` is not a ply of the live game.
      NotYourTurnError: If `viewer` may not choose the next action there.
      IllegalActionError: If `action` is not legal there; nothing changes.
    """
    with self._lock:
      self._check_viewer(viewer)
      state, played = self._position_to_play(viewer, ply)
      seat = plies.to_move(state).seat
      if seat is None or not self._seats[seat].is_human:
        raise errors.NotYourTurnError("No human seat is to move.", seat=seat)
      if viewer.kind != contract.ViewerKind.ORACLE and viewer.seat != seat:
        raise errors.NotYourTurnError(f"Seat {seat} is to move.", seat=seat)
      self._play_from(state, played, action)
      return self._build(self._state, len(self._plies), viewer)

  def apply_chance(
      self, viewer: contract.Viewer, outcome: int, *, ply: int | None = None
  ) -> frame_lib.Frame:
    """Chooses the outcome at a chance node under the `expose` policy.

    Only an omniscient viewer may choose, since an outcome may be hidden.
    `ply` is where to choose, as for `apply_action`.

    Args:
      viewer: The viewer choosing the outcome.
      outcome: The action id of the chance outcome to apply.
      ply: Where to choose, as for `apply_action`; None for the live position.

    Returns:
      The live position after play, as `viewer` may see it.

    Raises:
      NotFoundError: If `ply` is not a ply of the live game.
      NotYourTurnError: If chance is not to move there, the policy is
        `auto`, or `viewer` is not omniscient.
      IllegalActionError: If `outcome` is not possible there; nothing changes.
      ControllerFailedError: If an autoplaying seat's controller then fails.
    """
    with self._lock:
      self._check_viewer(viewer)
      if self._chance_policy != contract.ChancePolicy.EXPOSE:
        raise errors.NotYourTurnError("The server resolves chance nodes.")
      self._check_omniscient(viewer, "chooses chance outcomes")
      state, played = self._position_to_play(viewer, ply)
      if not state.is_chance_node():
        raise errors.NotYourTurnError("Chance is not to move.")
      self._play_from(state, played, outcome)
      return self._build(self._state, len(self._plies), viewer)

  def advance(
      self, viewer: contract.Viewer, *, max_plies: int | None = 1000
  ) -> frame_lib.Frame:
    """Lets controllers play their seats, whether or not they autoplay.

    Play stops when a human seat or an exposed chance node must act, the game
    ends, a controller fails, or `max_plies` seat actions (None: no limit)
    have been applied. Under the `auto` policy, chance outcomes are sampled as
    they come and do not count, so play never stops at a chance node. A
    controller that fails pauses its seat; nothing plays instead.

    Args:
      viewer: The viewer to build the returned frame for.
      max_plies: The most seat actions to apply; None for no limit.

    Returns:
      The live position after play, as `viewer` may see it.

    Raises:
      ControllerFailedError: If the session is paused, or a controller
        fails; the actions applied stand.
    """
    with self._lock:
      self._check_viewer(viewer)
      self._check_not_paused()
      self._settle(max_plies, autoplay_only=False)
      self._check_not_paused()
      return self._build(self._state, len(self._plies), viewer)

  def undo(self, viewer: contract.Viewer) -> frame_lib.Frame:
    """Takes back an action a seat chose, and every ply after it.

    The plies after it are chance outcomes and controllers' actions. A seat
    viewer takes back its own most recent action, unless a seat the client
    plays has acted since: that action is its seat's to take back. The
    oracle takes back the most recent action of any seat. Undoing resumes a
    paused session, and plays nothing.

    Args:
      viewer: The viewer taking back an action.

    Returns:
      The live position after the undo, as `viewer` may see it.

    Raises:
      NothingToUndoError: If `viewer` has no action to take back; a
        spectator never has.
    """
    with self._lock:
      self._check_viewer(viewer)
      step = self._last_action(viewer)
      if step is None:
        raise errors.NothingToUndoError("There is no action to undo.")
      self._state, rebuilt = self._rebuild(step)
      self._plies = list(rebuilt)
      self._seats.clear_failures()
      return self._build(self._state, step, viewer)

  def reset(
      self, viewer: contract.Viewer, *, history: Sequence[int] = ()
  ) -> frame_lib.Frame:
    """Restarts the live game from `history`, keeping the seats.

    Whoever the page shows the game to sets it up, so any viewer but a
    spectator, who only watches, resets. Resets resume a paused session; then
    chance and autoplaying seats move, and a failing controller pauses it.

    Args:
      viewer: The viewer resetting the game.
      history: The actions, chance outcomes included, to apply from the initial
        state.

    Returns:
      The live position after the reset and any autoplay, as `viewer` may see
      it.

    Raises:
      NotYourTurnError: If `viewer` is a spectator.
      IllegalActionError: If an action of `history` is illegal; nothing changes.
    """
    with self._lock:
      self._check_viewer(viewer)
      self._check_participant(viewer, "resets the game")
      self._state, rebuilt = plies.rebuild(self._loaded.game, history)
      self._plies = list(rebuilt)
      self._seats.clear_failures()
      self._settle(request.MAX_ADVANCE_PLIES, autoplay_only=True)
      return self._build(self._state, len(self._plies), viewer)

  def set_controller(
      self,
      viewer: contract.Viewer,
      seat: int,
      spec: controllers_lib.Spec,
      *,
      autoplay: bool = False,
  ) -> frame_lib.Frame:
    """Changes who plays `seat`, resuming it if its controller had failed.

    As with a reset, any viewer but a spectator changes it. Autoplaying seats
    then move as usual; if a controller fails, the frame shows it paused. A
    controller still being built leaves the seat preparing (see `builds`).

    Args:
      viewer: The viewer changing the seat.
      seat: The seat to change.
      spec: The controller to play the seat.
      autoplay: Whether the controller plays as soon as the seat is to move.

    Returns:
      The live position after any autoplay, as `viewer` may see it.

    Raises:
      NotYourTurnError: If `viewer` is a spectator.
      NotFoundError: If `seat` is not a seat of the game.
      ControllerUnavailableError: If the controller cannot play the game, or
        a human seat is asked to autoplay.
    """
    with self._lock:
      self._check_viewer(viewer)
      self._check_participant(viewer, "changes who plays a seat")
      if not 0 <= seat < len(self._seats):
        raise errors.NotFoundError(f"There is no seat {seat}.", seat=seat)
      self._seats.assign({seat: (spec, autoplay)})
      self._settle(request.MAX_ADVANCE_PLIES, autoplay_only=True)
      return self._build(self._state, len(self._plies), viewer)

  def check_closer(self, viewer: contract.Viewer) -> None:
    """Refuses a viewer that may not close the session.

    As with a reset, any viewer but a spectator closes it.

    Raises:
      NotFoundError: If `viewer` is a seat the game does not have.
      ViewerUnavailableError: If `viewer` is a seat of a game without seat
        views.
      NotYourTurnError: If `viewer` is a spectator.
    """
    with self._lock:
      self._check_viewer(viewer)
      self._check_participant(viewer, "closes the session")

  def _built(
      self, index: int, build: builds.Build[controllers_lib.Controller]
  ) -> None:
    """Finishes the seat `build` prepares, if it still does, and plays on."""
    with self._lock:
      seat = self._seats[index]
      if seat.build is not build:
        return  # Finished in its request, or superseded.
      seat.finish(index, refuse=False)
      self._settle(request.MAX_ADVANCE_PLIES, autoplay_only=True)

  def _position_to_play(
      self, viewer: contract.Viewer, ply: int | None
  ) -> tuple[pyspiel.State, list[plies.Ply]]:
    """Returns the live or a rebuilt earlier position, and its plies."""
    if ply is None or ply == len(self._plies):
      self._check_not_paused()
      return self._state, list(self._plies)
    if not 0 <= ply < len(self._plies):
      raise errors.NotFoundError(f"There is no ply {ply}.", ply=ply)
    self._check_omniscient(viewer, "plays from an earlier position")
    state, rebuilt = self._rebuild(ply)
    return state, list(rebuilt)

  def _play_from(
      self, state: pyspiel.State, played: list[plies.Ply], action: int
  ) -> None:
    """Applies `action` at `state`, from `_position_to_play`, making it live.

    An illegal action changes nothing. Playing from an earlier position
    resumes a paused session. Then chance and autoplaying seats move.

    Args:
      state: The position to play at: the live state or a rebuilt earlier one.
      played: The plies that reach `state`; the ply of `action` is appended to
        it.
      action: The action or chance outcome to apply.
    """
    branched = state is not self._state
    played.append(plies.apply(state, action, ply=len(played)))
    self._state, self._plies = state, played
    if branched:
      self._seats.clear_failures()
    self._settle(request.MAX_ADVANCE_PLIES, autoplay_only=True)
    self._check_not_paused()

  def _check_viewer(self, viewer: contract.Viewer) -> None:
    """Refuses a viewer the game cannot show anything it may see."""
    if viewer.seat is not None and not 0 <= viewer.seat < len(self._seats):
      raise errors.NotFoundError(
          f"There is no seat {viewer.seat}.", seat=viewer.seat
      )
    if viewer.seat is not None and not self._loaded.has_seat_views:
      raise errors.ViewerUnavailableError(
          f"{self.game_name} has no seat views.", seat=viewer.seat
      )

  def _check_omniscient(self, viewer: contract.Viewer, what: str) -> None:
    """Refuses a viewer that may not see the full state."""
    if not views.is_omniscient(self._loaded, viewer):
      raise errors.NotYourTurnError(
          f"{self.game_name} hides information, so only the oracle {what}."
      )

  def _check_participant(self, viewer: contract.Viewer, what: str) -> None:
    """Refuses a spectator, who watches the game and never changes it."""
    if viewer.kind == contract.ViewerKind.SPECTATOR:
      raise errors.NotYourTurnError(f"A spectator never {what}.")

  def _check_not_paused(self) -> None:
    """Refuses to move a paused session."""
    failed = self._seats.failed
    if failed is not None:
      raise errors.ControllerFailedError(seat=failed)

  def _apply(self, action: int) -> None:
    """Applies a legal action or chance outcome to the live state."""
    self._plies.append(plies.apply(self._state, action, ply=len(self._plies)))

  def _resolve_chance(self) -> None:
    """Samples chance outcomes while the `auto` policy has one due."""
    while (
        self._chance_policy == contract.ChancePolicy.AUTO
        and self._state.is_chance_node()
    ):
      seed = seats.derive_seed(self._seed, "chance", self._draws)
      self._draws += 1
      self._apply(plies.sample_chance(self._state, random.Random(seed)))

  def _settle(self, max_plies: int | None, *, autoplay_only: bool) -> None:
    """Resolves automatic chance and lets seats move, as `advance` describes.

    Only autoplaying seats move if `autoplay_only`. Stopping after `max_plies`
    seat actions (never, if None) bounds how long autoplay holds the lock.

    Args:
      max_plies: The most seat actions to apply; None for no limit.
      autoplay_only: Whether only autoplaying seats move.
    """
    played = 0
    self._resolve_chance()
    while played != max_plies and self._play_seat(autoplay_only=autoplay_only):
      played += 1
      self._resolve_chance()

  def _play_seat(self, *, autoplay_only: bool) -> bool:
    """Lets the controller of the seat to move play; returns whether it did.

    Nothing plays at a chance node, at the end of the game, while a seat is
    paused, or when the seat to move is a human's, is preparing, or (if
    `autoplay_only`) does not autoplay.

    Args:
      autoplay_only: Whether only an autoplaying seat may move.

    Returns:
      Whether the seat's controller applied an action.
    """
    state = self._state
    if (
        state.is_terminal()
        or state.is_chance_node()
        or self._seats.failed is not None
    ):
      return False
    index = plies.to_move(state).seat
    assert index is not None
    seat = self._seats[index]
    if not seat.moves(autoplay_only=autoplay_only):
      return False
    action = seat.choose_action(state, index)
    if action is None:
      return False
    self._apply(action)
    return True

  def _last_action(self, viewer: contract.Viewer) -> int | None:
    """Returns the index of the last ply `viewer` may take back, or None."""
    for index in reversed(range(len(self._plies))):
      seat = self._plies[index].seat
      if seat is None:
        continue
      if seat == viewer.seat or viewer.kind == contract.ViewerKind.ORACLE:
        return index
      if self._seats[seat].is_human:
        return None  # Another person has acted since.
    return None

  def _rebuild(self, step: int) -> tuple[pyspiel.State, tuple[plies.Ply, ...]]:
    """Rebuilds the position after `step` plies, checking it against play."""
    try:
      return plies.rebuild(
          self._loaded.game,
          [ply.action for ply in self._plies[:step]],
          expected=self._plies[:step],
      )
    except plies.DriftError as error:
      # The error's own text shows the plies, hidden ones included; the
      # refusal names only the ply's index, which every viewer may know.
      raise ReplayUnsupportedError(
          f"{self.game_name} cannot replay its history: rebuilding ply "
          f"{error.ply} does not reproduce it."
      ) from error

  def _build(
      self, state: pyspiel.State, step: int, viewer: contract.Viewer
  ) -> frame_lib.Frame:
    """Builds the frame of the position after `step` plies."""
    seat_views = self._seats.views()
    position = views.Position(
        info=self._info,
        loaded=self._loaded,
        status=views.status(self._state, seat_views),
        seats=seat_views,
        state=state,
        ply=step,
        history=tuple(self._plies),
    )
    return views.build(position, viewer)
