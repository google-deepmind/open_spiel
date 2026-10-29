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

"""Plays a game to its end through the session API, checking every frame.

A sweep opens a session whose seats are all played by seeded random
controllers under the `auto` chance policy, and advances it one step at a
time. At every position it builds the frame of the oracle, the spectator and
each seat, and checks each against the session's history replayed in a fresh
pyspiel state:

  * the frame survives a strict JSON round trip and carries every field;
  * it shows the position the history reaches, and never rests at a chance
    node;
  * the seat to move is offered the position's legal actions, each labelled;
  * in a game that hides information, every seat gets its own observation,
    and no seat or spectator gets the full state, the seed, chance outcomes,
    another seat's legal actions or any move but the seat's own;
  * at a terminal position, every seat's return is given.

At the end, every viewer's replays of the first, middle and last positions
are checked the same way: each shows the replayed position, and withholds
what the live frame would.

A game passes when it reaches a terminal position with every check holding.
`sweep_exceptions.py` reads the games the sweep is expected to fail on, and
says when a result disagrees with that list.

The verdict depends only on the game and the seed: play is seeded, and a game
that is not over is stopped after a fixed number of plies, never after a
fixed time. A wall-clock limit guards against a runner too slow to finish a
sweep at all; when it trips, the sweep ends with `TOO_SLOW`, which says
nothing about the game.
"""

from collections.abc import Callable
import dataclasses
import enum
import time
from typing import Protocol

from open_spiel.python.gui import contract
from open_spiel.python.gui import errors
from open_spiel.python.gui import frame as frame_lib
from open_spiel.python.gui import frame_checks

# The seed every sweep's session is opened with, so a sweep is reproducible.
SEED = 1234

# The plies, chance outcomes included, after which a game that is not over
# stops the sweep.
PLY_CAP = 1000

# The wall-clock seconds after which one game's sweep gives up on its runner,
# unless the caller passes a smaller limit. Far above what any game's sweep
# takes, so it trips only on a runner too slow to judge.
TIME_LIMIT_SECONDS = 300.0

# Returns the current time in seconds; only differences are used.
Clock = Callable[[], float]

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)


class Outcome(enum.StrEnum):
  """How a sweep ended.

  Attributes:
    PASSED: The game reached a terminal position with every check holding.
    REFUSED: The session could not be opened.
    PAUSED: A seat's controller failed.
    PLY_CAP: The game was not over after the ply cap.
    INVARIANT: A frame broke a check, or the session broke the API.
    ERROR: The session raised while playing, or the sweep's own replay of
      the game could not be set up.
    TOO_SLOW: The wall-clock limit ran out before any other outcome. This is
      a fact about the runner, not a verdict on the game.
  """

  PASSED = "passed"
  REFUSED = "refused"
  PAUSED = "paused"
  PLY_CAP = "ply_cap"
  INVARIANT = "invariant"
  ERROR = "error"
  TOO_SLOW = "too_slow"


class SweptSession(Protocol):
  """The part of the session API a sweep drives."""

  @property
  def history(self) -> tuple[int, ...]:
    """Every action of the live game, chance outcomes included."""

  @property
  def is_terminal(self) -> bool:
    """Whether the live game is over."""

  def frame(self, viewer: contract.Viewer) -> frame_lib.Frame:
    """Returns the live position, built for `viewer`."""

  def replay_frame(self, viewer: contract.Viewer, step: int) -> frame_lib.Frame:
    """Returns the position after `step` plies, built for `viewer`."""

  def advance(
      self, viewer: contract.Viewer, *, max_plies: int | None = ...
  ) -> frame_lib.Frame:
    """Lets the controllers play at most `max_plies` actions.

    Args:
      viewer: The viewer to build the returned frame for.
      max_plies: The most actions to play; None for no limit.

    Returns:
      The live position after play, built for `viewer`.

    Raises:
      errors.ControllerFailedError: If a controller fails; its seat pauses.
    """


# Opens a session of the named game with its default parameters, every seat
# played by a random controller and every chance outcome sampled, all seeded
# from `SEED`.
OpenSession = Callable[[str], SweptSession]


@dataclasses.dataclass(frozen=True)
class Result:
  """How one game's sweep ended.

  Attributes:
    game: The game's short name.
    outcome: How the sweep ended.
    detail: What went wrong, where the outcome alone does not say; else
      empty.
    plies: The plies played.
  """

  game: str
  outcome: Outcome
  detail: str
  plies: int

  @property
  def passed(self) -> bool:
    """Whether the game passed."""
    return self.outcome == Outcome.PASSED

  @property
  def judged(self) -> bool:
    """Whether the outcome is a verdict on the game, not on the runner."""
    return self.outcome != Outcome.TOO_SLOW

  def __str__(self) -> str:
    text = f"{self.game}: {self.outcome} after {self.plies} plies"
    return f"{text}: {self.detail}" if self.detail else text


@dataclasses.dataclass(frozen=True)
class _Timer:
  """The clock a sweep reads, when it started, and its limit in seconds."""

  clock: Clock
  start: float
  limit: float

  def elapsed(self) -> float:
    """Returns the seconds since the sweep started."""
    return self.clock() - self.start


def sweep_game(
    game_name: str,
    open_session: OpenSession,
    *,
    ply_cap: int = PLY_CAP,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
    clock: Clock = time.monotonic,
) -> Result:
  """Plays one game to its end, checking every frame on the way.

  Args:
    game_name: The game's short name, played with its default parameters.
    open_session: Opens the session to play.
    ply_cap: The plies after which a game that is not over stops the sweep.
    time_limit_seconds: The `clock` seconds after which the sweep stops with
      `TOO_SLOW`, unless another outcome is reached first.
    clock: Returns the current time in seconds.

  Returns:
    How the sweep ended. Anything the session raises ends the sweep and is
    reported in the result, never raised. Every outcome but `TOO_SLOW` is
    the same on every run with the same arguments.
  """
  start = clock()
  try:
    session = open_session(game_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return Result(game_name, Outcome.REFUSED, _describe(e), 0)
  try:
    replay = frame_checks.Replay(game_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return Result(game_name, Outcome.ERROR, f"Replay: {_describe(e)}", 0)
  try:
    return _play(
        game_name,
        session,
        replay,
        ply_cap,
        _Timer(clock, start, time_limit_seconds),
    )
  except frame_checks.InvariantError as e:
    return Result(game_name, Outcome.INVARIANT, str(e), replay.num_plies)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return Result(game_name, Outcome.ERROR, _describe(e), replay.num_plies)


def _play(
    game_name: str,
    session: SweptSession,
    replay: frame_checks.Replay,
    ply_cap: int,
    timer: _Timer,
) -> Result:
  """Advances `session` one step at a time until it ends or a limit is hit."""
  while True:
    replay.follow(session.history)
    oracle = _check_viewers(session, replay)
    plies = replay.num_plies
    if oracle.status == contract.SessionStatus.PAUSED:
      return Result(game_name, Outcome.PAUSED, _failures(oracle), plies)
    frame_checks.require(
        session.is_terminal == replay.state.is_terminal(),
        "The session and its history disagree on whether the game is over.",
    )
    if session.is_terminal:
      _check_replays(session, replay)
      return Result(game_name, Outcome.PASSED, "", plies)
    if plies >= ply_cap:
      return Result(game_name, Outcome.PLY_CAP, "", plies)
    elapsed = timer.elapsed()
    if elapsed > timer.limit:
      return Result(
          game_name,
          Outcome.TOO_SLOW,
          f"{elapsed:.1f} s elapsed, over the {timer.limit:.1f} s limit; the "
          "runner is too slow to finish the sweep, and this is not a "
          "verdict on the game.",
          plies,
      )
    try:
      session.advance(_ORACLE, max_plies=1)
    except errors.ControllerFailedError:
      # The next frame shows the paused seat and its failure.
      frame_checks.require(
          session.frame(_ORACLE).status == contract.SessionStatus.PAUSED,
          f"Advancing at ply {plies} raised ControllerFailedError, but the "
          "session is not paused.",
      )
      continue
    frame_checks.require(
        len(session.history) > plies,
        f"Advancing at ply {plies} applied no action.",
    )


def _viewers(replay: frame_checks.Replay) -> list[contract.Viewer]:
  """Returns the oracle, the spectator and every seat of the game."""
  return [_ORACLE, _SPECTATOR] + [
      contract.Viewer(contract.ViewerKind.SEAT, seat)
      for seat in range(replay.num_seats)
  ]


def _check_viewers(
    session: SweptSession, replay: frame_checks.Replay
) -> frame_lib.Frame:
  """Checks the live frame of every viewer, and returns the oracle's."""
  frames = {viewer: session.frame(viewer) for viewer in _viewers(replay)}
  for viewer, frame in frames.items():
    frame_checks.check_live_frame(frame, viewer, replay)
  return frames[_ORACLE]


def _check_replays(session: SweptSession, replay: frame_checks.Replay) -> None:
  """Checks every viewer's replays of the first, middle and last positions."""
  for step in sorted({0, replay.num_plies // 2, replay.num_plies}):
    for viewer in _viewers(replay):
      frame = session.replay_frame(viewer, step)
      frame_checks.check_replay_frame(frame, viewer, step, replay)


def _failures(frame: frame_lib.Frame) -> str:
  """Describes every failed controller of a paused frame."""
  return "; ".join(
      f"seat {seat.seat}: {seat.failure.error}: {seat.failure.message}"
      for seat in frame.seats
      if seat.failure is not None
  )


def _describe(error: Exception) -> str:
  """Names an exception and gives its message."""
  return f"{type(error).__name__}: {error}"
