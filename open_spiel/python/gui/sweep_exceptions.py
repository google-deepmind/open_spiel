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

"""The games the sweep is expected to fail on, and how.

`sweep_exceptions.json` lists each such game with where it fails, how, and
why. `disagreement` compares a sweep's result with that list: an unlisted game
must pass, and a listed game must fail exactly as listed, so a new failure of
a listed game (a leak in a game that only ran out of plies, say) is not
hidden by its entry.
"""

from collections.abc import Mapping
import dataclasses
import enum
import json
import os
from typing import Any

from open_spiel.python.gui import sweep

# The checked-in list of the games the sweep is expected to fail on.
EXCEPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sweep_exceptions.json"
)


class FailureKind(enum.StrEnum):
  """Where a listed game is expected to fail.

  Attributes:
    UNAVAILABLE: The catalogue marks the game not loadable, so the server
      refuses to open a session of it.
    BACKEND: The server opens a session of the game, but the sweep fails on
      it.
    BROWSER: The sweep passes, but a client that plays the game through the
      server's API fails on it; this sweep expects the game to pass.
  """

  UNAVAILABLE = "unavailable"
  BACKEND = "backend"
  BROWSER = "browser"


class ExpectedOutcome(enum.StrEnum):
  """How the sweep of a game listed as a backend failure is expected to end.

  Attributes:
    UNFINISHED: The game is not over at the ply cap.
    INVARIANT: A frame breaks a check.
    PAUSED: A seat's controller fails.
    ERROR: The session raises while playing.
  """

  UNFINISHED = "unfinished"
  INVARIANT = "invariant"
  PAUSED = "paused"
  ERROR = "error"

  def matches(self, outcome: sweep.Outcome) -> bool:
    """Whether a sweep that ended with `outcome` ended this way."""
    if self == ExpectedOutcome.UNFINISHED:
      return outcome == sweep.Outcome.PLY_CAP
    return outcome == sweep.Outcome(self.value)


@dataclasses.dataclass(frozen=True)
class ExpectedFailure:
  """A listed game, and how and why it fails.

  Attributes:
    kind: Where the game is expected to fail.
    reason: Why it fails.
    outcome: How the sweep ends, for a backend failure; else None.
  """

  kind: FailureKind
  reason: str
  outcome: ExpectedOutcome | None = None

  def __post_init__(self) -> None:
    if (self.kind == FailureKind.BACKEND) != (self.outcome is not None):
      raise ValueError("A backend failure, and only one, has an outcome.")

  def __str__(self) -> str:
    where = f"{self.kind} ({self.outcome})" if self.outcome else self.kind
    return f"{where}: {self.reason}"


def load_exceptions(path: str = EXCEPTIONS_PATH) -> dict[str, ExpectedFailure]:
  """Reads the games the sweep is expected to fail on.

  Args:
    path: A JSON object mapping each game's short name, in order and once, to
      an object with the `kind` of failure and its `reason`, and for a
      backend failure the `outcome` the sweep ends with.

  Returns:
    The expected failure of each listed game, by game.

  Raises:
    ValueError: If the file is not such an object.
  """

  def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
      raise ValueError(f"{path} lists {names} with a repeated name.")
    return dict(pairs)

  with open(path, encoding="utf-8") as f:
    listed = json.load(f, object_pairs_hook=unique)
  if not isinstance(listed, dict):
    raise ValueError(f"{path} is not an object keyed by game.")
  if list(listed) != sorted(listed):
    raise ValueError(f"{path} does not list its games in order.")
  return {
      game: _expected_failure(path, game, entry)
      for game, entry in listed.items()
  }


def _expected_failure(path: str, game: str, entry: Any) -> ExpectedFailure:
  """Reads one listed game's expected failure."""
  backend = isinstance(entry, dict) and entry.get("kind") == "backend"
  fields = {"kind", "reason", "outcome"} if backend else {"kind", "reason"}
  if (
      not isinstance(entry, dict)
      or set(entry) != fields
      or entry["kind"] not in set(FailureKind)
      or not isinstance(entry["reason"], str)
      or not entry["reason"].strip()
      or (backend and entry["outcome"] not in set(ExpectedOutcome))
  ):
    raise ValueError(
        f"{path} lists {game} without exactly a known kind and a reason, "
        "and for a backend failure a known outcome."
    )
  return ExpectedFailure(
      FailureKind(entry["kind"]),
      entry["reason"],
      ExpectedOutcome(entry["outcome"]) if backend else None,
  )


def disagreement(
    result: sweep.Result, exceptions: Mapping[str, ExpectedFailure]
) -> str | None:
  """Says how a sweep's result disagrees with the exceptions, if it does.

  A game listed as unavailable must be refused. A game listed as a backend
  failure must be opened and then end with its listed outcome. Every other
  game must pass.

  Args:
    result: How a game's sweep ended; a verdict on the game, not `TOO_SLOW`.
    exceptions: The expected failure of each listed game, by game.

  Returns:
    How the result disagrees with the exceptions, or None when it agrees.

  Raises:
    ValueError: If the result is not a verdict on the game.
  """
  if not result.judged:
    raise ValueError(f"{result}; it neither agrees nor disagrees.")
  expected = exceptions.get(result.game)
  if expected is None or expected.kind == FailureKind.BROWSER:
    agrees = result.passed
  elif expected.kind == FailureKind.UNAVAILABLE:
    agrees = result.outcome == sweep.Outcome.REFUSED
  else:
    assert expected.outcome is not None
    agrees = expected.outcome.matches(result.outcome)
  if agrees:
    return None
  if expected is None:
    return f"{result}. Fix it, or list it in the exceptions with the reason."
  return f"{result}, but it is listed as {expected}"
