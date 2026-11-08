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

"""Judging the all-games sweep: each game's verdict, and the run's summary.

`sweep_all.py` sweeps the registered games and records each result here. The
judging is tested on its own in `sweep_census_test.py`, with fake sweeps, since
the all-games sweep takes minutes and is not part of the test suite.
"""

from collections.abc import Mapping, Sequence

from open_spiel.python.gui import sweep
from open_spiel.python.gui import sweep_exceptions


def split_by_registration(
    exceptions: Mapping[str, sweep_exceptions.ExpectedFailure],
    registered: Sequence[str],
) -> tuple[dict[str, sweep_exceptions.ExpectedFailure], list[str]]:
  """Splits the listed games by whether this build registers them.

  Args:
    exceptions: The expected failure of each listed game, by game.
    registered: The short names of the games this build registers.

  Returns:
    The entries of the registered listed games, and the sorted names of the
    listed games this build does not register.
  """
  names = set(registered)
  present = {
      game: expected for game, expected in exceptions.items() if game in names
  }
  absent = sorted(set(exceptions) - names)
  return present, absent


class Census:
  """The verdicts of one run of the sweep.

  A game whose sweep is not a verdict, because the runner ran out of time,
  is not judged, and counts against the run as much as a disagreement does:
  a run passes only when every game it swept was judged and agreed with the
  list.

  Attributes:
    exceptions: The entries of the listed games this build registers.
    unregistered: The listed games this build does not register; they are
      named, not judged.
    num_swept: The games recorded.
    problems: How each game that disagreed with the list did, in order.
    not_judged: The games whose sweeps ran out of time, in order.
  """

  def __init__(
      self,
      exceptions: Mapping[str, sweep_exceptions.ExpectedFailure],
      registered: Sequence[str],
  ) -> None:
    self.exceptions, self.unregistered = split_by_registration(
        exceptions, registered
    )
    self.num_swept = 0
    self.problems: list[str] = []
    self.not_judged: list[str] = []

  @property
  def passed(self) -> bool:
    """Whether every game recorded was judged and agreed with the list."""
    return not self.problems and not self.not_judged

  def record(self, result: sweep.Result) -> str | None:
    """Judges one game's sweep against the list.

    Args:
      result: How the game's sweep ended.

    Returns:
      Why the result counts against the run, or None when it agrees with
      the list.
    """
    self.num_swept += 1
    if not result.judged:
      self.not_judged.append(result.game)
      return f"{result}. Not judged: rerun it on a faster machine."
    problem = sweep_exceptions.disagreement(result, self.exceptions)
    if problem is not None:
      self.problems.append(problem)
    return problem

  def check_catalogue(self, unloadable: Sequence[str]) -> str | None:
    """Checks the games listed as unavailable against the catalogue.

    Args:
      unloadable: The games the catalogue marks not loadable.

    Returns:
      How the two differ, or None when they name the same games. A
      difference counts against the run.
    """
    unavailable = sweep_exceptions.FailureKind.UNAVAILABLE
    listed = sorted(
        game
        for game, expected in self.exceptions.items()
        if expected.kind == unavailable
    )
    catalogued = sorted(unloadable)
    if listed == catalogued:
      return None
    problem = (
        "The games listed as unavailable are not those the catalogue marks "
        f"not loadable: listed only: {sorted(set(listed) - set(catalogued))}; "
        f"catalogue only: {sorted(set(catalogued) - set(listed))}"
    )
    self.problems.append(problem)
    return problem

  def summary(self) -> list[str]:
    """Returns the lines that end a run, the verdict last."""
    unregistered = ", ".join(self.unregistered) or "none"
    not_judged = ", ".join(self.not_judged) or "none"
    verdict = "PASSED" if self.passed else "FAILED"
    return [
        f"Listed but not registered in this build: {unregistered}",
        f"Not judged, out of time: {not_judged}",
        (
            f"{verdict}: {self.num_swept} swept, "
            f"{len(self.problems)} problems, {len(self.not_judged)} not "
            "judged"
        ),
    ]
