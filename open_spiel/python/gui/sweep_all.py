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

"""Sweeps every registered game, or the named ones, through the server.

Usage:
  python -m open_spiel.python.gui.sweep_all [GAME ...]

Each game must pass the sweep (`sweep.py`), or be listed in
`sweep_exceptions.json` with how and why it does not. A listed game that
passes, or fails another way, is a problem too, so the list never outlives
its reasons. A whole run also checks that the games listed as unavailable are
exactly those the catalogue marks not loadable.

A whole run takes minutes, so it is not part of the test suite, which sweeps
a few representative games in `games_smoke_test.py`. Run it after changing
the server, the list or a game. It prints a line per game and a summary, and
exits with status 1 if any game is a problem or ran out of time.
"""

import argparse
from collections.abc import Sequence

# Imported to register the games written in Python.
from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import loader
from open_spiel.python.gui import session
from open_spiel.python.gui import sweep
from open_spiel.python.gui import sweep_census
from open_spiel.python.gui import sweep_exceptions
import pyspiel


def open_random_session(game_name: str) -> session.Session:
  """Opens a session of the game with every seat played at random.

  Args:
    game_name: A registered short name, played with its default parameters.

  Returns:
    The session, seeded from `sweep.SEED`, with every chance outcome sampled.
  """
  num_seats = pyspiel.load_game(game_name).num_players()
  return session.Session.create(
      game_name,
      {},
      seed=sweep.SEED,
      chance_policy=contract.ChancePolicy.AUTO,
      controllers=[controllers.Spec(controllers.RANDOM)] * num_seats,
  )


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the sweep and prints its verdicts.

  Args:
    argv: The arguments after the program name; the process's when None.

  Returns:
    The exit status: 0 when every game swept agrees with the list, else 1.
  """
  parser = argparse.ArgumentParser(
      prog="python -m open_spiel.python.gui.sweep_all",
      description=(
          "Sweep OpenSpiel games through the GUI server's sessions, checking "
          "every frame, and judge each against sweep_exceptions.json."
      ),
  )
  parser.add_argument(
      "games",
      nargs="*",
      metavar="GAME",
      help="The games to sweep; every registered game when none is named.",
  )
  args = parser.parse_args(argv)
  registered = sorted(pyspiel.registered_names())
  unknown = sorted(set(args.games) - set(registered))
  if unknown:
    parser.error(f"not registered: {', '.join(unknown)}")
  census = sweep_census.Census(sweep_exceptions.load_exceptions(), registered)
  for game_name in args.games or registered:
    result = sweep.sweep_game(game_name, open_random_session)
    problem = census.record(result)
    line = f"ok    {result}" if problem is None else f"FAIL  {problem}"
    print(line, flush=True)
  if not args.games:
    problem = census.check_catalogue([
        entry.name
        for entry in loader.build_catalogue().games
        if not entry.loadable
    ])
    if problem is not None:
      print(f"FAIL  {problem}", flush=True)
  for line in census.summary():
    print(line, flush=True)
  return 0 if census.passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
