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


"""A stand-in for a bot that plays a trained checkpoint, for trying and tests.

  python -m open_spiel.python.gui.stand_in_checkpoint_bot write /tmp/ckpt \\
      --game gin_rummy --step 3
  python -m open_spiel.python.gui.play --game gin_rummy \\
      --bot open_spiel.python.gui.stand_in_checkpoint_bot:make_bot \\
      --checkpoint /tmp/ckpt@3

A checkpoint "<dir>@<step>" is the file <dir>/<step>/stand_in.json,
{"game": <short name>, "load_seconds": <seconds>}, which `write` makes;
loading it takes `load_seconds` (0 unless given), as a trained checkpoint's
load and warm-up does. The bot plays it the way a
checkpoint bot does: the checkpoint named in a request's `config.checkpoint`
(a Remote bot seat's `checkpoint` setting), else the one in the environment
variable CHECKPOINT_ENV, loaded once each; a request for another game, or
naming no checkpoint, is refused. It plays a legal action drawn from the
request's seed, and its notice names the checkpoint, so a test can see which
one answered. It takes either state mode and needs nothing beyond OpenSpiel.
"""

import argparse
from collections.abc import Mapping, Sequence
import json
import os
import pathlib
import random
import threading
import time
from typing import Any

from open_spiel.python.gui import remote_bot_server

# The variable naming the checkpoint a request's config does not.
CHECKPOINT_ENV = remote_bot_server.CHECKPOINT_ENV

# The file of a step that holds its game.
FILE_NAME = "stand_in.json"


def write(
    directory: str, step: int, game: str, load_seconds: float = 0.0
) -> str:
  """Writes a stand-in checkpoint; returns its "<dir>@<step>" name."""
  path = pathlib.Path(directory) / str(step)
  path.mkdir(parents=True, exist_ok=True)
  body = {"game": game, "load_seconds": load_seconds}
  (path / FILE_NAME).write_text(json.dumps(body) + "\n")
  return f"{directory}@{step}"


def read(checkpoint: str) -> str:
  """Loads a "<dir>@<step>" stand-in checkpoint; returns the game it plays.

  Loading takes the checkpoint's `load_seconds`.

  Raises:
    ValueError: If `checkpoint` is not "<dir>@<step>", or there is no
      stand-in checkpoint there.
  """
  directory, sep, step = checkpoint.rpartition("@")
  if not sep or not directory or not step.isdigit():
    raise ValueError(f"The checkpoint {checkpoint!r} is not <dir>@<step>.")
  path = pathlib.Path(directory) / step / FILE_NAME
  try:
    body = json.loads(path.read_text())
    game = body["game"]
    load_seconds = float(body.get("load_seconds", 0.0))
  except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
    raise ValueError(f"There is no stand-in checkpoint at {checkpoint}.") from e
  if not isinstance(game, str) or not game:
    raise ValueError(f"The stand-in checkpoint {checkpoint} names no game.")
  if not 0.0 <= load_seconds <= 600.0:
    raise ValueError(
        f"The stand-in checkpoint {checkpoint} loads in {load_seconds:g} s, "
        "not 0 to 600."
    )
  time.sleep(load_seconds)
  return game


class StandInCheckpointBot:
  """Plays a seeded random legal action as the checkpoint it is asked for.

  Attributes:
    state_modes: Both state modes.
    games: The default checkpoint's game, or None (a checkpoint is named
      per request).
  """

  state_modes = (remote_bot_server.SEAT_VIEW, remote_bot_server.FULL_STATE)

  def __init__(self, default: str | None) -> None:
    """Loads the default checkpoint, if any.

    Raises:
      ValueError: If the default checkpoint cannot be read.
    """
    self._lock = threading.Lock()
    self._games: dict[str, str] = {}
    self._default = (default or "").strip() or None
    self.games: tuple[str, ...] | None = None
    if self._default is not None:
      self.games = (self._game(self._default),)

  def _game(self, checkpoint: str) -> str:
    """Returns the checkpoint's game, read once."""
    with self._lock:
      if checkpoint not in self._games:
        self._games[checkpoint] = read(checkpoint)
      return self._games[checkpoint]

  def step(self, mode: str, request: dict[str, Any]) -> dict[str, Any]:
    """Returns a legal action and a notice naming the checkpoint.

    Raises:
      ValueError: If the request names no checkpoint and there is no
        default, or its checkpoint plays another game.
    """
    config = request.get("config") or {}
    if not isinstance(config, Mapping):
      raise ValueError("The request's config is not an object.")
    checkpoint = config.get("checkpoint") or self._default
    if not isinstance(checkpoint, str) or not checkpoint.strip():
      raise ValueError(
          "No checkpoint: give the seat a checkpoint or start the bot server "
          f"with {CHECKPOINT_ENV} set."
      )
    checkpoint = checkpoint.strip()
    game = self._game(checkpoint)
    played = str(request["game"]).split("(", 1)[0]
    if played != game:
      raise ValueError(
          f"The checkpoint {checkpoint} plays {game}, not {played}."
      )
    rng = random.Random(remote_bot_server.seed_of(mode, request))
    action = rng.choice(remote_bot_server.legal_actions(mode, request))
    name = pathlib.Path(checkpoint.rpartition("@")[0]).name
    step = checkpoint.rpartition("@")[2]
    return {"action": action, "notice": f"stand-in {name}@{step}"}


def make_bot(
    environ: Mapping[str, str] | None = None,
) -> StandInCheckpointBot:
  """Returns the bot, its default checkpoint from CHECKPOINT_ENV."""
  return StandInCheckpointBot(
      (os.environ if environ is None else environ).get(CHECKPOINT_ENV)
  )


def main(argv: Sequence[str] | None = None) -> None:
  """Writes a stand-in checkpoint: `write DIR --game GAME --step STEP`."""
  parser = argparse.ArgumentParser(
      prog="python -m open_spiel.python.gui.stand_in_checkpoint_bot",
      description="Write a stand-in checkpoint.",
  )
  parser.add_argument("command", choices=("write",))
  parser.add_argument("directory")
  parser.add_argument("--game", required=True)
  parser.add_argument("--step", type=int, default=1)
  args = parser.parse_args(argv)
  print(write(args.directory, args.step, args.game))


if __name__ == "__main__":
  main()
