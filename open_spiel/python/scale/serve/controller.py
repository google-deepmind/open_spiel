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

r"""The "checkpoint" controller of OpenSpiel's game GUI.

A seat played by an artifact.

The GUI's server (open_spiel.python.gui) plays a seat with a registered
controller. This module adds one, "checkpoint", which plays the seat with a
trained artifact through checkpoint_bot, so the GUI plays the move the
artifact makes in evaluation, from the seat's own view only.

Registration is opt-in: neither the GUI nor this package imports the other,
and the package's __init__ does not import this module. The GUI's plugin
loader imports it and calls register_into(controllers) with
open_spiel.python.gui.controllers, when asked to by

  python -m open_spiel.python.gui \
      --controllers open_spiel.python.scale.serve.controller

or by `--controllers auto` and the entry point OpenSpiel's distribution
declares for it:

  [project.entry-points."open_spiel.gui.controllers"]
  checkpoint = "open_spiel.python.scale.serve.controller"

`--controllers auto` imports this module, and so JAX: without JAX installed
it fails loud, like any plugin that cannot be imported.

register_into registers kind "checkpoint" (reads_full_state=False) with settings

  checkpoint     str, required (REQUIRED, min_length=1): "<dir or URL>@<step>"
                 ("@<step>" left out: the latest step). A scheme registered with
                 register_checkpoint_resolver is resolved first; anything else
                 is a local directory, a gs:// URL or a public Cloud
                 Storage HTTPS URL (read anonymously, ckpt.public).
  deterministic  bool, default True: greedy (True) or sampling (False).
  temperature    float, default 1.0 (minimum 0.0): the sampling temperature;
                 sampling at 0 is refused by build.

build(game, settings, seed) loads the artifact (once per process: the bots
share one PolicyServer), checks it plays `game` with its parameters and warms
up every seat; it raises the GUI's controllers.UnavailableError, with the
reason, for a missing setting, a missing artifact or a game the artifact does
not play. The controller it returns is a CheckpointBot: choose_action(state,
seat) checks the seat is to move and returns a legal action; last_probs
holds that decision's distribution. On a move where the agent's rules and
OpenSpiel's differ it still returns a legal action (checkpoint_bot, "Rules
differences"): it logs a WARNING, counts it in `stats` and describes it in
`last_notice`. The GUI's controller contract (version 8) reads
`last_notice` with the action and shows it to that seat's viewer and to
omniscient viewers; the notice changes nothing of play, and a blank one or
one over 200 characters fails the seat. CheckpointBot's `last_notice` is
therefore the seat's notice (checkpoint_bot.seat_notice): it names only the
seat's own legal actions and the move played, and is cut to
checkpoint_bot.NOTICE_MAX_LENGTH; the full detail goes to the log.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable, Mapping, Optional

from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import server as server_lib

NAME = "checkpoint"
LABEL = "Checkpoint"

_SCHEME = re.compile(r"([a-z][a-z0-9+.-]*)://")
_RESOLVERS: dict[str, Callable[[str], str]] = {}
_LOCK = threading.Lock()
_SERVER: dict[str, server_lib.PolicyServer] = {}


def register_checkpoint_resolver(scheme: str, fn: Callable[[str], str]) -> None:
  """Resolves checkpoint settings that start with "<scheme>://" through `fn`.

  fn(setting) receives the whole setting and returns "<dir or URL>" or
  "<dir or URL>@<step>". A setting whose scheme has no resolver is a local
  directory, a gs:// URL or a public Cloud Storage HTTPS URL
  (ckpt.public), as given.

  Args:
    scheme: the URL scheme.
    fn: the resolver.

  Raises:
    ValueError: `scheme` is not a URL scheme, or already has a resolver.
    TypeError: `fn` is not callable.
  """
  if not re.fullmatch(r"[a-z][a-z0-9+.-]*", scheme):
    raise ValueError(
        f"{scheme!r} is not a URL scheme (lower case, starting with a letter)"
    )
  if not callable(fn):
    raise TypeError(f"the resolver of {scheme!r} is not callable: {fn!r}")
  with _LOCK:
    if scheme in _RESOLVERS:
      raise ValueError(f"the scheme {scheme!r} already has a resolver")
    _RESOLVERS[scheme] = fn


def unregister_checkpoint_resolver(scheme: str) -> None:
  """Removes the resolver of `scheme`.

  Args:
    scheme: the URL scheme.

  Raises:
    ValueError: `scheme` has no resolver.
  """
  with _LOCK:
    if _RESOLVERS.pop(scheme, None) is None:
      raise ValueError(f"the scheme {scheme!r} has no resolver")


def resolve_checkpoint(setting: str) -> tuple[str, Optional[int]]:
  """(path, step) of a checkpoint setting (module doc).

  Args:
    setting: the checkpoint setting.

  Returns:
    (path, step); step None is the latest.

  Raises:
    ValueError: an empty setting, or a resolver that returns no string.
  """
  if not isinstance(setting, str) or not setting.strip():
    raise ValueError(
        'the checkpoint setting is required: "<dir or URL>@<step>"'
    )
  setting = setting.strip()
  m = _SCHEME.match(setting)
  with _LOCK:
    fn = _RESOLVERS.get(m.group(1)) if m else None
  if fn is not None:
    resolved = fn(setting)
    if not isinstance(resolved, str) or not resolved.strip():
      raise ValueError(
          f"the resolver of {m.group(1)!r} returned {resolved!r} for "
          f"{setting!r}"
      )
    setting = resolved.strip()
  return checkpoint_bot_lib.split_checkpoint(setting)


def shared_server() -> server_lib.PolicyServer:
  """The process's PolicyServer for checkpoint controllers.

  Made on first use.

  Returns:
    The PolicyServer.
  """
  with _LOCK:
    if "server" not in _SERVER:
      _SERVER["server"] = server_lib.PolicyServer()
    return _SERVER["server"]


def refusal(game: Any) -> Optional[str]:
  """Why no checkpoint can play `game`, or None when an env plays it.

  Args:
    game: the pyspiel game.

  Returns:
    The refusal, or None.
  """
  name = game.get_type().short_name
  if name not in games.served_games():
    return (
        f"{LABEL} plays only the games it has an env for, and {name} is not "
        "one."
    )
  return None


def build_bot(
    game: Any,
    settings: Mapping[str, Any],
    seed: int,
    *,
    server: Optional[server_lib.PolicyServer] = None,
) -> checkpoint_bot_lib.CheckpointBot:
  """The CheckpointBot of parsed `settings` for `game`, checked and warmed up.

  Args:
    game: the pyspiel game.
    settings: the parsed settings.
    seed: the bot's seed.
    server: the PolicyServer to serve through; None: shared_server().

  Returns:
    The CheckpointBot.

  Raises:
    ValueError: a missing checkpoint setting, or an artifact that does not
      play `game` (check_serves).
    FileNotFoundError: no artifact at the setting's path or step.
  """
  path, step = resolve_checkpoint(settings.get("checkpoint", ""))
  bot = checkpoint_bot_lib.checkpoint_bot(
      path,
      step=step,
      deterministic=bool(settings.get("deterministic", True)),
      temperature=float(settings.get("temperature", 1.0)),
      seed=int(seed),
      server=shared_server() if server is None else server,
  )
  bot.check_serves(game)
  bot.warmup(range(bot.num_players))
  return bot


def register_into(
    controllers: Any, *, server: Optional[server_lib.PolicyServer] = None
) -> None:
  """Registers kind "checkpoint" in the GUI's `controllers` module (module doc).

  Args:
    controllers: the GUI's controllers module (or a module with its register,
      Kind, Setting, REQUIRED and UnavailableError).
    server: the PolicyServer the bots serve through; None: shared_server().

  Raises:
    ValueError: a controller is already registered as "checkpoint".
  """
  setting = controllers.Setting
  settings = (
      setting(
          "checkpoint",
          str,
          controllers.REQUIRED,
          "The checkpoint to play.",
          min_length=1,
      ),
      setting(
          "deterministic",
          bool,
          True,
          "Whether it plays its most likely move rather than sampling.",
      ),
      setting(
          "temperature",
          float,
          1.0,
          "The temperature it samples its moves at.",
          minimum=0.0,
      ),
  )

  def build(game: Any, parsed: Mapping[str, Any], seed: int) -> Any:
    """The bot of a seat, or the GUI's UnavailableError.

    Args:
      game: the pyspiel game.
      parsed: the parsed settings.
      seed: the seat's seed.

    Returns:
      The CheckpointBot.

    Raises:
      controllers.UnavailableError: the bot cannot be built.
    """
    try:
      return build_bot(game, parsed, seed, server=server)
    except (ValueError, FileNotFoundError) as e:
      raise controllers.UnavailableError(
          f"{LABEL} {parsed.get('checkpoint')!r} cannot play "
          f"{game.get_type().short_name}: {e}"
      ) from e

  controllers.register(
      NAME,
      controllers.Kind(
          LABEL, settings, reads_full_state=False, build=build, refusal=refusal
      ),
  )
