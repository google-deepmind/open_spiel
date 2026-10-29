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

r"""An artifact as a Remote bot of OpenSpiel's game GUI (full_state mode).

OpenSpiel's GUI plays a "Remote bot" seat by posting each position to a bot
server (open_spiel/python/gui/REMOTE_BOTS.md, GUI contract 8). Its reference
server loads any bot from a factory, so this module serves a trained
artifact with no change to the GUI:

  OPEN_SPIEL_SCALE_REMOTE_BOT_CHECKPOINT="<dir or URL>@<step>" \
  python -m open_spiel.python.gui.remote_bot_server --port 9001 \
      --bot open_spiel.python.scale.serve.remote_bot:make_bot

and a seat given the controller Remote bot, URL http://localhost:9001 and
state `full_state` plays the artifact.

The protocol (contract 8). The GUI sends the protocol version in the
`Spiel-Protocol: 1` header (PROTOCOL_HEADER, PROTOCOL), never in the body,
so the body is exactly the bot's input; the reference server hands the bot
the body alone. The bot takes only `full_state` requests (POST /spiel/step),
whose body has exactly the keys FULL_STATE_KEYS: `game` (str(game)), `state`
(state.serialize()), `history` (action ids), `seat`, `legal_actions`,
`seed`, `game_encoding` ({"num_distinct_actions"}) and `config` (the seat's
JSON object). A request missing any of them is refused; other keys are
ignored. For each request the bot loads `game`, rebuilds the state with that
game's deserializer (`game.deserialize_state(request["state"])`), checks it
against the request (its history is `history`, the seat to move is `seat`,
its legal actions are `legal_actions`, the game's action count is
`game_encoding`'s) and answers `{"action": a, "notice": n}`: `a` is
checkpoint_bot's move at that state, which replays the state's history into
the artifact's env and lets the policy see only the seat's own view
(checkpoint_bot), and `n` is the reply's notice (below) or None. A request
it cannot answer (a seat view, no artifact, an artifact of another game, a
state that disagrees with the request) is a ValueError, which the server
answers as a refusal (HTTP 422).

The notice. Contract 8 shows a seat's notice (`SeatView.notice`: non-blank
text of at most NOTICE_MAX_LENGTH characters, else the seat fails) to that
seat's own viewer and to omniscient viewers, and has it change nothing of
play. A full-state seat's notice passes through the GUI's state policy
(controller_registry.full_state_notice): every notice is passed on under
ANALYSIS, and a stricter policy (a match) may drop or limit it, so nothing
here depends on the notice arriving. The bot is trusted with the full state
and keeps its notice to what its seat may see: a legal-set notice names
only the seat's own legal actions and the move it plays; a divergence
notice (the env could not replay the hand) says so and names the move
played, and the replay's error, which can name another seat's action or a
chance outcome, goes only to the bot server's log. A notice longer than
NOTICE_MAX_LENGTH is cut to it (the whole notice is logged); a blank one
is sent as None.

The artifact: the request's `config.checkpoint`, else the environment
variable CHECKPOINT_ENV, each "<dir or URL>@<step>" ("@<step>" left out: the
latest step). A public Cloud Storage HTTPS URL
(ckpt.public.ENDPOINT/<bucket>/<path>) is copied into the local artifact
cache by ckpt.public first; any other http(s) URL is refused. `config` may
also give `deterministic`
(default True: greedy) and `temperature` (default 1.0), and `seed` (default
0) seeds a sampling bot's keys. The artifact named by CHECKPOINT_ENV is
loaded, checked and warmed up (every seat, every history length) when the
factory runs, before the server listens; one named only in a request's
config is loaded and warmed up by the first request that names it. Each
distinct artifact and rule is loaded once per process.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping, Optional

from absl import logging

from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.ckpt import public as public_lib
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import controller as controller_lib

FULL_STATE = "full_state"
#: A full_state request's body keys (contract 8, POST /spiel/step).
FULL_STATE_KEYS = (
    "game",
    "state",
    "history",
    "seat",
    "legal_actions",
    "seed",
    "game_encoding",
    "config",
)
#: The header carrying the protocol version, and the version (contract 8).
PROTOCOL_HEADER = "Spiel-Protocol"
PROTOCOL = 1
#: The environment variable naming the artifact a request's config does not.
CHECKPOINT_ENV = plugins.ENV_PREFIX + "REMOTE_BOT_CHECKPOINT"
#: The notice helpers (checkpoint_bot: CheckpointBot.last_notice is already
#: the seat's notice; the step reply applies them again, a no-op).
NOTICE_MAX_LENGTH = checkpoint_bot_lib.NOTICE_MAX_LENGTH
DIVERGED_NOTICE = checkpoint_bot_lib.DIVERGED_NOTICE
clip_notice = checkpoint_bot_lib.clip_notice
seat_notice = checkpoint_bot_lib.seat_notice
_HTTP_PREFIXES = ("https://", "http://")


def local_artifact(path: str, step: Optional[int]) -> tuple[str, int]:
  """A local path and step for an artifact at `path`@`step`.

  A public Cloud Storage HTTPS URL (ckpt.public.ENDPOINT/<bucket>/<path>) is
  copied into the local artifact cache by ckpt.public: the folder's
  manifests to find the latest step, then the step's whole folder. Any other
  path is returned as it is, with its step (the latest when `step` is None).

  Args:
    path: a local directory, an etils.epath URL or a public HTTPS URL.
    step: the step, or None for the latest.

  Returns:
    (path the artifact readers open, step).

  Raises:
    ValueError: an http(s) URL that is not a public Cloud Storage URL.
    FileNotFoundError: no artifact at `path` (or at `path`@`step`).
  """
  if public_lib.is_https(path):
    url = public_lib.norm_https(path)
    if step is None:
      step = artifact_lib.latest_step(public_lib.localize(url, None))
    return public_lib.readable(url, int(step)), int(step)
  if path.startswith(_HTTP_PREFIXES):
    raise ValueError(
        f"{path} is not a public Cloud Storage URL ({public_lib.ENDPOINT}/"
        "<bucket>/<path>)"
    )
  return path, (artifact_lib.latest_step(path) if step is None else step)


class RemoteCheckpointBot:
  """Artifacts served to the GUI's Remote bot seats (module doc).

  Attributes:
    state_modes: ("full_state",), the requests it takes.
    games: the short name of the default artifact's game, or None (any game,
      an artifact named per request).
    default: the default artifact's settings ({"checkpoint": ...}), or None.
  """

  state_modes = (FULL_STATE,)

  def __init__(
      self,
      default_checkpoint: Optional[str] = None,
      *,
      warmup: bool = True,
  ):
    """Loads, checks and warms up the default artifact, if any.

    Args:
      default_checkpoint: "<dir or URL>@<step>" played when a request's
        config names none; None: every request must name one.
      warmup: whether a loaded artifact is warmed up (every seat, every
        history length) before its first move.

    Raises:
      ValueError: the default artifact's game has no pyspiel game string.
      FileNotFoundError: no artifact at the default checkpoint.
    """
    self._lock = threading.Lock()
    self._bots: dict[tuple[Any, ...], checkpoint_bot_lib.CheckpointBot] = {}
    self._served: set[tuple[tuple[Any, ...], str]] = set()
    self._warmup = bool(warmup)
    self.default: Optional[dict[str, Any]] = None
    self.games: Optional[tuple[str, ...]] = None
    if default_checkpoint is not None and default_checkpoint.strip():
      self.default = {"checkpoint": default_checkpoint.strip()}
      bot = self.bot(self.default)
      self.games = (bot.manifest.game,)

  def bot(self, config: Mapping[str, Any]) -> checkpoint_bot_lib.CheckpointBot:
    """The CheckpointBot `config` names, loaded once (module doc).

    Args:
      config: the request's config: `checkpoint`, `deterministic`,
        `temperature`, `seed`.

    Returns:
      The CheckpointBot.

    Raises:
      ValueError: no artifact named, bad settings, or an artifact whose game
        has no pyspiel game string.
      FileNotFoundError: no artifact at the named checkpoint.
    """
    if not isinstance(config, Mapping):
      raise ValueError(f"the request's config is not an object: {config!r}")
    setting = config.get("checkpoint")
    if setting is None and self.default is not None:
      setting = self.default["checkpoint"]
    if not isinstance(setting, str) or not setting.strip():
      raise ValueError(
          "no artifact: give the seat's config a checkpoint "
          '("<dir or URL>@<step>") or start the server with '
          f"{CHECKPOINT_ENV} set"
      )
    deterministic = config.get("deterministic", True)
    if not isinstance(deterministic, bool):
      raise ValueError(f"deterministic must be a bool, got {deterministic!r}")
    temperature = float(config.get("temperature", 1.0))
    seed = config.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
      raise ValueError(f"seed must be an int, got {seed!r}")
    key = (setting.strip(), deterministic, temperature, seed)
    with self._lock:
      if key in self._bots:
        return self._bots[key]
      path, step = controller_lib.resolve_checkpoint(setting)
      path, step = local_artifact(path, step)
      bot = checkpoint_bot_lib.checkpoint_bot(
          path,
          step=step,
          deterministic=deterministic,
          temperature=temperature,
          seed=seed,
          server=controller_lib.shared_server(),
      )
      if self._warmup:
        bot.warmup(range(bot.num_players))
      logging.info("remote bot: serving %r for %s", bot, setting)
      self._bots[key] = bot
      return bot

  def state_of(self, request: Mapping[str, Any]) -> tuple[Any, Any]:
    """The pyspiel game and state of a full_state request, checked.

    Args:
      request: the request's body.

    Returns:
      (game, state).

    Raises:
      ValueError: a missing field (FULL_STATE_KEYS), a state the game cannot
        deserialize, or a state whose history, seat to move, legal actions
        or action count disagree with the request.
    """
    # Lazy: pyspiel is optional (the serve package imports without it).
    import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    for field in FULL_STATE_KEYS:
      if field not in request:
        raise ValueError(f"the full_state request has no {field!r}")
    try:
      game = pyspiel.load_game(str(request["game"]))
      state = game.deserialize_state(str(request["state"]))
    except pyspiel.SpielError as e:
      raise ValueError(
          f"{request['game']} cannot rebuild the request's state: {e}"
      ) from e
    encoding = request["game_encoding"]
    if (
        not isinstance(encoding, Mapping)
        or encoding.get("num_distinct_actions") != game.num_distinct_actions()
    ):
      raise ValueError(
          f"the request's game_encoding {encoding!r} is not {request['game']}'s"
          f" ({game.num_distinct_actions()} distinct actions)"
      )
    history = [int(a) for a in state.history()]
    if history != [int(a) for a in request["history"]]:
      raise ValueError(
          f"the deserialized state's history {history} is not the request's "
          f"history {list(request['history'])}"
      )
    seat = request["seat"]
    if isinstance(seat, bool) or not isinstance(seat, int):
      raise ValueError(f"seat must be an int, got {seat!r}")
    if int(state.current_player()) != seat:
      raise ValueError(
          f"seat {seat} is not to move after the history; player "
          f"{state.current_player()} is"
      )
    legal = [int(a) for a in state.legal_actions(seat)]
    if legal != [int(a) for a in request["legal_actions"]]:
      raise ValueError(
          f"the state's legal actions {legal} are not the request's "
          f"{list(request['legal_actions'])}"
      )
    return game, state

  def step(self, mode: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """The reply to one request: {"action", "notice"} (module doc).

    Args:
      mode: "full_state" (a "seat_view" request is refused).
      request: the request's body.

    Returns:
      {"action": int, "notice": str or None}.

    Raises:
      ValueError: the request cannot be answered (module doc).
    """
    if mode != FULL_STATE:
      raise ValueError(
          f"the checkpoint bot takes {FULL_STATE} requests, not {mode}"
      )
    game, state = self.state_of(request)
    bot = self.bot(request["config"])
    seat = int(request["seat"])
    with self._lock:
      key = (id(bot), str(game))
      if key not in self._served:
        bot.check_serves(game)
        self._served.add(key)
      # One move at a time per process: the bot's last_* are its own, and
      # the server answers requests on threads.
      reply = bot.reply(state, seat)
    if reply.notice is not None:
      logging.warning(
          "remote bot: %s seat %d move %d: %s",
          game,
          seat,
          len(state.history()),
          reply.notice,
      )
    return {
        "action": int(reply.action),
        "notice": seat_notice(reply, state, seat),
    }


def make_bot(
    environ: Optional[Mapping[str, str]] = None,
) -> RemoteCheckpointBot:
  """The bot remote_bot_server's `--bot` names this factory for (module doc).

  Args:
    environ: the environment (default os.environ); CHECKPOINT_ENV names the
      default artifact.

  Returns:
    The RemoteCheckpointBot.
  """
  if environ is None:
    environ = os.environ
  return RemoteCheckpointBot(environ.get(CHECKPOINT_ENV))
