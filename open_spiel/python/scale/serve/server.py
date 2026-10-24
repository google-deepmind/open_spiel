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

"""PolicyServer: answer a pyspiel state with the move an artifact makes there.

A game server holds a pyspiel state and wants one move from a trained
artifact. PolicyServer loads the artifact, rebuilds the env that plays the
state's game parameters, replays the state's history through a HostAgent and
returns the decision, so a served move equals the move the artifact makes in
evaluation.

  server = PolicyServer()
  reply = server.act(state, checkpoint_dir="/ckpts/run", step=1000, seed=0)
  reply.action, reply.probs, reply.to_json()

Every request is stateless: the history is replayed each time. Loaded
artifacts and their HostAgents (jitted functions) are cached per checkpoint,
seat and game parameters; the cache holds `cache_size` artifacts and evicts
the least recently used one.

The seat key of a request is fold_in(key(seed), seat). Only a sampling
decision rule or a stochastic policy reads it. `seed` defaults to None, which
draws a fresh seed from the operating system per request, so repeated requests
for one state under a sampling rule sample independently; pass a seed to make
a request reproducible (the same seed and state give the same move).
"""

from __future__ import annotations

import collections
import dataclasses
import secrets
import threading
from typing import Any, Callable, Mapping, Optional

from absl import logging
import jax

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import wire


def seat_key(seed: Optional[int], seat: int):
  """The typed jax key of `seat` for a request's `seed` (None: a fresh seed).

  Args:
    seed: the request's seed, or None.
    seat: the seat.

  Returns:
    The typed PRNG key.
  """
  if seed is None:
    seed = secrets.randbits(63)
  return jax.random.fold_in(jax.random.key(int(seed)), int(seat))


@dataclasses.dataclass(frozen=True)
class Reply:
  """The answer to one request.

  Attributes:
    action: the chosen action (a pyspiel action id).
    probs: {legal action: probability} of the decision rule's distribution.
    value: the policy's value estimate for the acting seat, or None (a forced
      move does not run the policy).
    decider: "network", "search", "scripted" or "forced".
    artifact: the artifact's label, "<run_id>@<step>".
  """

  action: int
  probs: Mapping[int, float]
  value: Optional[float]
  decider: str
  artifact: str

  def to_json(self) -> dict[str, Any]:
    """The reply as JSON types.

    Returns:
      The JSON dict.
    """
    return {
        "action": int(self.action),
        "probs": {str(a): float(p) for a, p in sorted(self.probs.items())},
        "value": None if self.value is None else float(self.value),
        "decider": self.decider,
        "artifact": self.artifact,
    }


class PolicyServer:
  """Serves any artifact over pyspiel states (thread-safe).

  Attributes:
    allow_cross_variant: whether game parameters an artifact's training env
      does not cover are served.
    cache_size: artifacts kept loaded.
  """

  def __init__(
      self,
      *,
      loader: Callable[[str, int], types.Artifact] = artifact_lib.load_artifact,
      allow_cross_variant: bool = False,
      cache_size: int = 8,
  ):
    """Builds the server with an empty cache.

    Args:
      loader: loads an artifact from (checkpoint_dir, step).
      allow_cross_variant: serve game parameters an artifact's training env
        does not cover (for intentional cross-variant evaluation).
      cache_size: artifacts kept loaded (least recently used evicted first).

    Raises:
      ValueError: cache_size is below 1.
    """
    # An entry point (package doc): it loads and serves artifacts.
    x64.enable_x64()
    if int(cache_size) < 1:
      raise ValueError(f"cache_size must be >= 1, got {cache_size}")
    self.allow_cross_variant = bool(allow_cross_variant)
    self.cache_size = int(cache_size)
    self._loader = loader
    self._lock = threading.Lock()
    self._artifacts: collections.OrderedDict[
        tuple[str, int], types.Artifact
    ] = collections.OrderedDict()
    self._agents: dict[tuple[Any, ...], Any] = {}

  def num_loaded_artifacts(self) -> int:
    """The number of cached artifacts.

    Returns:
      The count.
    """
    return len(self._artifacts)

  def artifact(self, checkpoint_dir: str, step: int) -> types.Artifact:
    """The artifact at checkpoint_dir@step, loaded once and cached.

    Args:
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.

    Returns:
      The Artifact.
    """
    key = (checkpoint_dir.rstrip("/"), int(step))
    with self._lock:
      art = self._artifacts.get(key)
      if art is not None:
        self._artifacts.move_to_end(key)
        return art
    logging.info("PolicyServer: loading %s step %d", *key)
    art = self._loader(*key)
    with self._lock:
      self._artifacts[key] = art
      self._artifacts.move_to_end(key)
      while len(self._artifacts) > self.cache_size:
        old, _ = self._artifacts.popitem(last=False)
        self._agents = {k: v for k, v in self._agents.items() if k[:2] != old}
    return art

  def agent(
      self,
      checkpoint_dir: str,
      step: int,
      seat: int,
      game: str,
      params: Mapping[str, Any],
  ) -> host_agent.HostAgent:
    """The cached HostAgent of an artifact, a seat and pyspiel game parameters.

    Args:
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.
      seat: the seat.
      game: the pyspiel game's short name.
      params: pyspiel game parameters.

    Returns:
      The HostAgent.

    Raises:
      ValueError: the artifact plays another game, or the parameters are not
        a variant its env implements.
    """
    checkpoint_dir = checkpoint_dir.rstrip("/")
    art = self.artifact(checkpoint_dir, step)
    key = (
        checkpoint_dir,
        int(step),
        int(seat),
        game,
        tuple(sorted((k, repr(v)) for k, v in params.items())),
    )
    with self._lock:
      agent = self._agents.get(key)
      if agent is None:
        if art.manifest.game != game:
          raise ValueError(
              f"the checkpoint plays {art.manifest.game!r}; the request's game "
              "is "
              f"{game!r}"
          )
        agent = host_agent.HostAgent(
            art,
            seat,
            env=games.env_for(game, params),
            allow_cross_variant=self.allow_cross_variant,
        )
        self._agents[key] = agent
      return agent

  def decide(
      self,
      state: Any,
      *,
      checkpoint_dir: str,
      step: int,
      seed: Optional[int] = None,
  ) -> tuple[host_agent.HostAgent, host_agent.Decision]:
    """The HostAgent and its full Decision for `state` (what `act` summarises).

    `seed` seeds the seat key (module doc); None draws a fresh one, so a
    sampling rule does not return the same move for every request.

    Args:
      state: the pyspiel state.
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.
      seed: seeds the seat key (module doc); None draws a fresh one, so a
        sampling rule does not return the same move for every request.

    Returns:
      (HostAgent, Decision).

    Raises:
      ValueError: the state is terminal or a chance node, the replay does not
        reach pyspiel's node, or the artifact cannot play the state's game.
      RuntimeError: the decision's legal set disagrees with pyspiel's.
    """
    if state.is_terminal():
      raise ValueError("the state is terminal; there is no move to make")
    if state.is_chance_node():
      raise ValueError(
          "the state is a chance node; there is no player decision"
      )
    game = state.get_game()
    name = game.get_type().short_name
    params = wire.game_params(game)
    seat = int(state.current_player())
    agent = self.agent(checkpoint_dir, step, seat, name, params)
    hands = wire.pyspiel_history_to_raw(name, state.history(), params)
    decision = agent.act(
        hands,
        seat_key=seat_key(seed, seat),
        node_check=lambda s: wire.check_replayed_node(agent.env, s, state),
    )
    legal = set(state.legal_actions())
    dist = decision.distribution()
    if set(dist) != legal or decision.action not in legal:
      raise RuntimeError(
          f"decision legal set {sorted(dist)} / action {decision.action} "
          f"disagree with pyspiel's {sorted(legal)}"
      )
    return agent, decision

  def act(
      self,
      state: Any,
      *,
      checkpoint_dir: str,
      step: int,
      seed: Optional[int] = None,
  ) -> Reply:
    """The move of checkpoint_dir@step at `state` (module doc).

    Args:
      state: the pyspiel state.
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.
      seed: the request's seed (decide).

    Returns:
      The Reply.
    """
    agent, d = self.decide(
        state, checkpoint_dir=checkpoint_dir, step=step, seed=seed
    )
    value = None if d.value is None else float(d.value[agent.seat])
    return Reply(
        action=int(d.action),
        probs=d.distribution(),
        value=value,
        decider=d.decider,
        artifact=d.identity["label"],
    )
