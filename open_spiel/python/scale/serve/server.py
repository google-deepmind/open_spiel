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

A request decides by the artifact's default rule (its manifest's
default_rule) unless it names a `rule` (core.types.DecisionRule).

The env's rules and pyspiel's can disagree on a rare node (a rules
difference between the env and the installed OpenSpiel). By default a
request at such a node is refused (wire.ReplayDivergenceError, or a
RuntimeError). With repair=True, what a GUI seat asks for
(checkpoint_bot), it is answered and never refused for that reason: a
legal-set difference plays the policy renormalised over the actions both
allow (restrict_to_legal); a divergence earlier in the hand, which the env
cannot replay past, plays pyspiel's first legal action. Either is logged as
a WARNING, counted (PolicyServer.stats) and named in the Reply's notice.

Every request is stateless: the history is replayed each time. Loaded
artifacts and their HostAgents (jitted functions) are cached per checkpoint,
seat and game parameters (the seats of one artifact, game parameters and rule
share one env and one set of jitted functions); the cache holds `cache_size`
artifacts and evicts the least recently used one.

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
from typing import Any, Callable, Mapping, Optional, Sequence

from absl import logging
import jax
import numpy as np

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
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
    decider: "network", "search", "scripted" or "forced"; "fallback" for a
      repaired move made without the policy (module doc).
    artifact: the artifact's label, "<run_id>@<step>".
    notice: None, or what a repaired move (module doc) says about the
      divergence: the move, the actions only one side allows, the move
      played.
  """

  action: int
  probs: Mapping[int, float]
  value: Optional[float]
  decider: str
  artifact: str
  notice: Optional[str] = None

  def to_json(self) -> dict[str, Any]:
    """The reply as JSON types.

    Returns:
      The JSON dict.
    """
    out = {
        "action": int(self.action),
        "probs": {str(a): float(p) for a, p in sorted(self.probs.items())},
        "value": None if self.value is None else float(self.value),
        "decider": self.decider,
        "artifact": self.artifact,
    }
    if self.notice is not None:
      out["notice"] = self.notice
    return out


def restrict_to_legal(
    decision: host_agent.Decision, legal: Sequence[int]
) -> tuple[host_agent.Decision, str]:
  """The decision restricted to pyspiel's legal actions `legal`.

  The policy's distribution renormalised over the actions both the env and
  pyspiel allow, and its most probable action (the first of maximal
  probability); when they share none, or the policy gives them no mass, the
  pyspiel-legal action of highest logit, else pyspiel's first legal action.

  Args:
    decision: the env's decision.
    legal: pyspiel's legal actions (ids below the env's num_actions).

  Returns:
    (the restricted Decision, how its action was chosen).

  Raises:
    ValueError: `legal` is empty or names an action the env has no id for.
  """
  legal = sorted(int(a) for a in legal)
  probs = np.asarray(decision.probs, np.float64)
  if not legal or legal[0] < 0 or legal[-1] >= probs.shape[0]:
    raise ValueError(
        f"pyspiel legal actions {legal} are not ids of the env's "
        f"{probs.shape[0]} actions"
    )
  both = [a for a in legal if decision.legal[a]]
  mass = float(sum(probs[a] for a in both))
  out = np.zeros_like(probs)
  if both and mass > 0.0:
    for a in both:
      out[a] = probs[a] / mass
    action = max(both, key=lambda a: (out[a], -a))
    how = "the policy over the actions both allow"
  else:
    logits = decision.logits
    scored = (
        [] if logits is None else [a for a in legal if np.isfinite(logits[a])]
    )
    if scored:
      action = max(scored, key=lambda a: (float(logits[a]), -a))
      how = "the pyspiel-legal action of highest logit"
    else:
      action = legal[0]
      how = "pyspiel's first legal action"
    out[action] = 1.0
  mask = np.zeros(probs.shape[0], np.bool_)
  mask[legal] = True
  return (
      dataclasses.replace(decision, action=int(action), probs=out, legal=mask),
      how,
  )


def _labels(state: Any, seat: int, actions: Sequence[int]) -> str:
  """The game's text for `actions` of `seat`, comma separated, or "none".

  Args:
    state: the pyspiel state.
    seat: the seat.
    actions: action ids.

  Returns:
    The text.
  """
  return ", ".join(state.action_to_string(seat, int(a)) for a in actions) or (
      "none"
  )


class PolicyServer:
  """Serves any artifact over pyspiel states (thread-safe).

  Attributes:
    allow_cross_variant: whether game parameters an artifact's training env
      does not cover are served.
    cache_size: artifacts kept loaded.
    stats: counts of repaired decisions (module doc): "legal_mismatch"
      (the env's and pyspiel's legal sets differed at the move) and
      "diverged" (the env could not replay the hand's history).
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
    self.stats = {"legal_mismatch": 0, "diverged": 0}
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
      rule: Optional[types.DecisionRule] = None,
  ) -> host_agent.HostAgent:
    """The cached HostAgent of an artifact, a seat, game parameters and a rule.

    Args:
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.
      seat: the seat.
      game: the pyspiel game's short name.
      params: pyspiel game parameters.
      rule: the decision rule (None: the manifest's default_rule).

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
        rule,
    )
    with self._lock:
      agent = self._agents.get(key)
      if agent is None:
        if art.manifest.game != game:
          raise ValueError(
              f"the checkpoint plays {art.manifest.game!r}; the request's game "
              f"is {game!r}"
          )
        # Another seat's agent of the same artifact, game parameters and rule:
        # the new one serves its env and reuses its jitted calls (no second
        # compile of the policy or the replay).
        sibling = next(
            (
                a
                for k, a in self._agents.items()
                if k[:2] == key[:2] and k[3:] == key[3:]
            ),
            None,
        )
        agent = host_agent.HostAgent(
            art,
            seat,
            env=games.env_for(game, params) if sibling is None else sibling.env,
            rule=rule,
            allow_cross_variant=self.allow_cross_variant,
            share=sibling,
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
      rule: Optional[types.DecisionRule] = None,
      repair: bool = False,
  ) -> tuple[host_agent.HostAgent, host_agent.Decision]:
    """The HostAgent and its full Decision for `state` (what `act` summarises).

    `seed` seeds the seat key (module doc); None draws a fresh one, so a
    sampling rule does not return the same move for every request. `rule`
    overrides the manifest's default_rule.

    Args:
      state: the pyspiel state.
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.
      seed: seeds the seat key (module doc); None draws a fresh one, so a
        sampling rule does not return the same move for every request.
      rule: overrides the manifest's default_rule (None: the default).
      repair: answer a node where the env and pyspiel disagree instead of
        refusing it (module doc); the Decision's notice says so.

    Returns:
      (HostAgent, Decision).

    Raises:
      ValueError: the state is terminal or a chance node, the replay does not
        reach pyspiel's node (wire.ReplayDivergenceError,
        replay.IllegalHistoryActionError; not with repair), or the artifact
        cannot play the state's game.
      RuntimeError: the decision's legal set disagrees with pyspiel's (not
        with repair).
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
    agent = self.agent(checkpoint_dir, step, seat, name, params, rule)
    hands = wire.pyspiel_history_to_raw(name, state.history(), params)
    legal = sorted(state.legal_actions())
    try:
      decision = agent.act(
          hands,
          seat_key=seat_key(seed, seat),
          node_check=lambda s: wire.check_replayed_node(
              agent.env, s, state, allow_legal_mismatch=repair
          ),
      )
    except (
        wire.ReplayDivergenceError,
        replay.IllegalHistoryActionError,
    ) as e:
      if not repair:
        raise
      return agent, self._fallback(agent, state, seat, legal, e)
    dist = decision.distribution()
    if set(dist) != set(legal) or decision.action not in legal:
      if not repair:
        raise RuntimeError(
            f"decision legal set {sorted(dist)} / action {decision.action} "
            f"disagree with pyspiel's {legal}"
        )
      env_legal = sorted(dist)
      decision, how = restrict_to_legal(decision, legal)
      notice = (
          "engine/agent rules differ on this move (move "
          f"{len(state.history())}; only the engine allows "
          f"{_labels(state, seat, sorted(set(legal) - set(env_legal)))}; only "
          "the agent allows "
          f"{_labels(state, seat, sorted(set(env_legal) - set(legal)))}); "
          "playing the best engine-legal move, "
          f"{_labels(state, seat, [decision.action])}"
      )
      self._count("legal_mismatch")
      logging.warning(
          "%s@%d seat %d: legal-set mismatch: env %s, pyspiel %s; played %d "
          "(%s). History %s, state:\n%s",
          checkpoint_dir,
          step,
          seat,
          env_legal,
          legal,
          decision.action,
          how,
          state.history(),
          state,
      )
      decision = dataclasses.replace(decision, notice=notice)
    return agent, decision

  def _count(self, key: str) -> None:
    """Counts one repaired decision under `key` (stats).

    Args:
      key: the stats key.
    """
    with self._lock:
      self.stats[key] += 1

  def _fallback(
      self,
      agent: Any,
      state: Any,
      seat: int,
      legal: Sequence[int],
      error: Exception,
  ) -> host_agent.Decision:
    """The first pyspiel-legal action, when the env cannot replay the hand.

    Args:
      agent: the HostAgent.
      state: the pyspiel state.
      seat: the seat to move.
      legal: pyspiel's legal actions.
      error: why the replay failed.

    Returns:
      The fallback Decision, with its notice.

    Raises:
      ValueError: pyspiel has no legal action the env has an id for.
    """
    n = agent.env.num_actions
    if not legal or legal[-1] >= n:
      raise ValueError(
          f"pyspiel legal actions {list(legal)} are not ids of the env's {n} "
          "actions"
      ) from error
    action = int(legal[0])
    probs = np.zeros(n, np.float64)
    probs[action] = 1.0
    mask = np.zeros(n, np.bool_)
    mask[list(legal)] = True
    self._count("diverged")
    logging.warning(
        "seat %d: the env and pyspiel diverged earlier in the hand (%s); "
        "played pyspiel's first legal action %d of %s. History %s, "
        "state:\n%s",
        seat,
        error,
        action,
        list(legal),
        state.history(),
        state,
    )
    return host_agent.Decision(
        action=action,
        probs=probs,
        legal=mask,
        decider="fallback",
        logits=None,
        value=None,
        identity=agent.identity,
        notice=(
            "engine/agent rules differed earlier in this hand "
            f"({error}); playing the first engine-legal move, "
            f"{_labels(state, seat, [action])}"
        ),
    )

  def act(
      self,
      state: Any,
      *,
      checkpoint_dir: str,
      step: int,
      seed: Optional[int] = None,
      rule: Optional[types.DecisionRule] = None,
      repair: bool = False,
  ) -> Reply:
    """The move of checkpoint_dir@step at `state` (module doc).

    Args:
      state: the pyspiel state.
      checkpoint_dir: the run's checkpoint directory.
      step: the checkpoint step.
      seed: the request's seed (decide).
      rule: the decision rule (decide).
      repair: answer a node where the env and pyspiel disagree (decide).

    Returns:
      The Reply.
    """
    agent, d = self.decide(
        state,
        checkpoint_dir=checkpoint_dir,
        step=step,
        seed=seed,
        rule=rule,
        repair=repair,
    )
    value = None if d.value is None else float(d.value[agent.seat])
    return Reply(
        action=int(d.action),
        probs=d.distribution(),
        value=value,
        decider=d.decider,
        artifact=d.identity["label"],
        notice=d.notice,
    )
