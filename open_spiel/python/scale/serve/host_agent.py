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

"""HostAgent: one decision of an artifact for one seat, as in-graph play does.

A served move must be the move the policy makes in evaluation, or ratings
measured in the arena do not transfer to the served engine. HostAgent keeps no
state between requests; each `act` replays the match history and runs the
policy exactly as the Arena runs a row:

  agent = host_agent.HostAgent(artifact, seat)
  decision = agent.act(hands, seat_key=jax.random.key(0))
  decision.action, decision.distribution(), decision.decider

Per request:
  1. replay the history into the SeatSteps `seat` saw
     (`replay.replay_seat_steps`);
  2. a forced move (one legal action) skips the policy and decides over zero
     logits; its decider is "forced";
  3. otherwise init_carry, one observe_seq over the history (a stateless
     policy observes only the last ply), evaluate on the last ply, and the
     one `decide` in f64.

Compiles. Every jitted call is compiled once per input shape. A stateless
policy is served on the last ply alone: the replay builds that ply's SeatStep
only (serve.replay.replay_last_seat_step), and the policy path has one shape
whatever the history's length. A stateful policy's replay pads the history to
a power-of-two bucket (serve.replay, "Length buckets"). warmup(max_plies)
compiles every shape a match of up to max_plies plies uses, after which no
decision compiles; a second warmup up to the same length does nothing. A
stateful JAX policy observes the exact history, [1, T], and compiles once
per history length (warmup does not compile those). Agents of the seats of
one artifact can share their jitted calls (`share`), so the second seat
compiles nothing.

Keys reproduce the Arena: given the Arena's seat key and the row ply of the
match's first ply (`first_ply`), the init_carry key is plies.init_key(seat_key)
and the evaluate and decide keys are plies.ply_keys(seat_key, first_ply + T -
1). For the same history on XLA:CPU, HostAgent therefore returns the in-graph
action, and for a tabular, uniform or host-computed policy the distribution
bit for bit. A policy whose logits are float arithmetic in JAX (a network,
feed-forward or recurrent, or a search over one) agrees with the Arena's
logits within 1e-6 absolute plus 1e-6 relative, and its distribution
correspondingly (the action is the same), because XLA fuses and rounds the
rollout scan and the replay differently (core.types.Policy). Bit-identity is
a best-effort property, tested on XLA:CPU only for the batch shapes in the
tests: PAD_ROWS is a heuristic against one CPU kernel choice, not a guarantee
for every Arena chunk size. Other shapes, and GPU or TPU, can differ from the
Arena's reply by rounding.

NATIVE_JAX and CALLBACK_BATCHED policies run jitted. HOST_DRIVEN policies get
numpy SeatSteps and raw uint32[1, 2] keys, as HostArena gives them, and their
logits go through the same decide.

The decider of a decision is the policy spec's `decider` or the policy's
`decider` attribute ("network" when neither is set). A routing policy that
implements `decider_for(info)` names the decider of each request.
"""

import dataclasses
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.interop import host
from open_spiel.python.scale.policies import operators
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import replay

tree_map = jax.tree_util.tree_map

# The JAX path evaluates the request as this many replicas of one row. On
# XLA:CPU a dense layer's f32 result for a row of a 1-3 row batch can differ
# by ULPs from the same row in a larger batch (another dot kernel), so a lone
# row would not reproduce the Arena's batched logits bit for bit. This is an
# observed heuristic: 8 rows select the same kernel as the Arena's batches in
# the tested configurations, and XLA may choose by other shapes elsewhere.
PAD_ROWS = 8


decider_of = types.decider_of  # who chooses a policy's actions (never "forced")


def decider_for(policy: types.Policy, info: Any) -> str:
  """The decider of the request whose last-ply InfoState is `info` [1, ...].

  A policy with a `decider_for(info)` method answers itself (RoutedPolicy:
  its override's decider on the rows its route picks). A host-driven
  operator that routes internally (`policies.operators.routed_view`)
  reports its scripted part's decider on the rows that part plays.

  Args:
    policy: the artifact's policy.
    info: the last-ply InfoState [1, ...].

  Returns:
    The decider name.

  Raises:
    ValueError: the policy reports an unknown decider.
  """
  f = getattr(policy, "decider_for", None)
  if f is None:
    view = operators.routed_view(policy)
    if view is not None:
      use = np.asarray(view.route(info), np.bool_)
      assert use.shape == (1,), use.shape
      if bool(use[0]):
        return decider_for(view.override, info)
    d = decider_of(policy)
  else:
    d = f(info)
  if d not in types.POLICY_DECIDERS:
    raise ValueError(
        f"{policy.spec.arch_name}: decider {d!r} not in {types.POLICY_DECIDERS}"
    )
  return d


def label_of(artifact: types.Artifact) -> str:
  """The artifact's label: "<run_id>@<step>" (else the directory name).

  "<dir name>@<step>" when the manifest records no run id.

  Args:
    artifact: the Artifact.

  Returns:
    The label.
  """
  m = artifact.manifest
  run = (
      m.run_id
      if m.run_id is not None
      else artifact.path.rstrip("/").rsplit("/", 1)[-1]
  )
  return f"{run}@{m.step}"


def identity_of(
    artifact: types.Artifact, rule: types.DecisionRule
) -> dict[str, Any]:
  """The artifact identity echoed with every decision (JSON-serialisable).

  Args:
    artifact: the Artifact.
    rule: the DecisionRule served.

  Returns:
    The identity dict.
  """
  m = artifact.manifest
  return dict(
      label=label_of(artifact),
      run_id=m.run_id,
      step=m.step,
      path=artifact.path,
      game=m.game,
      rules_sha=m.rules_sha,
      env_spec=None if m.env_spec is None else dict(m.env_spec),
      obs_format=m.obs_format,
      arch_name=m.arch_name,
      framework=m.framework,
      capability=m.capability.value,
      params_sha256=m.params_sha256,
      git_commit=m.git_commit,
      schema_version=m.schema_version,
      parents=list(m.parents),
      rule=dict(kind=rule.kind, temperature=float(rule.temperature)),
  )


@dataclasses.dataclass(frozen=True)
class Decision:
  """One decision of a HostAgent.

  Attributes:
    action: the chosen action.
    probs: f64[A], decide's distribution (0 on illegal actions).
    legal: bool[A].
    decider: "network", "search", "scripted" or "forced"; "fallback" when a
      serving shell played on through an env/pyspiel divergence without the
      policy (server.PolicyServer.decide, repair).
    logits: f32[A]; None on a forced or fallback move.
    value: f32[P]; None on a forced or fallback move.
    identity: the artifact identity (identity_of).
    notice: None, or what a serving shell that repaired the decision says
      about it (server.PolicyServer.decide, repair).
  """

  action: int
  probs: np.ndarray  # f64[A], decide's distribution (0 on illegal actions)
  legal: np.ndarray  # bool[A]
  decider: str
  logits: Optional[np.ndarray]  # f32[A]; None on a forced move
  value: Optional[np.ndarray]  # f32[P]; None on a forced move
  identity: Mapping[str, Any]
  notice: Optional[str] = None

  @property
  def searched(self) -> bool:
    """Whether a search decided."""
    return self.decider == "search"

  def distribution(self) -> dict[int, float]:
    """{legal action: probability}.

    Returns:
      The distribution over the legal actions.
    """
    return {int(a): float(self.probs[a]) for a in np.flatnonzero(self.legal)}


class HostAgent:
  """Serves `artifact` for `seat`. Stateless: every `act` replays the match.

  Attributes:
    artifact: the artifact served.
    policy: its policy.
    params: its params.
    env: the env served.
    seat: the seat it plays.
    rule: the DecisionRule served.
    decider: who chooses the policy's actions (types.decider_of).
    identity: the artifact identity echoed with every decision (identity_of).
    host_driven: whether the policy is HOST_DRIVEN (no JAX path).
    pad_rows: rows the JAX path evaluates the request in.
    stateful: whether the policy carries state across plies (its spec's
      stateful).
    warmed_plies: the longest match warmup has compiled for (0: none).
  """

  def __init__(
      self,
      artifact: types.Artifact,
      seat: int,
      env: Optional[types.Env] = None,
      rule: Optional[types.DecisionRule] = None,
      pad_rows: int = PAD_ROWS,
      allow_cross_variant: bool = False,
      share: Optional["HostAgent"] = None,
  ):
    """Builds the agent; jits the policy path once.

    Args:
      artifact: the artifact served.
      seat: the seat it plays.
      env: the env served (default: the artifact's training env, rebuilt from
        its env_spec; an artifact without one: the game's default pyspiel
        parameters).
      rule: the decision rule (default: the manifest's default_rule).
      pad_rows: rows the JAX path evaluates the request in (PAD_ROWS).
      allow_cross_variant: serve an env the artifact's env_spec does not
        cover (envs.registry.check_serves), for intentional cross-variant
        evaluation.
      share: an agent of another seat of the same artifact policy, env, rule
        and pad_rows whose jitted calls this one reuses (the seat is an
        argument of every one), so a second seat compiles nothing; None jits
        its own.

    Raises:
      ValueError: the artifact's game, actions or players do not match the
        env, the env_spec does not cover it, the env refuses a stateful
        policy (core.types.check_policy_admitted), or the seat is out of
        range.
    """
    x64.assert_x64("HostAgent")
    assert isinstance(artifact, types.Artifact), type(artifact)
    self.artifact = artifact
    self.policy = artifact.policy
    self.params = artifact.params
    spec = self.policy.spec
    trained = artifact.manifest.env_spec
    if env is None:
      env = (
          games.env_for(spec.game, {})
          if trained is None
          else env_registry.training_env(artifact.manifest)
      )
    self.env = env
    if spec.game != self.env.name or artifact.manifest.game != self.env.name:
      raise ValueError(
          f"artifact game {artifact.manifest.game!r} (policy {spec.game!r}) != "
          "env "
          f"game {self.env.name!r}"
      )
    env_registry.check_serves(
        trained, self.env, allow_cross_variant=allow_cross_variant
    )
    assert spec.num_actions == self.env.num_actions, (
        spec.num_actions,
        self.env.num_actions,
    )
    assert spec.num_players == self.env.num_players, (
        spec.num_players,
        self.env.num_players,
    )
    types.check_policy_admitted(self.env, spec)
    if not 0 <= int(seat) < self.env.num_players:
      raise ValueError(f"seat {seat} is not in [0, {self.env.num_players})")
    self.seat = int(seat)
    self.rule = artifact.manifest.default_rule if rule is None else rule
    assert isinstance(self.rule, types.DecisionRule), self.rule
    self.decider = decider_of(self.policy)
    self.identity = identity_of(artifact, self.rule)
    self.host_driven = spec.capability == types.Capability.HOST_DRIVEN
    rule_ = self.rule
    self._decide = jax.jit(lambda lg, m, k: decide_lib.decide(lg, m, rule_, k))
    self._keys = jax.jit(
        lambda sk, ply: (plies.init_key(sk),) + plies.ply_keys(sk, ply)
    )
    if int(pad_rows) < 1:
      raise ValueError(f"pad_rows must be >= 1, got {pad_rows}")
    self.pad_rows = int(pad_rows)
    self.stateful = bool(spec.stateful)
    # The longest match warmup has compiled for (0: none).
    self.warmed_plies = 0
    if share is not None:
      if (
          share.policy is not self.policy
          or share.env is not self.env
          or share.rule != self.rule
          or share.pad_rows != self.pad_rows
      ):
        raise ValueError(
            "a shared agent must serve the same policy, env, rule and pad_rows"
        )
      # Its jitted calls read only the policy, the rule and pad_rows.
      self._decide, self._keys = share._decide, share._keys  # pylint: disable=protected-access
      if not self.host_driven:
        self._serve = share._serve  # pylint: disable=protected-access
        self._serve_last = share._serve_last  # pylint: disable=protected-access
    elif not self.host_driven:
      self._serve = jax.jit(self._serve_rows)
      self._serve_last = jax.jit(self._serve_last_rows)

  def act(
      self,
      hands: Sequence[Sequence[int]],
      *,
      seat_key: Any,
      first_ply: int = 0,
      node_check: Optional[Callable[[Any], None]] = None,
  ) -> Decision:
    """The decision of `self.seat` at the end of `hands` (one match).

    Args:
      hands: the match's raw hands, each a list of raw actions.
      seat_key: a typed jax PRNG key (the Arena's policy_key[row, seat]; a
        serving shell derives it from the request's seed).
      first_ply: the row ply of the match's first ply (0 for a match served from
        its start; the Arena's row_ply when reproducing a rollout).
      node_check: called with the replayed env state at the last ply before any
        policy work (the engine shell checks it against pyspiel's node).

    Returns:
      The Decision.

    Raises:
      ValueError: first_ply is negative, the history is not at a decision, or
        another seat is to move.
    """
    if int(first_ply) < 0:
      raise ValueError(f"first_ply must be >= 0, got {first_ply}")
    # A stateless policy reads the last ply only (module doc, "Compiles").
    rep = (
        replay.replay_seat_steps_padded(self.env, hands, self.seat)
        if self.stateful
        else replay.replay_last_seat_step(self.env, hands, self.seat)
    )
    if node_check is not None:
      node_check(rep.state)
    last_np = rep.last  # numpy [1]
    to_move = int(last_np.to_move[0])
    if to_move < 0:
      raise ValueError(
          "the history ends at a terminal state; there is no decision"
      )
    if to_move != self.seat:
      raise ValueError(f"seat {to_move} is to move, not seat {self.seat}")
    legal = np.asarray(last_np.legal_mask[0])
    last = tree_map(jnp.asarray, last_np)
    row_ply = jnp.int64(np.int64(first_ply) + np.int64(rep.num_plies - 1))
    init_key, eval_key, decide_key = self._keys(seat_key, row_ply)
    if int(legal.sum()) == 1:
      zeros = jnp.zeros((1, self.env.num_actions), jnp.float32)
      action, probs = self._decide(zeros, last.legal_mask, decide_key)
      return Decision(
          action=int(np.asarray(action)[0]),
          probs=np.asarray(probs)[0],
          legal=legal,
          decider="forced",
          logits=None,
          value=None,
          identity=self.identity,
      )
    out = self._run_policy(rep, last, init_key, eval_key)
    logits = np.asarray(out.logits, np.float32)
    assert logits.shape == (1, self.env.num_actions), logits.shape
    action, probs = self._decide(
        jnp.asarray(logits), last.legal_mask, decide_key
    )
    probs = np.asarray(probs)[0]
    assert probs.dtype == np.float64, probs.dtype
    decider = decider_for(self.policy, last.info)
    return Decision(
        action=int(np.asarray(action)[0]),
        probs=probs,
        legal=legal,
        decider=decider,
        logits=logits[0],
        value=np.asarray(out.value, np.float32)[0],
        identity=self.identity,
    )

  def _serve_rows(
      self, params, steps, seat, init_key, eval_key
  ) -> types.PolicyOutput:
    """The JAX path, one jitted call.

    The request's row replicated pad_rows times, each row run as the Arena runs
    a row (a vmap of batch-1 calls with the same keys, as core.arena._per_row):
    init_carry, observe_seq over the history, observe + evaluate on the last
    ply.

    Args:
      params: the policy params.
      steps: the request's SeatSteps [1, T].
      seat: i32[1], the seat.
      init_key: the init_carry key.
      eval_key: the evaluate key.

    Returns:
      Row 0's PolicyOutput, [1, ...].
    """
    pol, n = self.policy, self.pad_rows
    t = steps.seat.shape[1]
    rows = tree_map(lambda x: jnp.broadcast_to(x[0], (n,) + x.shape[1:]), steps)

    def _one(seq):
      """Serves one row's sequence: observe all but the last ply, then evaluate.

      Args:
        seq: The row's SeatSteps [T].

      Returns:
        The row's PolicyOutput.
      """
      s1 = tree_map(lambda x: x[None], seq)  # [1, T]
      carry = pol.init_carry(params, 1, seat, init_key)
      if pol.spec.stateful and t > 1:
        carry = pol.observe_seq(
            params, carry, tree_map(lambda x: x[:, :-1], s1)
        )
      last = tree_map(lambda x: x[:, -1], s1)
      carry = pol.observe(params, carry, last)
      return tree_map(
          lambda x: x[0], pol.evaluate(params, carry, last, eval_key)
      )

    out = jax.vmap(_one)(rows)
    return tree_map(lambda x: x[:1], out)

  def _serve_last_rows(
      self, params, last, seat, init_key, eval_key
  ) -> types.PolicyOutput:
    """_serve_rows of a stateless policy, given the last ply alone ([1, ...]).

    The same calls on the same values (init_carry, observe + evaluate on the
    last ply), with no history axis, so one compile serves every length.

    Args:
      params: the policy's params.
      last: the last ply's SeatStep [1].
      seat: the seat served.
      init_key: the init_carry key.
      eval_key: the evaluate key.

    Returns:
      The PolicyOutput [1, ...].
    """
    pol, n = self.policy, self.pad_rows
    rows = tree_map(lambda x: jnp.broadcast_to(x[0], (n,) + x.shape[1:]), last)

    def one(st):
      """The policy's output on one row's last ply.

      Args:
        st: the row's SeatStep (unbatched).

      Returns:
        The PolicyOutput of the row.
      """
      s1 = tree_map(lambda x: x[None], st)  # [1]
      carry = pol.init_carry(params, 1, seat, init_key)
      carry = pol.observe(params, carry, s1)
      return tree_map(lambda x: x[0], pol.evaluate(params, carry, s1, eval_key))

    out = jax.vmap(one)(rows)
    return tree_map(lambda x: x[:1], out)

  def _run_policy(self, rep, last, init_key, eval_key) -> types.PolicyOutput:
    """init_carry, observe_seq, evaluate on the last ply.

    A stateless policy: one observe of the last ply.

    Args:
      rep: the request's replay: a PaddedSeatReplay [1, T] (stateful policy)
        or a LastSeatStep (stateless).
      last: the last ply's SeatStep [1].
      init_key: the init_carry key.
      eval_key: the evaluate key.

    Returns:
      The PolicyOutput [1, ...] as numpy.
    """
    if self.host_driven:
      pol, p = self.policy, self.params
      last_np = host.to_numpy(last)
      carry = pol.init_carry(
          p, 1, np.asarray([self.seat], np.int32), host.raw_keys(init_key)[None]
      )
      if pol.spec.stateful:
        carry = pol.observe_seq(p, carry, host.to_numpy(rep.exact().steps))
      else:
        carry = pol.observe(p, carry, last_np)
      out = pol.evaluate(p, carry, last_np, host.raw_keys(eval_key)[None])
      assert isinstance(out, types.PolicyOutput), type(out)
      return host.to_numpy(out)
    seat = jnp.asarray([self.seat], jnp.int32)
    if self.stateful:
      return host.to_numpy(
          self._serve(self.params, rep.exact().steps, seat, init_key, eval_key)
      )
    return host.to_numpy(
        self._serve_last(self.params, last, seat, init_key, eval_key)
    )

  def warmup(self, max_plies: int) -> None:
    """Compiles every call a decision on a match of at most max_plies makes.

    Module doc, "Compiles": the replay's buckets, the keys, the decision rule
    and, for a stateless JAX policy, the policy. A stateful JAX policy still
    compiles once per history length on first use.

    Args:
      max_plies: the longest match in plies.

    Raises:
      ValueError: max_plies < 1.
    """
    if int(max_plies) < 1:
      raise ValueError(f"max_plies must be >= 1, got {max_plies}")
    if int(max_plies) <= self.warmed_plies:
      return  # compiled already (jit caches persist in the process)
    replay.warmup(
        self.env, self.seat, int(max_plies), last_only=not self.stateful
    )
    seat_key = jax.random.key(0)
    init_key, eval_key, decide_key = self._keys(seat_key, jnp.int64(0))
    # A last ply of this seat's shapes, every action legal (only shapes and
    # dtypes matter to a compile).
    last = tree_map(
        jnp.asarray, replay.first_decision_step(self.env, self.seat)
    )
    mask = jnp.asarray(np.ones((1, self.env.num_actions), np.bool_))
    jax.block_until_ready(
        self._decide(
            jnp.zeros((1, self.env.num_actions), jnp.float32), mask, decide_key
        )
    )
    # act lifts the policy's numpy logits with jnp.asarray, which since JAX
    # 0.12 compiles a staging op per shape and dtype on first use: lift one.
    jax.block_until_ready(
        jnp.asarray(np.zeros((1, self.env.num_actions), np.float32))
    )
    if not self.host_driven and not self.stateful:
      last = last._replace(legal_mask=mask)
      out = self._serve_last(
          self.params,
          last,
          jnp.asarray([self.seat], jnp.int32),
          init_key,
          eval_key,
      )
      jax.block_until_ready(out)
    self.warmed_plies = int(max_plies)
