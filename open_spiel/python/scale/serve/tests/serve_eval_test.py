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

"""Serve equals eval.

Serve equals eval: on 1000 states per game and policy, HostAgent returns
what the in-graph Arena chose at that decision, to the policy kind's
tolerance (serve_eval_rule):

  BITWISE    no XLA float arithmetic in the logits: tabular, uniform and
             host-computed (callback, host-driven) policies, and RoutedPolicy
             over them: the replayed SeatSteps, the logits, decide's f64
             distribution and the action (sampling rules included, so the
             per-ply key derivation is checked too) equal the Arena's bit for
             bit.
  NUMERIC    every other NATIVE_JAX policy (networks, feed-forward or
             recurrent, search over a network, the history fixture), or a
             RoutedPolicy over one: the replayed SeatSteps and the action are
             exact, the logits agree within 1e-6 absolute plus 1e-6 relative
             (ATOL_NUMERIC, RTOL_NUMERIC: a few f32 ULPs at any magnitude) and
             the distribution within the bound that implies (probs_atol). The
             Arena evaluates inside its rollout scan and a stateless replay is
             a different XLA program, so the last f32 bits can differ (jax
             0.12's XLA:CPU fuses an MLP's dots differently in the two).

For every decision the Arena made (self-play, two hands per match so a
cross-hand carry matters), HostAgent gets the raw history of the match
(chance outcomes included), the Arena's seat key and the match's first row
ply. The tolerance is the policy kind's, never chosen per test, and a planted
wrong action or a recurrent drift past 1e-6 fails (test_check_cases_fails_*).
"""

import sys

from absl import logging
import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve.tests import arena_ref
from open_spiel.python.scale.serve.tests import helpers
from open_spiel.python.scale.serve.tests import serve_eval_checks
from open_spiel.python.scale.serve.tests import serve_fixtures
from open_spiel.python.scale.testing import fixtures

_PKG = host_agent.__name__.rpartition(".serve.")[0]  # the package's import name

N_STATES = 1000
BITWISE = serve_eval_checks.BITWISE
NUMERIC = serve_eval_checks.NUMERIC
check_cases = serve_eval_checks.check_cases
serve_eval_rule = serve_eval_checks.serve_eval_rule
ENVS = {
    "kuhn_poker": kuhn.KuhnEnv,
    "leduc_poker": leduc.LeducEnv,
    "connect_four": connect_four.ConnectFourEnv,
}
# (game, policy, rule, arena batch, arena steps)
GRID = [
    ("kuhn_poker", "history", types.SAMPLE, 64, 48),
    ("kuhn_poker", "tabular", types.SAMPLE, 64, 48),
    ("kuhn_poker", "mlp", types.ARGMAX, 64, 48),
    ("kuhn_poker", "rnn", types.SAMPLE, 64, 48),
    ("leduc_poker", "history", types.SAMPLE, 64, 48),
    ("leduc_poker", "mlp", types.SAMPLE, 64, 48),
    ("leduc_poker", "rnn", types.ARGMAX, 64, 48),
    ("connect_four", "history", types.SAMPLE, 32, 64),
    ("connect_four", "mlp", types.SAMPLE, 32, 64),
    ("connect_four", "rnn", types.SAMPLE, 32, 64),
]


@pytest.mark.parametrize(
    "game,pname,rule,batch,steps",
    GRID,
    ids=[f"{g}-{p}-{r.kind}" for g, p, r, _, _ in GRID],
)
def test_serve_equals_eval(game, pname, rule, batch, steps):
  """Serve equals eval."""
  env = ENVS[game]()
  pol, params = helpers.make(env, pname)
  cases = arena_ref.run(
      env,
      pol,
      params,
      rule,
      batch=batch,
      num_steps=steps,
      hands_per_match=2,
      seed=3,
      max_cases=N_STATES,
  )
  assert len(cases) == N_STATES, len(cases)
  # The set must exercise history: later hands of a match and long histories.
  assert sum(len(c.hands) > 1 for c in cases) > 50
  assert max(len(c.steps.seat) for c in cases) >= 3
  art = helpers.artifact(pol, params, rule)
  agents = [
      host_agent.HostAgent(art, seat, env=env)
      for seat in range(env.num_players)
  ]
  kind = serve_eval_rule(pol)
  assert kind == (BITWISE if pname == "tabular" else NUMERIC), (pname, kind)
  _, bitwise = check_cases(env, agents, cases, rule, exact=kind == BITWISE)
  logging.info(
      "serve == eval %s/%s (%s): %d / %d states bitwise",
      game,
      pname,
      kind,
      bitwise,
      len(cases),
  )
  if kind == BITWISE:
    assert bitwise == len(cases)


def _rnn_cases(n=40):
  """Recorded Kuhn RNN cases and host agents for both seats.

  Args:
    n: The most cases.

  Returns:
    (env, agents, cases).
  """
  env = kuhn.KuhnEnv()
  pol, params = helpers.make(env, "rnn")
  cases = arena_ref.run(
      env,
      pol,
      params,
      types.SAMPLE,
      batch=16,
      num_steps=24,
      hands_per_match=2,
      seed=3,
      max_cases=n,
  )
  art = helpers.artifact(pol, params, types.SAMPLE)
  return env, [host_agent.HostAgent(art, s, env=env) for s in range(2)], cases


def test_check_cases_fails_a_wrong_action_even_for_a_recurrent_policy():
  """Check cases fails a wrong action even for a recurrent policy."""
  env, agents, cases = _rnn_cases()
  check_cases(
      env, agents, cases, types.SAMPLE, exact=False
  )  # the true cases pass
  c = next(c for c in cases if int(np.asarray(c.legal).sum()) > 1)
  legal = np.flatnonzero(np.asarray(c.legal))
  wrong = int(next(a for a in legal if a != c.action))
  with pytest.raises(AssertionError):
    check_cases(
        env, agents, [c._replace(action=wrong)], types.SAMPLE, exact=False
    )


def test_check_cases_fails_a_recurrent_drift_past_1e6():
  """Check cases fails a recurrent drift past 1e6."""
  env, agents, cases = _rnn_cases()
  c = next(c for c in cases if int(np.asarray(c.legal).sum()) > 1)
  drift = np.asarray(c.logits, np.float32).copy()
  drift[np.flatnonzero(np.asarray(c.legal))[0]] += np.float32(1e-4)
  with pytest.raises(AssertionError):
    check_cases(
        env, agents, [c._replace(logits=drift)], types.SAMPLE, exact=False
    )


def test_serve_eval_rule_per_policy_kind():
  """Serve eval rule per policy kind."""
  env = kuhn.KuhnEnv()
  mlp, _ = helpers.make(env, "mlp")
  rnn, _ = helpers.make(env, "rnn")
  hist, _ = helpers.make(env, "history")
  tab, _ = helpers.make(env, "tabular")
  assert serve_eval_rule(mlp) == serve_eval_rule(hist) == NUMERIC
  assert (
      serve_eval_rule(rnn) == NUMERIC
      and rnn.spec.stateful
      and hist.spec.stateful
  )
  assert serve_eval_rule(tab) == BITWISE
  assert serve_eval_rule(uniform.UniformPolicy(env.name, 2, 2)) == BITWISE
  callback = fixtures.CallbackTabularPolicy(
      env.name, kuhn.NUM_INFO_STATES, 2, 2, helpers.KUHN_INDEX
  )
  assert serve_eval_rule(callback) == BITWISE
  r = routed.RoutedPolicy(
      rnn,
      uniform.UniformPolicy(env.name, 2, 2),
      f"{_PKG}.envs.gin.endgame:in_domain",
  )  # resolved, never called
  assert serve_eval_rule(r) == NUMERIC


def test_serve_equals_eval_callback_tabular():
  """Serve equals eval callback tabular."""
  env = kuhn.KuhnEnv()
  pol = fixtures.CallbackTabularPolicy(
      env.name, kuhn.NUM_INFO_STATES, 2, 2, helpers.KUHN_INDEX
  )
  pol.set_host_tables(
      np.asarray(
          jax.random.normal(jax.random.key(5), (1, kuhn.NUM_INFO_STATES, 2)),
          np.float32,
      )
  )
  params = pol.init_params()
  cases = arena_ref.run(
      env,
      pol,
      params,
      types.SAMPLE,
      batch=16,
      num_steps=40,
      hands_per_match=2,
      seed=4,
      max_cases=300,
  )
  art = helpers.artifact(pol, params, types.SAMPLE)
  agents = [host_agent.HostAgent(art, s, env=env) for s in range(2)]
  check_cases(env, agents, cases, types.SAMPLE)


def test_serve_equals_eval_with_forced_moves():
  """Serve equals eval with forced moves.

  A game with forced moves: HostAgent's forced decision (no evaluate) is the
  Arena's distribution and action at those plies too.
  """
  env = serve_fixtures.ForcedKuhnEnv()
  pol, params = helpers.history(env)
  cases = arena_ref.run(
      env,
      pol,
      params,
      types.SAMPLE,
      batch=32,
      num_steps=40,
      hands_per_match=2,
      seed=7,
      max_cases=600,
  )
  art = helpers.artifact(pol, params, types.SAMPLE)
  agents = [host_agent.HostAgent(art, s, env=env) for s in range(2)]
  kind = serve_eval_rule(pol)
  forced, bitwise = check_cases(
      env, agents, cases, types.SAMPLE, exact=kind == BITWISE
  )
  assert forced > 50, forced
  if kind == BITWISE:
    assert bitwise == len(cases)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
