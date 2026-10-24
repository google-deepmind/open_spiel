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

"""HostAgent alone.

HostAgent alone: one evaluate per request (a search never searches the
history), forced moves, the decider and identity on each decision, dtypes,
determinism, fail-loud inputs, and an artifact loaded from disk.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.gin import (
    reference_eval as gin_reference_eval,
)
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.policies.tests import named_fns
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve.tests import helpers
from open_spiel.python.scale.serve.tests import serve_fixtures as sf

J, Q, K = 0, 1, 2
PASS, BET = 0, 1

# Histories ending at a decision with several earlier decisions of the seat to
# move.
LONG = {
    "kuhn_poker": (kuhn.KuhnEnv, [[J, Q, BET, PASS], [Q, K, PASS, BET]], 0),
    "leduc_poker": (leduc.LeducEnv, [[0, 2, 2, 2, 1, 5, 2, 2]], 0),
    "connect_four": (
        connect_four.ConnectFourEnv,
        [[3, 3, 4, 4, 2, 2, 0, 1]],
        0,
    ),
}


def _key(i=0):
  """A PRNG key.

  Args:
    i: The seed.

  Returns:
    The key.
  """
  return jax.random.key(i)


@pytest.mark.parametrize("game", sorted(LONG))
@pytest.mark.parametrize("stateful", [True, False])
def test_search_never_searches_historical_plies(game, stateful):
  """Search never searches historical plies."""
  ctor, hands, seat = LONG[game]
  env = ctor()
  rep = replay.replay_seat_steps(env, hands, seat)
  t = rep.num_plies
  own_plies = int(np.sum(np.asarray(rep.steps.to_move[0]) == seat))
  assert own_plies >= 3 and t >= 5, (own_plies, t)
  pol = sf.CountingSearchPolicy(
      env.name, env.num_actions, env.num_players, sims=4, stateful=stateful
  )
  params = pol.init_params()
  agent = host_agent.HostAgent(helpers.artifact(pol, params), seat, env=env)
  n = agent.pad_rows
  sf.reset_counts()
  reply = agent.act(hands, seat_key=_key())
  c = sf.counts()
  assert reply.decider == "search" and reply.searched
  # ONE evaluate call (over the pad_rows replicas of the request's row; a
  # vmapped debug callback fires once per call), at the last ply only: the
  # ply counter it saw is T (stateful) / 1 (stateless observes the last ply).
  assert c.get("evaluate", 0) == 1, (c, n)
  assert c.get("sim", 0) == 4, c
  assert set(sf.evaluated_plies()) == {
      t if stateful else 1
  }, sf.evaluated_plies()
  # observe_seq once over the T plies (stateful); the last ply only (stateless).
  assert c.get("observe", 0) == (t if stateful else 1), (c, t)
  # Control: the counters do see per-ply evaluation (evaluate_seq evaluates
  # every ply, 1..T).
  sf.reset_counts()
  carry = pol.init_carry(params, 1, jnp.asarray([seat], jnp.int32), _key())
  jax.block_until_ready(pol.evaluate_seq(params, carry, rep.steps, None))
  assert sf.counts().get("evaluate", 0) == t
  assert sorted(sf.evaluated_plies()) == list(range(1, t + 1))


def test_forced_move_skips_evaluate():
  """Forced move skips evaluate."""
  env = sf.ForcedKuhnEnv()
  pol = sf.CountingSearchPolicy(env.name, 2, 2, sims=2)
  agent = host_agent.HostAgent(
      helpers.artifact(pol, pol.init_params(), types.SAMPLE), 1, env=env
  )
  sf.reset_counts()
  reply = agent.act([[K, J, PASS]], seat_key=_key())
  c = sf.counts()
  assert reply.decider == "forced" and not reply.searched
  assert c.get("evaluate", 0) == 0 and c.get("observe", 0) == 0, c
  assert reply.action == PASS
  np.testing.assert_array_equal(reply.probs, [1.0, 0.0])
  assert reply.probs.dtype == np.float64
  assert reply.distribution() == {PASS: 1.0}
  assert reply.logits is None
  # Not forced: facing a bet, both actions are legal and the policy is asked.
  sf.reset_counts()
  reply = agent.act([[K, J, BET]], seat_key=_key())
  assert reply.decider == "search" and sf.counts()["evaluate"] == 1
  assert set(sf.evaluated_plies()) == {2}


def test_decision_identity_dtypes_and_distribution():
  """Decision identity dtypes and distribution."""
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  art = helpers.artifact(pol, params, types.ARGMAX, run_id="123", step=45)
  agent = host_agent.HostAgent(art, 0, env=env)
  reply = agent.act([[K, Q]], seat_key=_key())
  assert reply.decider == "network" and not reply.searched
  assert isinstance(reply.action, int)
  assert reply.probs.shape == (2,) and reply.probs.dtype == np.float64
  assert reply.logits.shape == (2,) and reply.logits.dtype == np.float32
  assert reply.value.shape == (2,) and reply.value.dtype == np.float32
  np.testing.assert_allclose(reply.probs.sum(), 1.0, rtol=0, atol=1e-15)
  # Known answer: the tabular row for (K, no history), softmax in f64, argmax.
  row = np.asarray(params["logits"])[K * len(kuhn.HISTORIES)]
  want = (
      np.exp(row.astype(np.float64) - row.max())
      / np.exp(row.astype(np.float64) - row.max()).sum()
  )
  np.testing.assert_allclose(reply.probs, want, rtol=1e-12)
  assert reply.action == int(np.argmax(row))
  ident = reply.identity
  assert (
      ident["label"] == "123@45"
      and ident["run_id"] == "123"
      and ident["step"] == 45
  )
  for k in (
      "arch_name",
      "params_sha256",
      "game",
      "rules_sha",
      "obs_format",
      "framework",
      "capability",
      "git_commit",
      "path",
      "schema_version",
      "rule",
  ):
    assert k in ident, k
  assert ident["arch_name"] == "tabular"
  assert ident["params_sha256"] == "sha-test"
  assert ident["rule"] == {"kind": "argmax", "temperature": 1.0}
  assert set(reply.distribution()) == {0, 1}


def test_decider_classification():
  """Decider classification."""
  env = kuhn.KuhnEnv()
  uni = uniform.UniformPolicy(env.name, 2, 2)
  assert host_agent.decider_of(uni) == "scripted"
  assert host_agent.decider_of(helpers.kuhn_tabular(env)[0]) == "network"
  assert host_agent.decider_of(helpers.network(env, "mlp")[0]) == "network"
  assert (
      host_agent.decider_of(sf.CountingSearchPolicy(env.name, 2, 2)) == "search"
  )

  plain = uniform.UniformPolicy(env.name, 2, 2)
  plain.decider = None  # nothing declared: a network
  assert host_agent.decider_of(plain) == "network"
  unk = uniform.UniformPolicy(env.name, 2, 2)
  unk.decider = "oracle"
  with pytest.raises(ValueError, match="decider"):
    host_agent.decider_of(unk)


def test_label_falls_back_to_the_directory_name():
  """Label falls back to the directory name."""
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  art = helpers.artifact(
      pol, params, run_id=None, step=3, path="/ckpts/kuhn_run"
  )
  assert host_agent.label_of(art) == "kuhn_run@3"
  assert host_agent.HostAgent(art, 0, env=env).identity["label"] == "kuhn_run@3"


def test_sampling_is_keyed_and_deterministic():
  """Sampling is keyed and deterministic."""
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  pol_params = pol.params_from_logits(
      jnp.zeros((12, 2), jnp.float32)
  )  # 50/50 everywhere
  agent = host_agent.HostAgent(
      helpers.artifact(pol, pol_params, types.SAMPLE), 0, env=env
  )
  a = [agent.act([[J, Q]], seat_key=_key(i)).action for i in range(40)]
  b = [agent.act([[J, Q]], seat_key=_key(i)).action for i in range(40)]
  assert a == b
  assert set(a) == {0, 1}
  del params


def test_rule_override():
  """Rule override."""
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  art = helpers.artifact(pol, params, types.SAMPLE)
  assert host_agent.HostAgent(art, 0, env=env).rule == types.SAMPLE
  assert (
      host_agent.HostAgent(art, 0, env=env, rule=types.ARGMAX).rule
      == types.ARGMAX
  )


def test_fail_loud():
  """Fail loud."""
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  art = helpers.artifact(pol, params)
  agent = host_agent.HostAgent(art, 0, env=env)
  with pytest.raises(ValueError, match="to move"):
    agent.act([[J, Q, PASS]], seat_key=_key())  # seat 1 to move
  with pytest.raises(ValueError, match="terminal"):
    agent.act([[J, Q, PASS, PASS]], seat_key=_key())
  with pytest.raises(ValueError, match="seat"):
    host_agent.HostAgent(art, 2, env=env)
  with pytest.raises(ValueError, match="game"):
    host_agent.HostAgent(art, 0, env=leduc.LeducEnv())
  with pytest.raises(ValueError, match="first_ply"):
    agent.act([[J, Q]], seat_key=_key(), first_ply=-1)


def test_env_resolved_from_manifest_game():
  """Env resolved from manifest game."""
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  agent = host_agent.HostAgent(helpers.artifact(pol, params), 0)
  assert agent.env.name == "kuhn_poker"


def test_loaded_artifact(tmp_path):
  """Loaded artifact.

  write_artifact -> load_artifact -> HostAgent: the decision echoes the manifest
  identity and equals the in-memory agent's decision.

  Args:
    tmp_path: pytest's temporary directory.
  """
  env = kuhn.KuhnEnv()
  pol, params = helpers.kuhn_tabular(env)
  man = ckpt.write_artifact(
      str(tmp_path),
      9,
      pol,
      params,
      rules_sha="r1",
      config_hash="c1",
      algo="test",
      env_name="kuhn_poker",
      default_rule=types.SAMPLE,
      run_id="555",
      git_commit="abc",
  )
  art = ckpt.load_artifact(str(tmp_path), 9)
  agent = host_agent.HostAgent(art, 1)
  mem = host_agent.HostAgent(
      helpers.artifact(pol, params, types.SAMPLE), 1, env=env
  )
  for hist in ([[J, Q, PASS]], [[K, J, BET]], [[Q, K, PASS]]):
    r, m = agent.act(hist, seat_key=_key(3)), mem.act(hist, seat_key=_key(3))
    np.testing.assert_array_equal(r.probs, m.probs)
    assert r.action == m.action
  ident = agent.act([[J, Q, PASS]], seat_key=_key()).identity
  assert ident["label"] == "555@9"
  assert ident["params_sha256"] == man.params_sha256
  assert ident["path"] == str(tmp_path)
  assert ident["rule"] == {"kind": "sample", "temperature": 1.0}


# --- real search and routed policies ---------------


def test_real_search_policy_is_served_c4():
  """Real search policy is served c4.

  The real SearchPolicy (not the counting fixture) over connect_four with
  PerfectInformationSampler: HostAgent accepts it and every non-forced reply is
  searched.
  """
  env = connect_four.ConnectFourEnv()
  base, _ = helpers.network(env, "mlp")
  pol = search.SearchPolicy(
      base,
      env,
      perfect_information.PerfectInformationSampler(env),
      search.MCTSConfig(num_simulations=4, dirichlet_fraction=0.0),
  )
  assert host_agent.decider_of(pol) == "search"
  params = pol.init_params(_key(3))
  agent = host_agent.HostAgent(
      helpers.artifact(pol, params, types.SAMPLE), 0, env=env
  )
  assert agent.decider == "search"
  reply = agent.act([[3, 3, 4, 4]], seat_key=_key(1))
  assert reply.decider == "search" and reply.searched
  assert reply.probs.dtype == np.float64 and reply.legal.all()
  np.testing.assert_allclose(reply.probs.sum(), 1.0, rtol=0, atol=1e-12)
  again = agent.act([[3, 3, 4, 4]], seat_key=_key(1))
  np.testing.assert_array_equal(again.probs, reply.probs)
  assert again.action == reply.action


def test_routed_policy_reports_the_routed_part_per_request():
  """Routed policy reports the routed part per request.

  RoutedPolicy(network tabular, uniform, facing_bet) on kuhn: the decision's
  decider is the override's ("scripted") facing a bet, the base's ("network")
  otherwise; the logits are the routed part's (known answer).
  """
  env = kuhn.KuhnEnv()
  tab, tab_params = helpers.kuhn_tabular(env)
  uni = uniform.UniformPolicy(env.name, 2, 2)
  pol = routed.RoutedPolicy(tab, uni, named_fns.FACING_BET)
  params = pol.make_params(tab_params, {})
  assert host_agent.decider_of(pol) == "network"
  agent = host_agent.HostAgent(
      helpers.artifact(pol, params, types.SAMPLE), 1, env=env
  )
  facing = agent.act([[K, Q, BET]], seat_key=_key())
  assert facing.decider == "scripted" and not facing.searched
  np.testing.assert_allclose(facing.probs, [0.5, 0.5], rtol=0, atol=1e-15)
  open_ = agent.act([[K, Q, PASS]], seat_key=_key())
  assert open_.decider == "network"


def test_gin_rule_bot_with_scripted_endgame_is_classifiable():
  """The gin rule bot, routed to the scripted endgame, has a decider."""
  bot = gin_reference_eval.rule_bot_entrant()
  routed_bot = gin_reference_eval.route_endgame(
      bot, env_registry.game("gin_rummy")
  )
  assert host_agent.decider_of(routed_bot.policy) == "scripted"


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
