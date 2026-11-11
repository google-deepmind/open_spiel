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

"""PolicyServer over pyspiel states and on-disk artifacts.

PolicyServer over pyspiel states and on-disk artifacts: a reply equals the
HostAgent decision for the same history and seat key, is deterministic per
seed, reports forced moves, refuses states it cannot answer, and caches
artifacts up to its cache size.
"""

import os
import sys

import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import tic_tac_toe
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.serve.tests import helpers
import pyspiel


@pytest.fixture(name="kuhn_ckpt", scope="module")
def _kuhn_ckpt_fixture(tmp_path_factory):
  """A Kuhn tabular checkpoint.

  Args:
    tmp_path_factory: The pytest fixture.

  Returns:
    The checkpoint folder.
  """
  d = tmp_path_factory.mktemp("kuhn_ckpt")
  pol, params = helpers.kuhn_tabular(kuhn.KuhnEnv())
  ckpt.write_artifact(
      str(d),
      3,
      pol,
      params,
      rules_sha="r",
      config_hash="c",
      algo="test",
      env_name="kuhn_poker",
      default_rule=types.SAMPLE,
      run_id="42",
  )
  return str(d)


@pytest.fixture(name="ttt_ckpt", scope="module")
def _ttt_ckpt_fixture(tmp_path_factory):
  """A tic-tac-toe history-policy checkpoint.

  Args:
    tmp_path_factory: The pytest fixture.

  Returns:
    The checkpoint folder.
  """
  d = tmp_path_factory.mktemp("ttt_ckpt")
  pol, params = helpers.history(tic_tac_toe.TicTacToeEnv(), cross_hand=False)
  ckpt.write_artifact(
      str(d),
      1,
      pol,
      params,
      rules_sha="r",
      config_hash="c",
      algo="test",
      env_name="tic_tac_toe",
      default_rule=types.ARGMAX,
  )
  return str(d)


def _kuhn_state(actions):
  """The pyspiel Kuhn state after `actions`.

  Args:
    actions: The actions.

  Returns:
    The state.
  """
  s = pyspiel.load_game("kuhn_poker").new_initial_state()
  for a in actions:
    s.apply_action(a)
  return s


def test_reply_equals_the_host_agent_decision(kuhn_ckpt):
  """Reply equals the host agent decision."""
  server = server_lib.PolicyServer()
  for actions in ([0, 1], [2, 0, 0], [1, 2, 1], [0, 2, 0, 1]):
    state = _kuhn_state(actions)
    reply = server.act(state, checkpoint_dir=kuhn_ckpt, step=3, seed=11)
    assert reply.action in state.legal_actions()
    assert set(reply.probs) == set(state.legal_actions())
    np.testing.assert_allclose(sum(reply.probs.values()), 1.0, atol=1e-12)
    assert reply.decider == "network" and reply.artifact == "42@3"
    seat = state.current_player()
    agent = host_agent.HostAgent(ckpt.load_artifact(kuhn_ckpt, 3), seat)
    d = agent.act(
        wire.hands_from_state(state), seat_key=server_lib.seat_key(11, seat)
    )
    assert d.action == reply.action
    assert reply.probs == {a: float(d.probs[a]) for a in state.legal_actions()}
    assert reply.value == float(d.value[seat])


def test_reply_json():
  """Reply json."""
  r = server_lib.Reply(
      action=1,
      probs={1: 0.75, 0: 0.25},
      value=None,
      decider="forced",
      artifact="7@3",
  )
  assert r.to_json() == {
      "action": 1,
      "probs": {"0": 0.25, "1": 0.75},
      "value": None,
      "decider": "forced",
      "artifact": "7@3",
  }


def test_replies_are_deterministic_per_seed(kuhn_ckpt):
  """Replies are deterministic per seed."""
  server = server_lib.PolicyServer()
  state = _kuhn_state([0, 1])
  kw = dict(checkpoint_dir=kuhn_ckpt, step=3)
  assert server.act(state, seed=5, **kw) == server.act(state, seed=5, **kw)


def test_forced_move(ttt_ckpt):
  """Forced move."""
  game = pyspiel.load_game("tic_tac_toe")
  s = game.new_initial_state()
  for a in (0, 1, 2, 4, 3, 5, 7, 6):
    s.apply_action(a)
  assert s.legal_actions() == [8] and not s.is_terminal()
  reply = server_lib.PolicyServer().act(s, checkpoint_dir=ttt_ckpt, step=1)
  assert reply.decider == "forced" and reply.action == 8
  assert reply.probs == {8: 1.0} and reply.value is None
  assert (
      reply.artifact == os.path.basename(ttt_ckpt) + "@1"
  )  # no run id: the dir name


def test_refusals(kuhn_ckpt):
  """Refusals."""
  server = server_lib.PolicyServer()
  leduc = pyspiel.load_game("leduc_poker").new_initial_state()
  leduc.apply_action(0)
  leduc.apply_action(2)
  with pytest.raises(ValueError, match="game"):
    server.act(leduc, checkpoint_dir=kuhn_ckpt, step=3)
  with pytest.raises(ValueError, match="chance"):
    server.act(_kuhn_state([0]), checkpoint_dir=kuhn_ckpt, step=3)
  with pytest.raises(ValueError, match="terminal"):
    server.act(_kuhn_state([0, 1, 0, 0]), checkpoint_dir=kuhn_ckpt, step=3)
  with pytest.raises(ValueError, match="cache_size"):
    server_lib.PolicyServer(cache_size=0)


def test_artifacts_are_cached_and_evicted_least_recently_used(
    kuhn_ckpt, ttt_ckpt
):
  """Artifacts are cached and evicted least recently used."""
  server = server_lib.PolicyServer(cache_size=1)
  kw = dict(checkpoint_dir=kuhn_ckpt, step=3)
  server.act(_kuhn_state([0, 1]), **kw)
  server.act(_kuhn_state([1, 0]), **kw)
  server.act(_kuhn_state([1, 0, 1]), **kw)
  assert server.num_loaded_artifacts() == 1
  s = pyspiel.load_game("tic_tac_toe").new_initial_state()
  server.act(s, checkpoint_dir=ttt_ckpt, step=1)
  assert server.num_loaded_artifacts() == 1
  assert server.act(_kuhn_state([0, 1]), **kw).artifact == "42@3"


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
