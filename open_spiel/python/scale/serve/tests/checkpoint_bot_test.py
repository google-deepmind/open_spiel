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

"""checkpoint_bot on tiny kuhn and leduc artifacts (fast).

Privacy: at every decision of a seat, changing only the OTHER seat's private
card (pyspiel confirms the seat's information state is unchanged) leaves the
bot's action and distribution unchanged, while changing the seat's own card
changes the distribution somewhere (the test is not vacuous). The bot's move
equals PolicyServer.act's, a fixed seed reproduces a sampling bot, a game the
artifact does not play is refused, and the step defaults to the latest one,
also for an artifact read anonymously from a public HTTPS URL.
A rules difference between the env and pyspiel is played through (repair),
logged, counted and noticed; PolicyServer without repair still refuses it.
"""

import dataclasses
import sys

import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.ckpt import public
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve.tests import checkpoint_bot_helpers as cbh
from open_spiel.python.scale.serve.tests import helpers
from open_spiel.python.scale.testing import public_bucket
import pyspiel


def _kuhn_bot(deterministic=True, seed=0, **kw):
  """A tabular kuhn bot from an in-memory artifact.

  Args:
    deterministic: Whether the bot plays argmax.
    seed: The bot's seed.
    **kw: More memory_bot options.

  Returns:
    The bot.
  """
  pol, params = helpers.kuhn_tabular(kuhn.KuhnEnv())
  return cbh.memory_bot(
      pol, params, deterministic=deterministic, seed=seed, **kw
  )


def _leduc_bot(deterministic=True, seed=0):
  """A network leduc bot from an in-memory artifact.

  Args:
    deterministic: Whether the bot plays argmax.
    seed: The bot's seed.

  Returns:
    The bot.
  """
  pol, params = helpers.network(games.env_for("leduc_poker", {}), "mlp", seed=3)
  return cbh.memory_bot(pol, params, deterministic=deterministic, seed=seed)


@pytest.mark.parametrize("deterministic", [True, False])
def test_kuhn_privacy(deterministic):
  """Kuhn privacy."""
  bot = _kuhn_bot(deterministic=deterministic, seed=5)
  pairs, own = cbh.check_privacy(
      bot, pyspiel.load_game("kuhn_poker"), num_playouts=24, seed=1
  )
  assert pairs >= 10, pairs
  assert own, "no decision's distribution depends on the seat's own card"


@pytest.mark.parametrize("deterministic", [True, False])
def test_leduc_privacy(deterministic):
  """Leduc privacy."""
  bot = _leduc_bot(deterministic=deterministic, seed=7)
  pairs, own = cbh.check_privacy(
      bot, pyspiel.load_game("leduc_poker"), num_playouts=6, seed=2
  )
  assert pairs >= 10, pairs
  assert own, "no decision's distribution depends on the seat's own card"


def test_attributes_and_policy_server_agreement():
  """Attributes and policy server agreement."""
  bot = _kuhn_bot(deterministic=False, seed=11)
  assert bot.game_string == "kuhn_poker"
  assert bot.num_players == 2
  game = pyspiel.load_game("kuhn_poker")
  bot.check_serves(game)
  for hist in ([0, 1], [2, 0, 0], [1, 2, 0, 1], [2, 1, 1]):
    s = game.new_initial_state()
    for a in hist:
      s.apply_action(a)
    seat = s.current_player()
    a = bot.choose_action(s, seat)
    assert a in s.legal_actions()
    # Reaches private _server of the code under test: the reference move.
    ref = bot._server.act(  # pylint: disable=protected-access
        s,
        checkpoint_dir=bot.path,
        step=bot.step,
        seed=11,  # pylint: disable=protected-access
        rule=types.DecisionRule("sample", 1.0),
    )
    assert a == ref.action and bot.last_probs == dict(ref.probs)
    assert set(bot.last_probs) == set(s.legal_actions())
    assert abs(sum(bot.last_probs.values()) - 1.0) < 1e-9


def test_wrong_seat_refused():
  """Wrong seat refused."""
  bot = _kuhn_bot()
  s = pyspiel.load_game("kuhn_poker").new_initial_state()
  s.apply_action(0)
  s.apply_action(1)
  with pytest.raises(ValueError, match="not to move"):
    bot.choose_action(s, 1)


def test_fixed_seed_reproduces_and_seeds_differ():
  """Fixed seed reproduces and seeds differ."""
  game = pyspiel.load_game("kuhn_poker")
  runs = {}
  for seed in (3, 3, 4, 5, 6):
    bot = _kuhn_bot(deterministic=False, seed=seed)
    runs.setdefault(seed, []).append(cbh.self_play(bot, game, deals=range(12)))
  assert runs[3][0] == runs[3][1]
  assert len({tuple(map(tuple, r[0])) for r in runs.values()}) > 1


def test_game_mismatch_refused():
  """Game mismatch refused."""
  bot = _kuhn_bot()
  with pytest.raises(ValueError, match="plays 'kuhn_poker'"):
    bot.check_serves(pyspiel.load_game("leduc_poker"))
  with pytest.raises(ValueError, match="players"):
    bot.check_serves(pyspiel.load_game("kuhn_poker(players=3)"))
  s = pyspiel.load_game("leduc_poker").new_initial_state()
  s.apply_action(0)
  s.apply_action(1)
  with pytest.raises(ValueError, match="kuhn_poker"):
    bot.choose_action(s, 0)


def test_latest_step_from_disk(tmp_path):
  """Latest step from disk."""
  pol, params = helpers.kuhn_tabular(kuhn.KuhnEnv())
  for step in (2, 10, 7):
    ckpt.write_artifact(
        str(tmp_path),
        step,
        pol,
        params,
        rules_sha="r",
        config_hash="c",
        algo="test",
        env_name="kuhn_poker",
    )
  (tmp_path / "99").mkdir()  # a step directory that holds no artifact
  assert ckpt.steps(str(tmp_path)) == (2, 7, 10)
  bot = checkpoint_bot_lib.checkpoint_bot(str(tmp_path) + "/")
  assert bot.step == 10 and bot.path == str(tmp_path)
  with pytest.raises(FileNotFoundError):
    checkpoint_bot_lib.checkpoint_bot(str(tmp_path / "empty"))


def test_bot_from_a_public_url(tmp_path, monkeypatch):
  """A bot reads its artifact anonymously from a public HTTPS URL."""
  pol, params = helpers.kuhn_tabular(kuhn.KuhnEnv())
  run = tmp_path / "served" / "example-bucket" / "runs" / "kuhn"
  for step in (2, 5):
    ckpt.write_artifact(
        str(run),
        step,
        pol,
        params,
        rules_sha="r",
        config_hash="c",
        algo="test",
        env_name="kuhn_poker",
    )
  server = public_bucket.FakeBucketServer(str(tmp_path / "served")).start()
  try:
    monkeypatch.setattr(public, "ENDPOINT", server.url)
    monkeypatch.setenv(public.CACHE_ENV, str(tmp_path / "cache"))
    url = f"{server.url}/example-bucket/runs/kuhn"
    path, step = checkpoint_bot_lib.split_checkpoint(url + "/")
    bot = checkpoint_bot_lib.checkpoint_bot(path, step=step)
    local = checkpoint_bot_lib.checkpoint_bot(str(run), step=5)
    assert bot.step == 5 and bot.path == url
    game = pyspiel.load_game("kuhn_poker")
    for hist in ([0, 1], [2, 0, 0], [1, 2, 0, 1]):
      s = game.new_initial_state()
      for a in hist:
        s.apply_action(a)
      seat = s.current_player()
      assert bot.choose_action(s, seat) == local.choose_action(s, seat)
      assert bot.last_probs == local.last_probs
  finally:
    server.stop()


def test_bad_temperature_refused():
  """Bad temperature refused."""
  with pytest.raises(ValueError, match="temperature"):
    checkpoint_bot_lib.checkpoint_bot(
        "/nonexistent", step=1, deterministic=False, temperature=0.0
    )


def test_split_checkpoint():
  """Split checkpoint."""
  split = checkpoint_bot_lib.split_checkpoint
  assert split("/a/b@12") == ("/a/b", 12)
  assert split("/a/run") == ("/a/run", None)
  assert split("/a@b/c") == ("/a@b/c", None)


def test_policy_server_rule_override_is_cached_per_rule():
  """Policy server rule override is cached per rule."""
  pol, params = helpers.kuhn_tabular(kuhn.KuhnEnv())
  srv = server_lib.PolicyServer(loader=cbh.memory_loader(pol, params))
  a = srv.agent("mem://k", 1, 0, "kuhn_poker", {}, rule=types.ARGMAX)
  b = srv.agent("mem://k", 1, 0, "kuhn_poker", {}, rule=types.SAMPLE)
  assert a is not b and a.rule == types.ARGMAX and b.rule == types.SAMPLE
  # manifest default
  assert srv.agent("mem://k", 1, 0, "kuhn_poker", {}).rule == types.SAMPLE


class _RulesDiffer:
  """A pyspiel state whose legal actions (and history) a test replaces.

  Stands in for a node where the env's rules and pyspiel's disagree; every
  other call goes to the real state.
  """

  def __init__(self, state, legal, history=None):
    self._state = state
    self._legal = list(legal)
    self._history = history

  def legal_actions(self, *unused_args):
    """The replaced legal actions.

    Args:
      *unused_args: pyspiel's optional player, ignored.

    Returns:
      The actions.
    """
    return list(self._legal)

  def history(self):
    """The replaced (or the real) history.

    Returns:
      The history.
    """
    if self._history is None:
      return self._state.history()
    return list(self._history)

  def __getattr__(self, name):
    return getattr(self._state, name)

  def __str__(self):
    return str(self._state)


def _leduc_first_decision():
  """A leduc state at its first decision (legal: call 1, raise 2).

  Returns:
    The state.
  """
  s = pyspiel.load_game("leduc_poker").new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  assert s.legal_actions() == [1, 2], s.legal_actions()
  return s


def _warnings(monkeypatch):
  """Records server_lib's WARNING log lines.

  Args:
    monkeypatch: pytest monkeypatch fixture.

  Returns:
    The list the lines are appended to.
  """
  lines = []
  monkeypatch.setattr(
      server_lib.logging,
      "warning",
      lambda msg, *args: lines.append(msg % args),
  )
  return lines


def test_legal_set_mismatch_plays_the_policy_over_both_sets(monkeypatch):
  """The seat plays on, renormalised over the actions both sides allow."""
  warned = _warnings(monkeypatch)
  bot = _leduc_bot()
  real = _leduc_first_decision()
  differ = _RulesDiffer(real, [0, 1])  # pyspiel: fold, call; env: call, raise
  a = bot.choose_action(differ, 0)
  assert a == 1, a
  assert bot.last_probs == {0: 0.0, 1: 1.0}, bot.last_probs
  assert bot.stats == {"legal_mismatch": 1, "diverged": 0}, bot.stats
  assert bot.last_notice.startswith("engine/agent rules differ on this move")
  assert "best engine-legal move" in bot.last_notice
  assert len(bot.last_notice) <= checkpoint_bot_lib.NOTICE_MAX_LENGTH
  assert len(warned) == 1 and "legal-set mismatch" in warned[0], warned
  assert "env [1, 2], pyspiel [0, 1]" in warned[0], warned
  # an ordinary move clears the notice and counts nothing
  bot.choose_action(real, 0)
  assert bot.last_notice is None and bot.stats["legal_mismatch"] == 1
  # empty intersection: a pyspiel-legal action is still played
  assert bot.choose_action(_RulesDiffer(real, [0]), 0) == 0
  assert bot.stats["legal_mismatch"] == 2 and bot.last_notice is not None


def test_divergence_earlier_in_the_hand_plays_a_legal_move(monkeypatch):
  """An env that cannot replay the hand still answers, noticed and counted."""
  warned = _warnings(monkeypatch)
  bot = _leduc_bot()
  real = _leduc_first_decision()
  # fold (0) with no bet to call: legal in no leduc env, so the replay fails
  differ = _RulesDiffer(real, real.legal_actions(), history=[0, 3, 0])
  assert bot.choose_action(differ, 0) == 1  # pyspiel's first legal action
  assert bot.stats == {"legal_mismatch": 0, "diverged": 1}, bot.stats
  # The seat's notice (contract 8 shows it to the seat): the replay's error,
  # which can name another seat's card, stays in the log.
  assert bot.last_notice == checkpoint_bot_lib.DIVERGED_NOTICE.format(
      move=real.action_to_string(0, 1)
  ), bot.last_notice
  assert len(warned) == 1 and "diverged earlier" in warned[0], warned


def test_policy_server_without_repair_still_refuses():
  """Arena-like callers keep the strict check: no repair, no answer."""
  bot = _leduc_bot()
  real = _leduc_first_decision()
  # Reaches private _server of the code under test.
  srv = bot._server  # pylint: disable=protected-access
  with pytest.raises(ValueError, match="legal set"):
    srv.act(
        _RulesDiffer(real, [0, 1]),
        checkpoint_dir=bot.path,
        step=bot.step,
        seed=0,
    )
  with pytest.raises(ValueError, match="illegal action"):
    srv.act(
        _RulesDiffer(real, [1, 2], history=[0, 3, 0]),
        checkpoint_dir=bot.path,
        step=bot.step,
        seed=0,
    )
  assert srv.stats == {"legal_mismatch": 0, "diverged": 0}, srv.stats


def test_restrict_to_legal():
  """restrict_to_legal: intersection, highest logit, first legal."""
  d = host_agent.Decision(
      action=1,
      probs=np.array([0.0, 0.6, 0.4]),
      legal=np.array([False, True, True]),
      decider="network",
      logits=np.array([3.0, 1.0, 2.0], np.float32),
      value=None,
      identity={},
  )
  r, how = server_lib.restrict_to_legal(d, [0, 2])
  assert (r.action, r.distribution(), how) == (
      2,
      {0: 0.0, 2: 1.0},
      "the policy over the actions both allow",
  )
  r, how = server_lib.restrict_to_legal(d, [0])
  assert r.action == 0 and "highest logit" in how, how
  forced = dataclasses.replace(d, logits=None)
  r, how = server_lib.restrict_to_legal(forced, [0])
  assert r.action == 0 and how == "pyspiel's first legal action", how
  only_env = host_agent.Decision(
      action=0,
      probs=np.array([1.0, 0.0, 0.0]),
      legal=np.array([True, False, False]),
      decider="network",
      logits=np.array([0.0, 1.0, 5.0], np.float32),
      value=None,
      identity={},
  )
  assert server_lib.restrict_to_legal(only_env, [1, 2])[0].action == 2
  with pytest.raises(ValueError, match="not ids"):
    server_lib.restrict_to_legal(d, [3])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
