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

"""remote_bot in process: a leduc artifact on disk as a full_state Remote bot.

The reply to a full_state request built as the GUI builds it (contract 8:
exactly FULL_STATE_KEYS, the protocol version in a header, not the body) is
checkpoint_bot's move at the deserialized state, with checkpoint_bot's
notice kept to what the seat may see and cut to the protocol's length (a
blank one sent as None); a request the bot cannot answer, one missing a
contract key included, is a ValueError (the reference server's HTTP 422).
The artifact comes from the default or the request's config; a public bucket
URL (a local stand-in) serves like the local folder, and any other http(s)
URL is refused. The HTTP round trip through OpenSpiel's reference bot server
is remote_bot_http_test.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.ckpt import public
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import remote_bot
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve.tests import remote_bot_helpers as rbh
from open_spiel.python.scale.testing import public_bucket
import pyspiel


@pytest.fixture(name="leduc_dir", scope="module")
def _leduc_dir(tmp_path_factory):
  """A leduc artifact on disk, written once per module.

  Args:
    tmp_path_factory: pytest's temporary directory factory.

  Returns:
    The artifact's directory.
  """
  return rbh.write_leduc(tmp_path_factory.mktemp("leduc"))


def test_request_is_contract_8s():
  """The test request is contract 8's /spiel/step body, key for key."""
  game = pyspiel.load_game(rbh.GAME)
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  body = rbh.request_of(game, s, 0)
  assert tuple(body) == remote_bot.FULL_STATE_KEYS, list(body)
  assert "protocol" not in body  # contract 8: the Spiel-Protocol header
  assert (remote_bot.PROTOCOL_HEADER, remote_bot.PROTOCOL) == (
      "Spiel-Protocol",
      1,
  )
  assert body["game_encoding"] == {
      "num_distinct_actions": game.num_distinct_actions()
  }


def test_reply_is_checkpoint_bots_move(leduc_dir):
  """The served move is checkpoint_bot's at the deserialized state."""
  served = remote_bot.RemoteCheckpointBot(leduc_dir, warmup=False)
  assert served.games == (rbh.GAME,)
  ref = checkpoint_bot_lib.checkpoint_bot(leduc_dir)
  game = pyspiel.load_game(rbh.GAME)
  rng = np.random.default_rng(0)
  n = 0
  for g in range(8):

    def choose(s, seat):
      """The served reply at `s` for `seat`.

      Args:
        s: the state.
        seat: the seat to move.

      Returns:
        (action, notice).
      """
      reply = served.step("full_state", rbh.request_of(game, s, seat))
      assert set(reply) == {"action", "notice"}, reply
      return reply["action"], reply["notice"]

    for hist, action, notice in rbh.play(game, choose, bot_seat=g % 2, rng=rng):
      s = game.new_initial_state()
      for a in hist:
        s.apply_action(a)
      assert action == ref.choose_action(s, s.current_player()), hist
      assert notice is None
      n += 1
  assert n >= 8, n


def test_config_names_the_artifact(leduc_dir):
  """With no default artifact, the request's config names it."""
  served = remote_bot.RemoteCheckpointBot(None, warmup=False)
  assert served.games is None
  game = pyspiel.load_game(rbh.GAME)
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  reply = served.step(
      "full_state", rbh.request_of(game, s, 0, {"checkpoint": leduc_dir + "@3"})
  )
  assert reply["action"] in s.legal_actions()
  with pytest.raises(ValueError, match="no artifact"):
    served.step("full_state", rbh.request_of(game, s, 0))
  env = {remote_bot.CHECKPOINT_ENV: leduc_dir}
  assert remote_bot.make_bot(env).games == (rbh.GAME,)


def test_refusals(leduc_dir):
  """A request the bot cannot answer is a ValueError (HTTP 422)."""
  served = remote_bot.RemoteCheckpointBot(leduc_dir, warmup=False)
  game = pyspiel.load_game(rbh.GAME)
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  good = rbh.request_of(game, s, 0)
  with pytest.raises(ValueError, match="full_state requests"):
    served.step("seat_view", good)
  with pytest.raises(ValueError, match="history"):
    served.step("full_state", dict(good, history=[0, 4]))
  with pytest.raises(ValueError, match="not to move"):
    served.step("full_state", dict(good, seat=1))
  with pytest.raises(ValueError, match="legal actions"):
    served.step("full_state", dict(good, legal_actions=[1]))
  for key in remote_bot.FULL_STATE_KEYS:
    with pytest.raises(ValueError, match=f"no {key!r}"):
      served.step("full_state", {k: v for k, v in good.items() if k != key})
  with pytest.raises(ValueError, match="game_encoding"):
    served.step(
        "full_state", dict(good, game_encoding={"num_distinct_actions": 9})
    )
  with pytest.raises(ValueError, match="config is not an object"):
    served.step("full_state", dict(good, config="{}"))
  assert served.step("full_state", dict(good, extra=1))["action"] in (
      s.legal_actions()
  )
  kuhn = pyspiel.load_game("kuhn_poker")
  k = kuhn.new_initial_state()
  k.apply_action(0)
  k.apply_action(1)
  with pytest.raises(ValueError, match="kuhn_poker"):
    served.step("full_state", rbh.request_of(kuhn, k, 0))
  with pytest.raises(ValueError, match="deterministic"):
    served.step("full_state", dict(good, config={"deterministic": "yes"}))


def test_notice_is_the_bots_and_clipped():
  """last_notice reaches the reply, cut to the protocol's 200 characters."""
  assert remote_bot.clip_notice(None) is None
  assert remote_bot.clip_notice("short") == "short"
  long = "x" * 500
  clipped = remote_bot.clip_notice(long)
  assert len(clipped) == remote_bot.NOTICE_MAX_LENGTH
  assert clipped.endswith("...")
  # Contract 8 fails a seat on a blank notice; the bot sends None instead.
  assert remote_bot.clip_notice("") is None
  assert remote_bot.clip_notice("  \n") is None


def test_divergence_notice_keeps_to_the_seat():
  """A fallback's notice drops the replay error; a legal-set one is kept."""
  game = pyspiel.load_game(rbh.GAME)
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  secret = "replay failed at history index 1: action 3 is not legal"

  def reply(decider, notice):
    """A Reply of action 1 by `decider` with `notice`.

    Args:
      decider: the Reply's decider.
      notice: the Reply's notice.

    Returns:
      The Reply.
    """
    return server_lib.Reply(
        action=1,
        probs={1: 1.0},
        value=None,
        decider=decider,
        artifact="a@3",
        notice=notice,
    )

  fallback = remote_bot.seat_notice(
      reply("fallback", f"engine/agent rules differed ({secret})"), s, 0
  )
  assert fallback == remote_bot.DIVERGED_NOTICE.format(
      move=s.action_to_string(0, 1)
  ), fallback
  assert secret not in fallback and "3" not in fallback
  kept = "engine/agent rules differ on this move (move 2; ...)"
  assert remote_bot.seat_notice(reply("network", kept), s, 0) == kept
  assert remote_bot.seat_notice(reply("network", None), s, 0) is None
  assert remote_bot.seat_notice(reply("fallback", None), s, 0) is None


def test_notice_in_process(leduc_dir, monkeypatch):
  """A legal-set mismatch's notice is the reply's notice."""
  monkeypatch.setenv(remote_bot.CHECKPOINT_ENV, leduc_dir)
  served = rbh.mismatch_bot()
  game = pyspiel.load_game(rbh.GAME)
  s = game.new_initial_state()
  s.apply_action(0)
  s.apply_action(3)
  reply = served.step("full_state", rbh.request_of(game, s, 0))
  assert reply["action"] == 1, reply  # call: the only action both allow
  assert reply["notice"].startswith("engine/agent rules differ on this move")
  assert len(reply["notice"]) <= remote_bot.NOTICE_MAX_LENGTH


def test_public_url_is_read_through_the_cache(tmp_path, monkeypatch):
  """A public bucket URL (a local stand-in) serves like the local folder."""
  root = tmp_path / "served"
  local = rbh.write_leduc(root / "bkt" / "runs" / "leduc")
  server = public_bucket.FakeBucketServer(str(root)).start()
  try:
    monkeypatch.setattr(public, "ENDPOINT", server.url)
    monkeypatch.setenv(public.CACHE_ENV, str(tmp_path / "cache"))
    url = f"{server.url}/bkt/runs/leduc"
    path, step = remote_bot.local_artifact(url, None)
    assert step == 3 and path.startswith(str(tmp_path / "cache")), path
    with pytest.raises(ValueError, match="not a public Cloud Storage URL"):
      remote_bot.local_artifact("https://example.com/bkt/runs/leduc", None)
    served = remote_bot.RemoteCheckpointBot(url + "@3", warmup=False)
    game = pyspiel.load_game(rbh.GAME)
    s = game.new_initial_state()
    s.apply_action(0)
    s.apply_action(3)
    reply = served.step("full_state", rbh.request_of(game, s, 0))
    ref = checkpoint_bot_lib.checkpoint_bot(local)
    assert reply["action"] == ref.choose_action(s, 0), reply
  finally:
    server.stop()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
