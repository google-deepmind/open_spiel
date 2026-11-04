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

"""checkpoint_bot on real artifacts (a long test).

A tic-tac-toe artifact trained on the PPO example's config (two updates) is
loaded from disk at its latest step and plays full games against itself through
pyspiel: every move is legal and equals PolicyServer.act's; a fixed seed
reproduces a sampling bot's games; a connect_four game is refused. A kuhn
artifact trained by MMD and a gin rummy network artifact keep the seat's
decisions independent of the other seat's private cards. The CPU latency of a
decision after warmup is printed for tic-tac-toe and for a connect_four network
of the same width.
"""

import dataclasses
import statistics
import sys
import time

import jax
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve.tests import checkpoint_bot_helpers as cbh
import pyspiel

UPDATES = 2


def _train(config, path):
  """Trains config `config` for UPDATES tiny updates.

  Its artifact is path@UPDATES.

  Args:
    config: the config name.
    path: the checkpoint dir.

  Returns:
    path.
  """
  cfg = common.with_overrides(
      common.load_config(config), num_updates=UPDATES, num_envs=16, num_steps=8
  )
  runner.run(dataclasses.replace(cfg, keep_every=UPDATES), checkpoint_dir=path)
  assert ckpt.exists(path, UPDATES)
  return path


@pytest.fixture(scope="module", name="ttt_dir")
def _ttt_dir(tmp_path_factory):
  """The PPO tic-tac-toe example's config.

  The run its --checkpoint_dir makes.

  Args:
    tmp_path_factory: pytest's tmp_path_factory.

  Returns:
    The run's checkpoint dir.
  """
  return _train("ppo_tic_tac_toe", str(tmp_path_factory.mktemp("ttt") / "run"))


def _network_artifact(path, game, encoder, widths, seed=0, params=None):
  """Writes a network artifact of `game` at init (or given) params.

  Args:
    path: The checkpoint directory.
    game: The game.
    encoder: The encoder.
    widths: The hidden widths.
    seed: The init seed.
    params: The game parameters, or None.

  Returns:
    The path.
  """
  env = games.env_for(game, params or {})
  shape = list(enc_registry.get(encoder).shape)
  pol = policy_lib.NetworkPolicy(
      **common.network_policy_kwargs(
          game,
          encoder,
          shape,
          env.num_actions,
          env.num_players,
          catalog.mlp(widths, "relu", "shared"),
      )
  )
  params = pol.init_params(jax.random.key(seed))
  ckpt.write_artifact(
      path,
      1,
      pol,
      params,
      rules_sha=env_registry.rules_sha(env),
      env_spec=env_registry.spec_of(env).to_json(),
      config_hash="c",
      algo="test",
      env_name=game,
  )
  return path


def test_ttt_self_play_equals_policy_server(ttt_dir):
  """TTT self play equals policy server."""
  bot = checkpoint_bot_lib.checkpoint_bot(ttt_dir)
  assert (
      bot.step == UPDATES
      and bot.game_string == "tic_tac_toe"
      and bot.num_players == 2
  )
  game = pyspiel.load_game(bot.game_string)
  bot.check_serves(game)
  srv = server_lib.PolicyServer()
  moves = 0

  def choose(state, seat):
    """The bot's move, checked against the PolicyServer's.

    Args:
      state: the pyspiel state.
      seat: the seat to move.

    Returns:
      The action.
    """
    nonlocal moves
    a = bot.choose_action(state, seat)
    ref = srv.act(
        state, checkpoint_dir=ttt_dir, step=UPDATES, seed=0, rule=types.ARGMAX
    )
    assert a == ref.action and bot.last_probs == dict(ref.probs)
    if bot.manifest.default_rule == types.ARGMAX:
      assert a == srv.act(state, checkpoint_dir=ttt_dir, step=UPDATES).action
    moves += 1
    return a

  cbh.self_play(bot, game, deals=range(2), choose=choose)
  assert moves >= 10


def test_ttt_sampling_reproduces(ttt_dir):
  """TTT sampling reproduces."""
  game = pyspiel.load_game("tic_tac_toe")
  a = cbh.self_play(
      checkpoint_bot_lib.checkpoint_bot(ttt_dir, deterministic=False, seed=9),
      game,
      deals=range(4),
  )
  b = cbh.self_play(
      checkpoint_bot_lib.checkpoint_bot(ttt_dir, deterministic=False, seed=9),
      game,
      deals=range(4),
  )
  assert a == b
  others = [
      cbh.self_play(
          checkpoint_bot_lib.checkpoint_bot(
              ttt_dir, deterministic=False, seed=s
          ),
          game,
          deals=range(4),
      )
      for s in (10, 11)
  ]
  assert any(o != a for o in others)


def test_connect_four_refused(ttt_dir):
  """Connect four refused."""
  bot = checkpoint_bot_lib.checkpoint_bot(ttt_dir, step=UPDATES)
  c4 = pyspiel.load_game("connect_four")
  with pytest.raises(
      ValueError, match="plays 'tic_tac_toe', not 'connect_four'"
  ):
    bot.check_serves(c4)
  s = c4.new_initial_state()
  with pytest.raises(ValueError, match="tic_tac_toe"):
    bot.choose_action(s, 0)


def test_trained_kuhn_privacy(tmp_path):
  """Trained kuhn privacy."""
  path = _train("mmd_kuhn_poker", str(tmp_path / "kuhn"))
  for deterministic in (True, False):
    bot = checkpoint_bot_lib.checkpoint_bot(
        path, deterministic=deterministic, seed=2
    )
    assert bot.step == UPDATES
    pairs, _ = cbh.check_privacy(
        bot, pyspiel.load_game("kuhn_poker"), num_playouts=16, seed=4
    )
    assert pairs >= 8, pairs


def test_gin_privacy(tmp_path):
  """Gin privacy."""
  path = _network_artifact(
      str(tmp_path / "gin"), "gin_rummy", "gin_rummy_observation_tensor", [32]
  )
  bot = checkpoint_bot_lib.checkpoint_bot(path, deterministic=False, seed=4)
  bot.check_serves(pyspiel.load_game("gin_rummy"))
  pairs, _ = cbh.check_privacy(
      bot,
      pyspiel.load_game("gin_rummy"),
      num_playouts=1,
      seed=3,
      max_alternatives=4,
      max_decisions=6,
  )
  assert pairs >= 4, pairs


def test_gin_oklahoma_artifact_serves(tmp_path):
  """An Oklahoma gin artifact (the env draws the first player at chance).

  The installed pyspiel loads its game string and the bot serves it.

  Args:
    tmp_path: pytest's tmp_path.
  """
  path = _network_artifact(
      str(tmp_path / "gin_ok"),
      "gin_rummy",
      "gin_rummy_observation_tensor",
      [32],
      params={"oklahoma": True},
  )
  bot = checkpoint_bot_lib.checkpoint_bot(path, seed=1)
  assert bot.game_string == "gin_rummy(oklahoma=true)"
  game = pyspiel.load_game(bot.game_string)
  bot.check_serves(game)
  state = game.new_initial_state()
  rng = np.random.default_rng(0)
  while state.is_chance_node():
    actions, probs = zip(*state.chance_outcomes())
    state.apply_action(
        int(rng.choice(actions, p=np.asarray(probs) / sum(probs)))
    )
  assert state.current_player() == 0
  assert bot.choose_action(state, 0) in state.legal_actions()


def _latency(bot, game, deals):
  """Plays `deals` with the bot on every seat, timing each decision.

  Args:
    bot: The CheckpointBot.
    game: The pyspiel game.
    deals: The deal seeds.

  Returns:
    The decision times, seconds.
  """
  times = []

  def choose(state, seat):
    """The bot's move, timed.

    Args:
      state: the pyspiel state.
      seat: the seat to move.

    Returns:
      The action.
    """
    t0 = time.perf_counter()
    a = bot.choose_action(state, seat)
    times.append(time.perf_counter() - t0)
    return a

  cbh.self_play(bot, game, deals=deals, choose=choose)
  return times


@pytest.mark.parametrize("which", ["tic_tac_toe", "connect_four"])
def test_cpu_latency_after_warmup(which, ttt_dir, tmp_path):
  """Cpu latency after warmup."""
  if which == "tic_tac_toe":
    path = ttt_dir
  else:
    path = _network_artifact(
        str(tmp_path / "c4"),
        "connect_four",
        "connect_four_flat_planes",
        [128, 128],
    )
  t0 = time.perf_counter()
  bot = checkpoint_bot_lib.checkpoint_bot(path)
  game = pyspiel.load_game(bot.game_string)
  bot.check_serves(game)
  load_s = time.perf_counter() - t0
  t0 = time.perf_counter()
  bot.warmup(range(bot.num_players), games_per_seat=2)
  warm_s = time.perf_counter() - t0
  times = _latency(bot, game, deals=range(100, 104))
  ms = sorted(1e3 * t for t in times)

  def q(f):
    """The f-quantile of the latencies.

    Args:
      f: The quantile, in [0, 1].

    Returns:
      Milliseconds.
    """
    return ms[min(len(ms) - 1, int(f * len(ms)))]

  print(
      f"\nLATENCY {which} on {jax.devices()[0].platform}: load {load_s:.2f} s, "
      f"warmup {warm_s:.2f} s; {len(ms)} decisions after warmup: median "
      f"{statistics.median(ms):.1f} ms, p90 {q(0.9):.1f} ms, max {ms[-1]:.1f} "
      "ms"
  )
  assert ms


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
