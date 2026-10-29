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

"""universal_poker is servable.

universal_poker is servable: on random pyspiel playouts of the canonical
HUNL game the replayed env node equals pyspiel's at every prefix (actor,
legal set, terminal; returns at terminal), and PolicyServer serves an
on-disk NetworkPolicy artifact (universal_poker_info_state encoder) over
pyspiel states, each reply equal to the HostAgent decision.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import universal_poker as up
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.testing import optional_pyspiel_games

ENC = "universal_poker_info_state"


def _playouts(game, n, seed):
  """Every prefix state of n random pyspiel playouts.

  Raises are biased small so hands run past the flop.

  Args:
    game: the pyspiel game.
    n: the playouts.
    seed: the numpy seed.

  Returns:
    The prefix states (pyspiel state clones).
  """
  rng = np.random.default_rng(seed)
  out = []
  for _ in range(n):
    s = game.new_initial_state()
    while True:
      out.append(s.clone())
      if s.is_terminal():
        break
      if s.is_chance_node():
        acts, probs = zip(*s.chance_outcomes())
        s.apply_action(
            int(rng.choice(acts, p=np.asarray(probs) / np.sum(probs)))
        )
      else:
        legal = s.legal_actions()
        small = [a for a in legal if a <= 1] or legal
        s.apply_action(int(rng.choice(small if rng.random() < 0.7 else legal)))
  return out


@pytest.fixture(name="game", scope="module")
def _game_fixture():
  """The optional universal_poker game.

  Returns:
    The pyspiel game.
  """
  return optional_pyspiel_games.load_game(up.GAME_STRING)


@pytest.fixture(name="env", scope="module")
def _env_fixture(game):
  """The env serving `game`.

  Args:
    game: The pyspiel game.

  Returns:
    The env.
  """
  return games.env_for("universal_poker", wire.game_params(game))


def test_replayed_node_equals_pyspiel(game, env):
  """Replayed node equals pyspiel."""
  n_checked = n_showdowns = 0
  for state in _playouts(game, 12, seed=1):
    if state.is_chance_node():
      continue
    hands = wire.hands_from_state(state)
    seat = 0 if state.is_terminal() else state.current_player()
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)  # raises on any mismatch
    if state.is_terminal():
      np.testing.assert_array_equal(
          np.asarray(env.returns(rep.state)),
          np.asarray(state.returns(), np.float32),
      )
      n_showdowns += int(rep.state.folded_player < 0)
    n_checked += 1
  assert n_checked > 40 and n_showdowns >= 1


@pytest.fixture(name="poker_ckpt", scope="module")
def _poker_ckpt_fixture(tmp_path_factory, env):
  """A universal_poker network checkpoint.

  Args:
    tmp_path_factory: The pytest fixture.
    env: The env.

  Returns:
    The checkpoint folder.
  """
  d = tmp_path_factory.mktemp("universal_poker_ckpt")
  enc = enc_registry.get(ENC)
  net = {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [32, 32], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  pol = policy_lib.NetworkPolicy(
      env.name, ENC, env.num_actions, env.num_players, ENC, list(enc.shape), net
  )
  params = pol.init_params(jax.random.key(3))
  ckpt.write_artifact(
      str(d),
      5,
      pol,
      params,
      rules_sha=env_registry.rules_sha(env),
      config_hash="c",
      algo="test",
      env_name=env.name,
      default_rule=types.SAMPLE,
      run_id="424",
      env_spec=env_registry.spec_of(env).to_json(),
  )
  return str(d)


def test_policy_server_serves_an_artifact_over_pyspiel_states(game, poker_ckpt):
  """Policy server serves an artifact over pyspiel states."""
  server = server_lib.PolicyServer()
  states = [
      s
      for s in _playouts(game, 3, seed=2)
      if not s.is_chance_node() and not s.is_terminal()
  ]
  assert len(states) >= 6
  art = ckpt.load_artifact(poker_ckpt, 5)
  for state in states[:8]:
    reply = server.act(state, checkpoint_dir=poker_ckpt, step=5, seed=7)
    assert reply.action in state.legal_actions()
    assert set(reply.probs) == set(state.legal_actions())
    np.testing.assert_allclose(sum(reply.probs.values()), 1.0, atol=1e-12)
    assert reply.artifact == "424@5"
    seat = state.current_player()
    d = host_agent.HostAgent(art, seat).act(
        wire.hands_from_state(state), seat_key=server_lib.seat_key(7, seat)
    )
    assert d.action == reply.action
    assert reply.decider == ("forced" if len(reply.probs) == 1 else "network")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
