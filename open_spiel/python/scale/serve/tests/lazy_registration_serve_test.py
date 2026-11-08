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

"""A card move transformer checkpoint serves in a fresh process, no imports.

The hearts and euchre card move transformer artifacts record names (the
encoder hearts_card_planes / euchre_card_planes, the torso kind
card_move_transformer, the head kind card_move_pointer) whose modules
register them at import. One process writes a tiny artifact of each; a
second, fresh interpreter that imports only the serving API loads it and
answers one decision: hearts through PolicyServer.act on a pyspiel state,
euchre through the PolicyServer's artifact and a HostAgent on the env's raw
history (pyspiel's released euchre is another rule set; serving it through
pyspiel needs the corrected engine, euchre_serve_test). The reader checks
that none of the registering modules was loaded before the lookup and that
the lookup loaded them (registration_tables_test is the static half).
"""

import json
import sys
import textwrap

import pytest

from open_spiel.python.scale.launch import subprocess_env

# Both scripts take the package's import name as argv[1] and import through
# it, so they run unchanged wherever the package lives.
WRITE = textwrap.dedent("""
    import importlib
    import sys
    PKG = sys.argv[1]
    enc_registry = importlib.import_module(PKG + ".encoders.registry")
    env_registry = importlib.import_module(PKG + ".envs.registry")
    cmt = importlib.import_module(PKG + ".networks.card_move_transformer")
    fresh_artifact = importlib.import_module(
        PKG + ".serve.tests.fresh_artifact")
    game, encoder, special, path = sys.argv[2:]
    env = env_registry.make(game)
    enc = enc_registry.get(encoder)
    net = cmt.card_move_transformer(
        depth=1, embed_dim_per_head_over8=1, n_head=1, ff_factor=1,
        in_channels=enc.shape[0], n_card=enc.shape[1],
        n_special_action=int(special), compute_dtype="float32")
    fresh_artifact.write(path, env, encoder, step=1, network=net)
""")

READ = textwrap.dedent("""
    import importlib
    import json
    import sys
    PKG = sys.argv[1]
    WATCH = [f"{PKG}.encoders.hearts_card_planes",
             f"{PKG}.encoders.euchre_card_planes",
             f"{PKG}.networks.card_move_transformer"]
    import jax
    import pyspiel
    PolicyServer = importlib.import_module(PKG + ".serve").PolicyServer
    host_agent = importlib.import_module(PKG + ".serve.host_agent")
    game, path = sys.argv[2:]
    before = [m for m in WATCH if m in sys.modules]
    server = PolicyServer()
    if game == "hearts":
      state = pyspiel.load_game("hearts").new_initial_state()
      while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
      reply = server.act(state, checkpoint_dir=path, step=1)
      action, legal = reply.action, state.legal_actions()
    else:
      # Dealer 0, cards 0..19 dealt to seats 0, 1, 2, 3, 0, ..., up-card 20:
      # seat 1 bids first.
      hands = [[0] + list(range(20)) + [20]]
      agent = host_agent.HostAgent(server.artifact(path, 1), 1)
      decision = agent.act(hands, seat_key=jax.random.key(0))
      action = int(decision.action)
      legal = [a for a, p in decision.distribution().items()]
    after = [m for m in WATCH if m in sys.modules]
    print(json.dumps({"before": before, "after": after,
                      "action": action, "legal": legal}))
""")

GAMES = {
    # game: (encoder, special-action tokens, the modules its load must import)
    "hearts": (
        "hearts_card_planes",
        0,
        ("encoders.hearts_card_planes", "networks.card_move_transformer"),
    ),
    "euchre": (
        "euchre_card_planes",
        7,
        ("encoders.euchre_card_planes", "networks.card_move_transformer"),
    ),
}


def _run(args, timeout=600):
  """Runs a child Python on the CPU and returns its stdout.

  Args:
    args: the interpreter's arguments.
    timeout: seconds.

  Returns:
    The child's stdout.
  """
  r = subprocess_env.run_python(
      args, env={"JAX_PLATFORMS": "cpu"}, timeout=timeout
  )
  assert r.returncode == 0, r.stdout + r.stderr
  return r.stdout


@pytest.mark.parametrize("game", sorted(GAMES))
def test_checkpoint_serves_in_a_fresh_process_with_no_import(tmp_path, game):
  encoder, special, loaded = GAMES[game]
  path = str(tmp_path / f"{game}_cmt")
  _run(["-c", WRITE, subprocess_env.PACKAGE, game, encoder, str(special), path])
  out = json.loads(
      _run(["-c", READ, subprocess_env.PACKAGE, game, path])
      .strip()
      .splitlines()[-1]
  )
  assert not out["before"], out
  for m in loaded:
    assert f"{subprocess_env.PACKAGE}.{m}" in out["after"], out
  assert out["action"] in out["legal"], out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
