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

"""Breakthrough and hex Env trajectories match recorded fingerprints.

Each case plays fixed-seed random games through the Env API and hashes every
legal mask, actor, return, InfoState and the legal mask of the world rebuilt
from each InfoState. The fingerprints pin the Envs' behaviour across changes
to the underlying JAX games.
"""

import hashlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import breakthrough
from open_spiel.python.scale.envs import hex as hex_env

CASES = {
    "breakthrough_8x8": lambda: breakthrough.BreakthroughEnv(rows=8, columns=8),
    "breakthrough_6x5": lambda: breakthrough.BreakthroughEnv(rows=6, columns=5),
    "hex_5": lambda: hex_env.HexEnv(board_size=5),
    "hex_11": lambda: hex_env.HexEnv(board_size=11),
}
NUM_GAMES = 24


def fingerprint(env):
  """A fingerprint of seeded playouts of `env`.

  Args:
    env: The env.

  Returns:
    {"sha256": hex digest of the trajectories, "plies": ply count}.
  """
  apply = jax.jit(env.apply)
  legal = jax.jit(env.legal_mask)
  actor = jax.jit(env.actor)
  term = jax.jit(env.terminal)
  rets = jax.jit(env.returns)
  info = jax.jit(env.info_state)
  world = jax.jit(env.world_from_info)
  h = hashlib.sha256()
  lengths = []
  for g in range(NUM_GAMES):
    rng = np.random.default_rng(g)
    s = env.root()
    n = 0
    while True:
      m = np.asarray(legal(s), np.bool_)
      h.update(m.tobytes())
      h.update(np.asarray(actor(s), np.int32).tobytes())
      h.update(np.asarray(term(s), np.bool_).tobytes())
      h.update(np.asarray(rets(s), np.float32).tobytes())
      for seat in (0, 1):
        i = info(s, jnp.int32(seat))
        for leaf in jax.tree.leaves(i):
          h.update(np.asarray(leaf).astype(np.int64).tobytes())
        w = world(i)
        h.update(np.asarray(legal(w), np.bool_).tobytes())
        h.update(np.asarray(actor(w), np.int32).tobytes())
      if not m.any():
        break
      a = int(rng.choice(np.flatnonzero(m)))
      h.update(np.int64(a).tobytes())
      s = apply(s, jnp.int32(a))
      n += 1
    lengths.append(n)
  return {"sha256": h.hexdigest(), "plies": int(sum(lengths))}


GOLDEN = {
    "breakthrough_6x5": (
        "f62b62d010d743c6994275cb31117e6be2428d15a4c1df65fc7d89342a76dbf1",
        611,
    ),
    "breakthrough_8x8": (
        "b8a71526fb0f2af3ab933a778a0f18d368b47a48e8cac0aa8996665a1728d2aa",
        1440,
    ),
    "hex_11": (
        "5a026192930bdf083e3897c202d099a0ca2c54281ecf772cc040b735830cb4ca",
        2601,
    ),
    "hex_5": (
        "68a67304f576d669f8854de44ed32316b4a9731fe70cc769f5bcc87dc1d0a4e6",
        489,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_trajectories_match_recorded_fingerprint(name):
  """Trajectories match recorded fingerprint.

  Args:
    name: The parametrized case.
  """
  got = fingerprint(CASES[name]())
  assert (got["sha256"], got["plies"]) == GOLDEN[name]


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
