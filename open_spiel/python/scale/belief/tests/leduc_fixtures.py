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

"""Leduc belief test fixtures: an opponent-model table and hand-built states."""

import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.belief.tests import named_fns
from open_spiel.python.scale.policies import tabular

FOLD, CALL, RAISE = 0, 1, 2

# Opponent model table: index = rank * 2 + facing (a raise already in the
# current round). Probabilities over (fold, call, raise); the env masks fold
# when there is no bet and raise after the round's second raise, and decide
# renormalises over the legal set.
TABLE = {
    (0, 0): (0.2, 0.9, 0.1),
    (0, 1): (0.6, 0.3, 0.1),
    (1, 0): (0.2, 0.5, 0.5),
    (1, 1): (0.3, 0.5, 0.2),
    (2, 0): (0.2, 0.1, 0.9),
    (2, 1): (0.1, 0.2, 0.7),
}


def table_policy():
  """The TABLE opponent model as a tabular policy and its params.

  Returns:
    (tabular policy, its params).
  """
  pol = tabular.TabularPolicy(
      game="leduc_poker",
      num_states=6,
      num_actions=3,
      num_players=2,
      index_fn=named_fns.LEDUC_INDEX_FN,
  )
  logits = np.zeros((6, 3), np.float32)
  for (rank, facing), probs in TABLE.items():
    logits[rank * 2 + facing] = np.log(np.asarray(probs, np.float64)).astype(
        np.float32
    )
  return pol, pol.params_from_logits(logits)


def state(env, cards, round1, pub=None, round2=()):
  """The leduc state after the deal `cards` and the given rounds.

  Args:
    env: the LeducEnv.
    cards: the private cards dealt, in order.
    round1: the round 1 actions.
    pub: the public card (None: round 1 has not ended).
    round2: the round 2 actions.

  Returns:
    The (unbatched) decision state.
  """
  s = env.root()
  for c in cards:
    s = env.apply_chance(s, jnp.int32(c))
  for a in round1:
    s = env.apply(s, jnp.int32(a))
  if pub is not None:
    assert bool(env.is_chance(s))
    s = env.apply_chance(s, jnp.int32(pub))
  for a in round2:
    s = env.apply(s, jnp.int32(a))
  assert not bool(env.is_chance(s))
  return s
