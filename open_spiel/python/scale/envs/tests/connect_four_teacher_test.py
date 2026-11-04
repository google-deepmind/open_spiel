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

"""The Connect Four prefer-center teacher (envs/connect_four/teacher.py).

Known answers on hand-built boards; the Policy contract; out-of-domain rows
are NaN.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs.connect_four import teacher
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import steps as steps_lib


def _info(board, seat):
  """A batch-of-one ConnectFourInfo.

  Args:
    board: The board.
    seat: The seat.

  Returns:
    The InfoState.
  """
  return connect_four.ConnectFourInfo(
      seat=jnp.asarray([seat], jnp.int32), board=jnp.asarray([board], jnp.int32)
  )


def _board(cols_full=(), moves=()):
  """A 42-cell board (row 0 at the top).

  Columns in cols_full are filled to the top with alternating discs; `moves`
  drops single discs.

  Args:
    cols_full: The columns to fill to the top.
    moves: The columns to drop one disc in, in order.

  Returns:
    i32[42], -1 empty, else the disc's seat.
  """
  b = -np.ones((6, 7), np.int32)
  disc = 0
  for c in cols_full:
    for r in range(5, -1, -1):
      b[r, c] = disc
      disc = 1 - disc
  for c in moves:
    r = max(i for i in range(6) if b[i, c] == -1)
    b[r, c] = disc
    disc = 1 - disc
  return b.reshape(-1)


@pytest.mark.parametrize(
    "full,want",
    [
        ((), 3),
        ((3,), 2),
        ((3, 2), 4),
        ((3, 2, 4), 1),
        ((3, 2, 4, 1), 5),
        ((3, 2, 4, 1, 5), 0),
        ((3, 2, 4, 1, 5, 0), 6),
    ],
)
def test_known_answers(full, want):
  """Known answers.

  Args:
    full: The parametrized case.
    want: The parametrized case.
  """
  b = _board(full)
  seat = int(np.sum(b == 0) > np.sum(b == 1))
  info = _info(b, seat)
  assert int(teacher.actions(info)[0]) == want
  assert bool(teacher.in_domain(info)[0])
  step = types.SeatStep(
      info=info,
      legal_mask=jnp.asarray([b[:7] == -1]),
      seat=info.seat,
      to_move=info.seat,
      new_hand=jnp.asarray([True]),
      new_match=jnp.asarray([True]),
      last_action=jnp.asarray([-1], jnp.int32),
      last_actor=jnp.asarray([-1], jnp.int32),
  )
  out = teacher.PreferCenterPolicy().evaluate({}, (), step, None)
  lg = np.asarray(out.logits[0])
  assert lg[want] == 0.0 and np.all(np.isneginf(np.delete(lg, want)))


def test_out_of_domain_rows_are_nan():
  """Out of domain rows are nan."""
  b = _board((3, 2, 4, 1, 5, 0, 6))  # full board: nothing legal
  assert not bool(teacher.in_domain(_info(b, 0))[0])
  b2 = _board(moves=(3,))  # seat 1 to move; ask seat 0
  assert not bool(teacher.in_domain(_info(b2, 0))[0])
  assert bool(teacher.in_domain(_info(b2, 1))[0])


def test_contract_and_registry():
  """Contract and registry."""
  env = connect_four.ConnectFourEnv()
  pol = registry.build(teacher.ARCH, {})
  s1, s2 = steps_lib.two_ply_steps(env, batch=4, seat=0)
  check_policy.check_policy(pol, {}, {}, s1, s2)


def _random_states(n_games=300, seed=0):
  """Every ply of random uniform games.

  Args:
    n_games: The games.
    seed: The seed.

  Returns:
    (env, states).
  """
  env = connect_four.ConnectFourEnv()
  pol = uniform.UniformPolicy("connect_four", 7, 2)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  table = arena_lib.self_play_table(n_games, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(seed), n_games, 2)
  st = ar.reset([{}], table, dk, pk)
  states = []
  for _ in range(43):
    states.append(st.env_state)
    st, _, _ = ar.rollout([{}], table, st, 1)
  return env, jax.tree.map(lambda *xs: jnp.concatenate(xs), *states)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
