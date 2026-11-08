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

"""Padding rows reach a RowMaskedPolicy marked as padding.

The Arena evaluates a CALLBACK_BATCHED slot in one chunk of every row: the
rows the slot owns first, then copies of an owned row as padding. A policy
that implements types.RowMaskedPolicy receives the chunk through
evaluate_rows with the padding marked, so its host side counts only real
rows and the real rows' outputs equal an unpadded evaluation. Without the
gate (the control) the host would count the padding as real rows.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core.tests import numpy_policies
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import uniform

tree_map = jax.tree_util.tree_map
B = 8
OWNED = (
    1,
    4,
    6,
)  # rows whose actor (seat 0, Kuhn's first mover) sits in the recorder's slot


def _table():
  """A table where slot 0 owns seat 0 on the OWNED rows.

  Returns:
    The SlotTable.
  """
  seat_slot = np.array(
      [[0, 1] if b in OWNED else [1, 0] for b in range(B)], np.int32
  )
  persp = np.argmin(seat_slot, axis=1).astype(np.int32)
  return arena_lib.SlotTable(seat_slot, np.zeros((B, 2), np.int32), persp)


def _evaluate_slot0(policy):
  """One Arena evaluation of slot 0 at the first ply.

  Args:
    policy: the slot-0 policy.

  Returns:
    (SlotEval, the owned rows' actor steps).
  """
  env = kuhn.KuhnEnv()
  unif = uniform.UniformPolicy("kuhn_poker", 2, 2)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(policy, 1, types.SAMPLE),
          arena_lib.Slot(unif, 1, types.SAMPLE),
      ],
  )
  table = _table()
  params = [tree_map(lambda x: jnp.asarray(x)[None], policy.init_params()), {}]
  dk, pk = arena_lib.independent_keys(jax.random.key(3), B, 2)
  st = ar.reset(params, table, dk, pk)
  # Reaches private _steps of the code under test.
  steps = ar._steps(st)  # pylint: disable=protected-access
  # Reaches private _keys of the code under test.
  ek, dkk = ar._keys(st, steps)  # pylint: disable=protected-access
  # Reaches private _actor_view of the code under test.
  _, mine, actor_step, _ = ar._actor_view(0, table, steps)  # pylint: disable=protected-access
  assert np.flatnonzero(np.asarray(mine)).tolist() == list(OWNED)
  # Reaches private _evaluate_slot of the code under test.
  se = ar._evaluate_slot(  # pylint: disable=protected-access
      0, params[0], table, steps, st.carries[0], ek, dkk, st.hand_ply
  )
  jax.block_until_ready(se)
  return se, tree_map(lambda x: x[np.asarray(OWNED)], actor_step)


def test_padding_rows_are_marked_and_do_not_change_real_rows():
  """Padding rows are marked and do not change the real rows."""
  rec = numpy_policies.NumpyRowRecorder("kuhn_poker", 2, 2)
  assert arena_lib.gates_padding(arena_lib.Slot(rec, 1, types.SAMPLE))
  se, step = _evaluate_slot0(rec)
  assert rec.rows_seen() == B and rec.real_rows_seen() == len(OWNED)
  for seat, _, real in rec.calls:
    assert (seat[real] == 0).all()
  ref = numpy_policies.NumpyRowRecorder("kuhn_poker", 2, 2)
  out = ref.evaluate(ref.init_params(), (), step, None)
  np.testing.assert_array_equal(
      np.asarray(se.logits)[np.asarray(OWNED)], np.asarray(out.logits)
  )


def test_control_without_the_gate_padding_counts_as_real(monkeypatch):
  """Control: without the gate the padding counts as real rows."""
  monkeypatch.setattr(arena_lib, "gates_padding", lambda slot: False)
  rec = numpy_policies.NumpyRowRecorder("kuhn_poker", 2, 2)
  _evaluate_slot0(rec)
  assert rec.real_rows_seen() == B


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
