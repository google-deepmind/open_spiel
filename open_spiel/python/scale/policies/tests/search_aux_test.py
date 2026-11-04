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

"""Tests for the search-operator aux and vocabulary.

The operator aux keys are registered with their dtype and shape and the
"search_" prefix is reserved; the Arena checks registered aux statically and
refuses non-finite values; the operator vocabularies accept exactly their
values; and a batch has one target_value writer (the Arena when a teacher emits
AUX_TARGET_VALUE, else a sources.targets builder).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import auxiliary as aux
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources import targets
from open_spiel.python.scale.sources.tests import helpers

tree_map = jax.tree_util.tree_map


def test_operator_aux_keys_are_registered_and_the_prefix_reserved():
  """Operator aux keys are registered and the prefix reserved."""
  assert operator_vocab.AUX_SEARCH_VALUE == "search_value"
  assert operator_vocab.AUX_SEARCH_Q == "search_q"
  assert operator_vocab.AUX_SEARCH_Q_SE == "search_q_se"
  assert operator_vocab.AUX_SEARCH_DECIDED == "search_decided"
  assert operator_vocab.AUX_SPECS[operator_vocab.AUX_SEARCH_VALUE] == (
      "float32",
      (),
  )
  assert operator_vocab.AUX_SPECS[operator_vocab.AUX_SEARCH_Q] == (
      "float32",
      ("A",),
  )
  assert operator_vocab.AUX_SPECS[operator_vocab.AUX_SEARCH_Q_SE] == (
      "float32",
      ("A",),
  )
  assert operator_vocab.AUX_SPECS[operator_vocab.AUX_SEARCH_DECIDED] == (
      "bool",
      (),
  )
  assert operator_vocab.AUX_SPECS[types.AUX_TARGET_POLICY] == (
      "float32",
      ("A",),
  )
  assert all(
      operator_vocab.is_reserved_aux(k)
      for k in operator_vocab.AUX_SPECS
      if k.startswith("search_")
  )
  assert not operator_vocab.is_reserved_aux(
      "opp_hand"
  ) and operator_vocab.is_reserved_aux("search_anything")
  for k, (dtype, shape) in operator_vocab.AUX_SPECS.items():
    assert aux.spec(k).dtype == dtype and aux.spec(k).shape == shape
  assert aux.is_reserved("search_anything") and not aux.is_reserved("opp_hand")
  with pytest.raises(ValueError, match="already registered"):
    aux.register_aux_key(
        operator_vocab.AUX_SEARCH_VALUE, ("A",), "float32", writer="x"
    )
  with pytest.raises(KeyError, match="registered"):
    aux.spec("no_such_aux_key")
  # re-exported, one definition
  assert search.AUX_SEARCH_VALUE is operator_vocab.AUX_SEARCH_VALUE
  assert search.AUX_SEARCH_Q is operator_vocab.AUX_SEARCH_Q


def test_operator_vocabularies_accept_exactly_their_values():
  """Operator vocabularies accept exactly their values."""
  assert "topk_pruning+rollout" in operator_vocab.PRIOR_DEPENDENCE
  assert "rollout" in operator_vocab.PRIOR_DEPENDENCE["topk_pruning+rollout"]
  for v in (
      "puct_prior+value",
      "mmd_anchor+rollout",
      "rollout_only",
      "topk_pruning",
      "topk_pruning+rollout",
      "tie_break_only",
      "tie_break_only+sampler_head",
      "topk_pruning+rollout+sampler_head",
  ):
    assert operator_vocab.check_prior_dependence(v) == v
  for bad in ("rollout", "puct", "topk_pruning+sampler", "+sampler_head", ""):
    with pytest.raises(ValueError):
      operator_vocab.check_prior_dependence(bad)
  assert operator_vocab.VALUE_BIASES == ("unbiased", "fused_upper_bound")
  assert types.DECIDERS == ("network", "search", "scripted", "forced")
  assert types.POLICY_DECIDERS == types.DECIDERS[:3]


def test_check_reserved_aux_checks_dtype_and_shape_at_trace_time():
  """Check reserved aux checks dtype and shape at trace time."""
  ok = {
      operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((4,), jnp.float32),
      operator_vocab.AUX_SEARCH_Q: jnp.zeros((4, 3), jnp.float32),
      operator_vocab.AUX_SEARCH_DECIDED: jnp.ones((4,), jnp.bool_),
      "free/key": jnp.zeros((4, 7)),
  }
  arena_lib.check_reserved_aux(ok, 4, 3)
  with pytest.raises(TypeError, match="search_value"):
    arena_lib.check_reserved_aux(
        {operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((4,), jnp.float64)}, 4, 3
    )
  with pytest.raises(ValueError, match="search_q"):
    arena_lib.check_reserved_aux(
        {operator_vocab.AUX_SEARCH_Q: jnp.zeros((4, 2), jnp.float32)}, 4, 3
    )
  with pytest.raises(ValueError, match="reserved"):
    arena_lib.check_reserved_aux(
        {"search_undeclared": jnp.zeros((4,), jnp.float32)}, 4, 3
    )
  fin = arena_lib.reserved_aux_finite(
      {
          operator_vocab.AUX_SEARCH_Q: jnp.asarray(
              [[0, 1, 2], [0, jnp.nan, 0]], jnp.float32
          ),
          operator_vocab.AUX_SEARCH_DECIDED: jnp.asarray([True, False]),
      },
      2,
  )
  np.testing.assert_array_equal(np.asarray(fin), [True, False])


class _AuxPolicy(tabular.TabularPolicy):
  """A kuhn tabular policy that also emits the given aux (test probe).

  Attributes:
    aux_fn: the probe's aux function.
  """

  def __init__(self, env, aux_fn):
    super().__init__(
        "kuhn_poker",
        kuhn.NUM_INFO_STATES,
        env.num_actions,
        env.num_players,
        helpers.KUHN_INDEX,
    )
    self.aux_fn = aux_fn

  def evaluate(self, params, carry, step, key):
    """The tabular output with the probe's aux added.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.

    Returns:
      The PolicyOutput.
    """
    out = super().evaluate(params, carry, step, key)
    b = out.logits.shape[0]
    return out._replace(aux=dict(out.aux, **self.aux_fn(b)))


def _roll(env, pol, steps=6, batch=4):
  """A short self-play rollout of `pol`.

  Args:
    env: The env.
    pol: The policy.
    steps: The plies.
    batch: The rows.

  Returns:
    The rollout's output.
  """
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  params = [tree_map(lambda x: x[None], pol.init_params(jax.random.key(0)))]
  table = arena_lib.self_play_table(batch, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(1), batch, 2)
  st = ar.reset(params, table, dk, pk)
  return ar.rollout(params, table, st, steps)


def test_arena_refuses_a_non_finite_registered_aux_value():
  """Arena refuses a non finite registered aux value."""
  env = kuhn.KuhnEnv()
  fine = _AuxPolicy(
      env,
      lambda b: {operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((b,), jnp.float32)},
  )
  _roll(env, fine)
  nan = _AuxPolicy(
      env,
      lambda b: {
          operator_vocab.AUX_SEARCH_VALUE: jnp.full((b,), jnp.nan, jnp.float32)
      },
  )
  with pytest.raises(arena_lib.BadPlyError, match="registered aux value"):
    _roll(env, nan)
  wrong = _AuxPolicy(
      env,
      lambda b: {
          operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((b, 1), jnp.float32)
      },
  )
  with pytest.raises(ValueError, match="search_value"):
    _roll(env, wrong)


def test_arena_copying_aux_target_value_makes_it_the_value_target_writer():
  """Arena copying aux target value makes it the value target writer.

  A teacher's AUX_TARGET_VALUE copied by the Arena sets
  labels[TARGET_VALUE_VALID] (the plies the slot acted), so a later
  with_mc_value_targets refuses instead of overwriting.
  """
  env = kuhn.KuhnEnv()
  teacher = helpers.kuhn_tabular(env)
  # emit_value default True
  mix = dagger.DaggerPolicy(helpers.kuhn_tabular(env), teacher)
  _, batch, _ = _roll(env, mix)
  assert types.TARGET_VALUE_VALID in batch.labels
  np.testing.assert_array_equal(
      np.asarray(batch.labels[types.TARGET_VALUE_VALID]),
      np.asarray(batch.valid),
  )
  with pytest.raises(ValueError, match="written once"):
    targets.with_mc_value_targets(batch, gamma=1.0)
  # Without AUX_TARGET_VALUE the Arena writes nothing: the MC builder is the
  # writer.
  quiet = dagger.DaggerPolicy(
      helpers.kuhn_tabular(env), teacher, emit_value=False
  )
  _, b2, _ = _roll(env, quiet)
  assert types.TARGET_VALUE_VALID not in b2.labels
  out = targets.with_mc_value_targets(b2, gamma=1.0)
  assert types.TARGET_VALUE_VALID in out.labels


def test_dagger_sources_write_value_targets_once():
  """Dagger sources write value targets once.

  DaggerSource "teacher": the Arena's label; "return": the MC builder's
  (the teacher's value is not emitted, so nothing is overwritten).
  """
  env = kuhn.KuhnEnv()
  teacher, student = helpers.kuhn_tabular(env), helpers.kuhn_tabular(env)
  tp = teacher.params_from_logits(helpers.constant_logits(1))
  sp = student.init_params()
  table = arena_lib.self_play_table(8, 2)
  for vt in ("teacher", "return"):
    src = dagger.DaggerSource(
        env,
        student,
        teacher,
        student_params=sp,
        teacher_params=tp,
        table=table,
        num_steps=8,
        beta=schedules.constant(1.0),
        value_target=vt,
        return_lookahead=False,
    )
    b = src.next(jax.random.key(2))
    lab = np.asarray(b.labels[types.TARGET_VALUE_VALID])
    if vt == "teacher":
      np.testing.assert_array_equal(lab, np.asarray(b.valid))
    else:
      assert not (lab & ~np.asarray(b.valid)).any()


def test_env_labels_may_not_carry_the_writer_label():
  """Env labels may not carry the writer label."""

  class Labelled(kuhn.KuhnEnv):
    """A kuhn env whose labels claim the writer's label."""

    def labels(self, s, seat):
      """A TARGET_VALUE_VALID label, which only the writer may set.

      Args:
        s: the state.
        seat: the observing seat.

      Returns:
        The labels.
      """
      del s, seat  # Unused.
      return {types.TARGET_VALUE_VALID: jnp.bool_(True)}

  env = Labelled()
  with pytest.raises(ValueError, match="target_value_valid"):
    _roll(env, helpers.kuhn_tabular(env))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
