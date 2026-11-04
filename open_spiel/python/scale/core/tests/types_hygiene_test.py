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

"""The contract modules load no array framework; their records validate.

core.types, core.auxiliary, core.frameworks and policies.operator_vocab load no
array framework, and the contract records validate their fields.
"""

import sys

import pytest

from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import types
from open_spiel.python.scale.launch import subprocess_env

_PKG = subprocess_env.PACKAGE
_PROBE = r"""
import sys
import PKG
import PKG.core
import PKG.core.types as t
import PKG.core.auxiliary
import PKG.core.frameworks
import PKG.policies.operator_vocab
bad = sorted(m for m in sys.modules
             if m.split('.')[0] in ('jax', 'jaxlib', 'flax', 'numpy', 'optax'))
assert not bad, bad
assert t.Capability.HOST_DRIVEN and t.SeatStep._fields == (
    'info', 'legal_mask', 'seat', 'to_move', 'new_hand', 'new_match',
    'last_action', 'last_actor')
""".replace("PKG", _PKG)


def test_types_imports_no_framework():
  """The contract modules import no array framework."""
  r = subprocess_env.run_python(["-c", _PROBE], timeout=120)
  assert r.returncode == 0, r.stderr


def test_sentinels_and_rules():
  """The sentinels hold their values; DecisionRule and PolicySpec validate.

  Raises:
    AssertionError: DecisionRule or PolicySpec accepted a bad field.
  """
  assert types.NO_ACTOR == -1 and types.NO_ACTION == -1
  assert types.TrajectoryBatch._fields == (
      "steps",
      "init_carry",
      "action",
      "behavior_logits",
      "behavior_logp",
      "value_pred",
      "target_policy",
      "target_value",
      "reward",
      "done",
      "valid",
      "labels",
      "aux",
      "bootstrap_step",
  )
  for bad in (
      dict(kind="greedy"),
      dict(temperature=0.0),
      dict(temperature=-1.0),
      dict(temperature=float("inf")),
      dict(temperature=float("nan")),
  ):
    try:
      types.DecisionRule(**bad)
    except AssertionError:
      continue
    raise AssertionError(f"DecisionRule accepted {bad}")
  spec = types.PolicySpec(
      "g",
      "o",
      2,
      2,
      False,
      types.Capability.NATIVE_JAX,
      "jax",
      "arch",
      {"k": 1},
  )
  assert spec.arch_kwargs == {"k": 1}
  for bad in (
      ("g", "o", 2, 2, False, "native_jax", "jax", "arch"),
      ("g", "o", 2, 2, False, types.Capability.NATIVE_JAX, "jax", ""),
      (
          "g",
          "o",
          2,
          2,
          False,
          types.Capability.NATIVE_JAX,
          "jax",
          "a",
          {"f": object()},
      ),
  ):
    try:
      types.PolicySpec(*bad)
    except (AssertionError, TypeError):
      continue
    raise AssertionError(f"PolicySpec accepted {bad}")


def test_policy_spec_decider_and_publishable():
  """PolicySpec validates decider and publishable.

  Raises:
    AssertionError: PolicySpec accepted a bad decider or publishable.
  """
  spec = types.PolicySpec(
      "g", "o", 2, 2, False, types.Capability.NATIVE_JAX, "jax", "arch"
  )
  assert spec.decider == "network" and spec.publishable
  search = types.PolicySpec(
      "g",
      "o",
      2,
      2,
      False,
      types.Capability.NATIVE_JAX,
      "jax",
      "s",
      decider="search",
      publishable=False,
  )
  assert search.decider == "search" and not search.publishable
  for bad in (
      dict(decider="oracle"),
      dict(decider="forced"),
      dict(publishable=1),
      dict(framework=""),
  ):
    kw = dict(
        game="g",
        obs_format="o",
        num_actions=2,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name="a",
    )
    kw.update(bad)
    try:
      types.PolicySpec(**kw)
    except AssertionError:
      continue
    raise AssertionError(f"PolicySpec accepted {bad}")


def test_row_masked_policy_is_recognised_by_its_method():
  """RowMaskedPolicy is recognised by its evaluate_rows method."""

  class Masked:
    """A class with evaluate_rows only."""

    def evaluate_rows(self, params, carry, step, key, rows):
      """Does nothing.

      Args:
        params: unused.
        carry: unused.
        step: unused.
        key: unused.
        rows: unused.
      """
      del params, carry, step, key, rows

  assert isinstance(Masked(), types.RowMaskedPolicy)
  assert not isinstance(object(), types.RowMaskedPolicy)


def test_frameworks_registry():
  """The frameworks registry adds identifiers once and refuses others.

  Raises:
    AssertionError: register_framework accepted a non-identifier.
  """
  assert frameworks.names()[:2] == ("jax", "numpy")
  frameworks.register_framework("acme_fw")
  frameworks.register_framework("acme_fw")
  assert frameworks.names().count("acme_fw") == 1
  try:
    frameworks.register_framework("not a name")
  except ValueError:
    return
  raise AssertionError("register_framework accepted a non-identifier")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
