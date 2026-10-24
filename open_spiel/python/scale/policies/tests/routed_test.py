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

"""Tests for RoutedPolicy.

Per-row routing between two policies, and the scripted-phase
helpers.
"""

import gc
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.policies.tests import named_fns
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import steps as steps_lib

KUHN_INDEX = f"{kuhn.__name__}:info_index"
ROUTE = named_fns.FACING_BET
PASS, BET = 0, 1


def _tab():
  """A Kuhn tabular policy.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)


def _always(pol, a):
  """Tabular params that play `a` everywhere.

  Args:
    pol: The tabular policy.
    a: The action.

  Returns:
    The params.
  """
  return pol.params_from_logits(
      jnp.zeros((12, 2), jnp.float32).at[:, a].set(10.0)
  )


def test_routes_per_row_known_answer():
  """Routes per row known answer.

  Base = always-bet, override = always-pass when facing a bet: the routed
  policy bets first and folds to a bet.
  """
  env = kuhn.KuhnEnv()
  tab = _tab()
  pol = routed.RoutedPolicy(tab, tab, ROUTE)
  params = pol.make_params(_always(tab, BET), _always(tab, PASS))
  s = env.deal(0, 2)
  steps = []
  for hist in ([], [PASS], [BET], [PASS, BET]):
    st = s
    for a in hist:
      st = env.apply(st, a)
    seat = int(env.actor(st))
    steps.append((st, seat))
  states = jax.tree_util.tree_map(
      lambda *x: jnp.stack(x), *[x[0] for x in steps]
  )
  seats = jnp.array([x[1] for x in steps], jnp.int32)
  step = plies.seat_step(env, states, seats, True, True, -1, -1)
  out = pol.evaluate(params, pol.init_carry(params, 4, seats, None), step, None)
  assert out.logits.dtype == jnp.float32 and out.value.dtype == jnp.float32
  np.testing.assert_array_equal(
      np.argmax(np.asarray(out.logits), 1), [BET, BET, PASS, PASS]
  )


def test_spec_and_registry_round_trip():
  """Spec and registry round trip."""
  tab = _tab()
  u = uniform.UniformPolicy("kuhn_poker", 2, 2)
  pol = routed.RoutedPolicy(tab, u, ROUTE)
  assert pol.spec.arch_name == routed.ARCH
  assert pol.spec.capability == types.Capability.NATIVE_JAX
  assert not pol.spec.stateful
  rebuilt = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert rebuilt.spec == pol.spec
  # Capability is the weaker of the two; statefulness is either's.
  cb = fixtures.CallbackTabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)
  h = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  assert (
      routed.RoutedPolicy(cb, h, ROUTE).spec.capability
      == types.Capability.CALLBACK_BATCHED
  )
  assert routed.RoutedPolicy(cb, h, ROUTE).spec.stateful


def test_passes_check_policy_with_a_stateful_part():
  """Passes check policy with a stateful part."""
  env = kuhn.KuhnEnv()
  h = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  tab = _tab()
  pol = routed.RoutedPolicy(h, tab, ROUTE)
  pa = pol.init_params(jax.random.key(0))
  pb = pol.make_params(
      h.make_params(b=jnp.ones(2)), tab.params_from_logits(jnp.ones((12, 2)))
  )
  step1, step2 = steps_lib.two_ply_steps(env, 8, 1)
  check_policy.check_policy(pol, pa, pb, step1, step2)


def test_plays_in_the_arena_like_its_parts():
  """Plays in the arena like its parts.

  Routed(always-bet, always-pass when facing a bet) vs always-bet: as p0 it
  bets and is called (showdown +-2); as p1 it folds to the bet (-1).
  """
  env = kuhn.KuhnEnv()
  tab = _tab()
  pol = routed.RoutedPolicy(tab, tab, ROUTE)
  rp = pol.make_params(_always(tab, BET), _always(tab, PASS))

  def stack(p):
    """Adds a leading member axis of 1.

    Args:
      p: The params.

    Returns:
      The stacked params.
    """
    return jax.tree_util.tree_map(lambda x: x[None], p)

  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.ARGMAX),
          arena_lib.Slot(tab, 1, types.ARGMAX),
      ],
  )
  table = arena_lib.rotated_table(64, [0, 1], [0, 0], 0)
  dk, pk = arena_lib.duplicate_keys(jax.random.key(0), 64, 2)
  _, batch, _ = ar.collect(
      [stack(rp), stack(_always(tab, BET))], table, dk, pk, 4
  )
  rew = np.asarray(batch.reward)[np.asarray(batch.done)]
  seat = np.asarray(batch.steps.seat)[:, 0]
  first = np.asarray(batch.reward)[
      np.arange(128), np.argmax(np.asarray(batch.done), 1)
  ]
  assert set(np.abs(first[seat == 0]).tolist()) == {2.0}
  np.testing.assert_array_equal(first[seat == 1], -1.0)
  assert rew.size >= 128


class _HostDriven:
  """Only a spec: a HOST_DRIVEN policy RoutedPolicy must refuse.

  Attributes:
    spec: the PolicySpec.
  """

  def __init__(self):
    self.spec = types.PolicySpec(
        game="kuhn_poker",
        obs_format="none",
        num_actions=2,
        num_players=2,
        stateful=False,
        capability=types.Capability.HOST_DRIVEN,
        framework="jax",
        arch_name="testing.host_driven_spec",
        arch_kwargs={},
    )


def test_rejects_mismatch_and_host_driven():
  """Rejects mismatch and host driven."""
  tab = _tab()
  with pytest.raises(ValueError, match="num_actions"):
    routed.RoutedPolicy(tab, uniform.UniformPolicy("kuhn_poker", 3, 2), ROUTE)
  with pytest.raises(NotImplementedError):
    routed.RoutedPolicy(tab, _HostDriven(), ROUTE)


def test_to_places_every_part_that_can_be_placed():
  """To places every part that can be placed.

  RoutedPolicy.to(device) moves each part that has .to (a TorchNet) and
  returns itself; a routed policy with no placeable part refuses.
  """
  moved = []
  base = _tab()
  base.to = lambda device: moved.append(("base", device)) or base
  pol = routed.RoutedPolicy(base, _tab(), ROUTE)
  assert pol.to("cpu") is pol and moved == [("base", "cpu")]
  with pytest.raises(TypeError, match="no part"):
    routed.RoutedPolicy(_tab(), _tab(), ROUTE).to("cpu")


# -- scripted phases -----------------------------------------------------------


class _Script(types.Policy):
  """A scripted phase: logits 1 on action 0; declares its phase_route.

  Attributes:
    spec: the PolicySpec.
  """

  phase_route = named_fns.NEVER

  def __init__(self):
    self.spec = types.PolicySpec(
        game="x",
        obs_format="none",
        num_actions=2,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name="testing.script_phase",
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key  # Unused.
    return {}

  def encode(self, info):
    """The InfoState itself.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      The InfoState, unchanged.
    """
    return info

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key  # Unused.
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step  # Unused.
    return carry

  def evaluate(self, params, carry, step, key):
    """Logits 1 on action 0.

    Args:
      params: the params.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput.
    """
    del params, carry, key  # Unused.
    b = step.legal_mask.shape[0]
    return types.PolicyOutput(
        logits=jnp.zeros((b, 2), jnp.float32).at[:, 0].set(1.0),
        value=jnp.zeros((b, 2), jnp.float32),
        aux={},
    )


def test_with_scripted_phase_routes_once_and_unwraps():
  """With scripted phase routes once and unwraps."""
  base = uniform.UniformPolicy("x", 2, 2)
  with registry.temporary("testing.script_phase", _Script):
    assert routed.with_scripted_phase(base, None) is base
    pol = routed.with_scripted_phase(base, "testing.script_phase")
    assert isinstance(pol, routed.RoutedPolicy)
    assert pol.spec.arch_kwargs["route"] == _Script.phase_route
    # already routed
    assert routed.with_scripted_phase(pol, "testing.script_phase") is pol
    p, params = routed.routed_pair(base, {}, "testing.script_phase")
    assert isinstance(p, routed.RoutedPolicy) and params == {
        "base": {},
        "override": {},
    }
    assert routed.unwrap_routed(p, params) == (p.base, {})
    assert routed.unwrap_routed(base, {}) == (base, {})


def test_the_gate_cache_frees_a_part():
  """The gate cache frees a part."""
  part = uniform.UniformPolicy("kuhn_poker", 2, 2)
  key = (id(part), True)
  # Reaches private _gated of the code under test.
  f = routed._gated(part)  # pylint: disable=protected-access
  # Reaches private _GATED, _gated of the code under test.
  assert routed._gated(part) is f and key in routed._GATED  # pylint: disable=protected-access
  del part, f
  gc.collect()
  # Reaches private _GATED of the code under test.
  assert key not in routed._GATED  # pylint: disable=protected-access


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
