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

"""Fixtures for the serving tests.

Fixtures for the serving tests: a counting "search" policy and a kuhn
variant with a forced move.
"""

import collections

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.testing import fixtures

# Host-side call counters, bumped by jax.debug.callback at run time (not at
# trace time), so jitted calls are counted per execution.
COUNTS: collections.Counter[str] = collections.Counter()
# Plies (the policy's own never-reset observe counter) at which evaluate ran.
EVALUATED: list[int] = []


def reset_counts() -> None:
  """Clears COUNTS and EVALUATED (after pending callbacks)."""
  jax.effects_barrier()
  COUNTS.clear()
  EVALUATED.clear()


def evaluated_plies() -> list[int]:
  """The plies evaluate ran at, after pending callbacks.

  Returns:
    The plies.
  """
  jax.effects_barrier()
  return list(EVALUATED)


def counts() -> dict[str, int]:
  """The callback counts, after pending callbacks.

  Returns:
    {name: count}.
  """
  jax.effects_barrier()
  return dict(COUNTS)


def _bump(name):
  """A callback counting calls under `name`.

  Args:
    name: The counter name.

  Returns:
    The callback.
  """

  def cb():
    COUNTS[name] += 1

  return cb


def _record_plies(plies):
  EVALUATED.extend(int(p) for p in np.asarray(plies).reshape(-1))


class CountingSearchPolicy(fixtures.HistoryPolicy):
  """HistoryPolicy maths with call counters; `evaluate` stands in for a search.

  HistoryPolicy maths with call counters; `evaluate` stands in for a
  search: it runs `sims` "simulations" of the base forward and averages them.

  decider = "search" (the class attribute HostAgent reads). Counters are
  bumped per executed callback (under vmap, once per row): "evaluate",
  "sim", "observe". The carry adds "ply", observes since the match began
  (never reset by new_hand), and evaluate records it in EVALUATED, so a
  test can name WHICH plies were evaluated.

  Attributes:
    sims: the "simulations" per evaluate.
    spec: the PolicySpec.
  """

  decider = "search"
  ARCH = "testing.counting_search"

  def __init__(self, game, num_actions, num_players, sims=3, stateful=True):
    super().__init__(game, num_actions, num_players, cross_hand=False)
    self.sims = sims
    self.spec = types.PolicySpec(
        game=game,
        obs_format="seatstep",
        num_actions=num_actions,
        num_players=num_players,
        stateful=stateful,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            game=game,
            num_actions=num_actions,
            num_players=num_players,
            sims=sims,
            stateful=stateful,
        ),
    )

  def init_carry(self, params, batch, seat, key):
    """HistoryPolicy's carry plus the ply counter.

    Args:
      params: the params.
      batch: the batch size.
      seat: the seats.
      key: the PRNG key.

    Returns:
      The carry.
    """
    c = super().init_carry(params, batch, seat, key)
    c["ply"] = jnp.zeros((batch,), jnp.int64)
    return c

  def observe(self, params, carry, step):
    """HistoryPolicy's observe, counting it and the plies.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep.

    Returns:
      The new carry.
    """
    jax.debug.callback(_bump("observe"))
    ply = jnp.where(step.new_match, jnp.int64(0), carry["ply"]) + jnp.int64(1)
    base = {k: v for k, v in carry.items() if k != "ply"}
    return {**super().observe(params, base, step), "ply": ply}

  def evaluate(self, params, carry, step, key):
    """The mean of `sims` base forwards, counted and recorded.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep.
      key: the PRNG key.

    Returns:
      The PolicyOutput.
    """
    jax.debug.callback(_bump("evaluate"))
    jax.debug.callback(_record_plies, carry["ply"])
    carry = {k: v for k, v in carry.items() if k != "ply"}
    total = None
    for _ in range(self.sims):
      jax.debug.callback(_bump("sim"))
      out = super().evaluate(params, carry, step, key)
      total = out.logits if total is None else total + out.logits
    return out._replace(
        logits=(total / jnp.float32(self.sims)).astype(jnp.float32)
    )


class ForcedKuhnEnv(kuhn.KuhnEnv):
  """Kuhn where the second player, facing a pass, may only pass.

  A forced move; every other node is kuhn's.
  """

  def legal_mask(self, s):
    """Kuhn's legal mask, pass only when facing a pass.

    Args:
      s: the env state.

    Returns:
      bool[A].
    """
    m = super().legal_mask(s)
    facing_pass = (jnp.asarray(s.num_bets) == 1) & (jnp.asarray(s.bets)[0] == 0)
    forced = jnp.asarray([True, False])
    return jnp.where(facing_pass & ~self.terminal(s), forced & m, m)
