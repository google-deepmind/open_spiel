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

"""Tests for SearchPolicy MCTSConfig.chance_value.

"exact" values an afterstate by the
expectation over its outcomes; "sampled" draws one outcome per simulation.
On the toy coin tree the sampled afterstate value is one of the two outcome
values and unbiased over keys (mean -0.4), and equals the exact value where
the action reaches no chance node; on leduc the sampled search's root values
agree with the exact search within a declared tolerance.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies.tests import search_fixtures
from open_spiel.python.scale.policies.tests import toy_games

#: The largest mean |exact - sampled| root value allowed on leduc (64
#: simulations, 4 worlds, 16 roots).
LEDUC_MEAN_TOL = 0.25


def test_default_is_exact_and_recorded():
  """Default is exact and recorded."""
  cfg = search.MCTSConfig()
  assert (
      cfg.chance_value == "exact" and cfg.to_json()["chance_value"] == "exact"
  )
  assert (
      search.MCTSConfig(chance_value="sampled").to_json()["chance_value"]
      == "sampled"
  )
  with pytest.raises(ValueError):
    search.MCTSConfig(chance_value="mean")


def test_sampled_afterstate_value_known_answer_on_the_coin():
  """Sampled afterstate value known answer on the coin."""
  env = toy_games.coin()
  pol = search.SearchPolicy(
      search_fixtures.uniform_policy(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      search_fixtures.cfg(chance_value="sampled"),
  )
  node = search_fixtures.toy_node(env, 1)
  vals = []
  for i in range(400):
    # Reaches private _decision_fn of the code under test.
    out, _ = pol._decision_fn(  # pylint: disable=protected-access
        {}, jax.random.key(i), jnp.asarray([1], jnp.int32), node
    )
    vals.append(float(out.afterstate_value[0]))
    np.testing.assert_allclose(
        np.exp(np.asarray(out.chance_logits[0])), [0.3, 0.7], atol=1e-6
    )
  vals = np.asarray(vals)
  assert set(np.round(vals, 6)) <= {1.0, -1.0}, set(vals)
  assert abs(vals.mean() - (-0.4)) < 0.15, vals.mean()  # 400 draws: SE ~0.046
  # Reaches private _decision_fn of the code under test.
  out0, _ = pol._decision_fn(  # pylint: disable=protected-access
      {}, jax.random.key(0), jnp.asarray([0], jnp.int32), node
  )
  np.testing.assert_allclose(
      np.asarray(out0.afterstate_value), [0.1], atol=1e-6
  )


def test_sampled_matches_exact_on_leduc_within_tolerance():
  """Sampled matches exact on leduc within tolerance."""
  env, exact_pol = search_fixtures.leduc_search(k=4, sims=64)
  sampled = search.SearchPolicy(
      exact_pol.base,
      env,
      exact_pol.sampler,
      search.MCTSConfig(
          **{**exact_pol.cfg.__dict__, "chance_value": "sampled"}
      ),
  )
  params = exact_pol.init_params(jax.random.key(0))
  # Seat 1 to act after a check (round 1) on 16 deals; its call closes the
  # round.
  diffs = []
  for c0, c1 in [(0, 4), (1, 2), (2, 5), (3, 0)]:
    s = search_fixtures.leduc_state(env, (c0, c1), (1,))
    a = search_fixtures.run(exact_pol, env, s, 1, params=params)
    b = search_fixtures.run(sampled, env, s, 1, params=params)
    diffs.append(
        abs(
            float(a.aux[search.AUX_SEARCH_VALUE][0])
            - float(b.aux[search.AUX_SEARCH_VALUE][0])
        )
    )
  print(
      f"leduc: mean |exact - sampled| root value {np.mean(diffs):.4f} (max "
      f"{np.max(diffs):.4f}) over {len(diffs)} roots; "
      f"tolerance {LEDUC_MEAN_TOL}"
  )
  assert np.mean(diffs) < LEDUC_MEAN_TOL, diffs


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
