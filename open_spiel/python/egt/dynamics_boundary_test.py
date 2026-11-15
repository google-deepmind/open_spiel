# Copyright 2019 DeepMind Technologies Limited
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

"""Boltzmann dynamics have a finite continuous extension to simplex faces."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import integrate
from scipy import special
from open_spiel.python.egt import dynamics


def independent_reference(state, fitness, temperature):
  state = np.asarray(state)
  weighted_logs = special.xlogy(state, state)
  exploitation = state * (fitness - state @ fitness) / temperature
  return exploitation - weighted_logs + state * weighted_logs.sum()


class BoltzmannBoundaryTest(parameterized.TestCase):

  @parameterized.product(
      state=[
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.25, 0.75, 0.0],
          [0.0, 0.3, 0.7],
      ],
      temperature=[0.25, 1.0, 2.0],
  )
  def test_boundary_derivatives_match_continuous_entropy_extension(
      self, state, temperature
  ):
    state = np.array(state)
    before = state.copy()
    fitness = np.array([2.0, -1.0, 0.5])
    with np.errstate(divide='raise', invalid='raise'):
      actual = dynamics.boltzmannq(state, fitness, temperature)
    expected = independent_reference(state, fitness, temperature)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    self.assertTrue(np.isfinite(actual).all())
    self.assertAlmostEqual(actual.sum(), 0.0)
    np.testing.assert_array_equal(
        actual[state == 0], np.zeros(np.count_nonzero(state == 0))
    )
    np.testing.assert_array_equal(state, before)

  def test_boundary_values_are_limits_of_interior_derivatives(self):
    state = np.array([0.3, 0.7, 0.0])
    fitness = np.array([1.0, -2.0, 4.0])
    interior = (1.0 - 1e-12) * state + np.full(3, 1e-12 / 3)
    np.testing.assert_allclose(
        dynamics.boltzmannq(state, fitness),
        dynamics.boltzmannq(interior, fitness),
        atol=2e-10,
        rtol=0,
    )

  def test_public_population_dynamics_integrate_a_boundary_initial_state(self):
    payoff = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    dynamics_fn = dynamics.SinglePopulationDynamics(
        np.stack([payoff, payoff.T]), dynamics.boltzmannq
    )
    initial = np.array([0.4, 0.6, 0.0])
    # Fail before integration when an invalid derivative would stall the solver.
    self.assertTrue(np.isfinite(dynamics_fn(initial)).all())
    actual = integrate.solve_ivp(
        lambda t, y: dynamics_fn(y, t),
        (0.0, 0.25),
        initial,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )
    expected = integrate.solve_ivp(
        lambda t, y: independent_reference(y, payoff @ y, 1.0),
        (0.0, 0.25),
        initial,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )
    self.assertTrue(actual.success)
    np.testing.assert_allclose(
        actual.y[:, -1], expected.y[:, -1], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(actual.y.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_array_equal(actual.y[2], np.zeros(actual.t.size))

  def test_interior_formula_and_integer_vertices_remain_valid(self):
    rng = np.random.default_rng(21)
    for _ in range(20):
      state = rng.dirichlet(np.ones(4))
      fitness = rng.normal(size=4)
      old = dynamics.replicator(state, fitness) / 0.7 - state * (
          np.log(state) - state.dot(np.log(state))
      )
      np.testing.assert_allclose(
          dynamics.boltzmannq(state, fitness, 0.7), old, rtol=1e-12, atol=1e-15
      )
    np.testing.assert_array_equal(
        dynamics.boltzmannq(np.array([1, 0]), np.array([2.0, 3.0])), [0.0, 0.0]
    )


if __name__ == '__main__':
  absltest.main()
