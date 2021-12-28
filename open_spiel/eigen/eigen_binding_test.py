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

# Lint as python3
"""Test that Python numpy arrays can be passed to C++ Eigen library."""

import time

from absl.testing import absltest
import numpy as np

import pyspiel_eigen_test


class PyEigenTest(absltest.TestCase):

  def test_square_matrix_elements(self):
    x = np.array([[1, 2], [3, 4]]).astype(float)
    expected = np.array([[1, 2], [3, 4]]) ** 2
    actual = pyspiel_eigen_test.square(x)
    np.testing.assert_array_equal(expected, actual)

  def test_transpose_and_square_matrix_elements(self):
    x = np.array([[1, 2], [3, 4]]).astype(float)
    x = x.transpose()
    expected = np.array(
        [[1, 9],
         [4, 16]])
    actual = pyspiel_eigen_test.square(x)
    np.testing.assert_array_equal(expected, actual)

  def test_transpose_then_slice_and_square_matrix_elements(self):
    x = np.array([[1, 2], [3, 4]]).astype(float)
    x = x.transpose()
    expected = np.array([[9], [16]])
    actual = pyspiel_eigen_test.square(x[0:, 1:])
    np.testing.assert_array_equal(expected, actual)

  def test_square_vector_elements(self):
    x = np.array([1, 2, 3]).astype(float)
    expected = np.array([[1], [4], [9]])
    actual = pyspiel_eigen_test.square(x)
    np.testing.assert_array_equal(expected, actual)

  def test_allocate_cxx(self):
    actual = pyspiel_eigen_test.matrix()
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(expected, actual)

  def test_flags_copy_or_reference(self):
    # A test implementing
    # https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#returning-values-to-python
    start = time.time()
    a = pyspiel_eigen_test.BigMatrix()
    print("Alloc: ", time.time() - start)

    start = time.time()
    m = a.get_matrix()
    print("Ref get: ", time.time() - start)
    self.assertTrue(m.flags.writeable)
    self.assertFalse(m.flags.owndata)

    start = time.time()
    v = a.view_matrix()
    print("Ref view: ", time.time() - start)
    self.assertFalse(v.flags.writeable)
    self.assertFalse(v.flags.owndata)

    start = time.time()
    c = a.copy_matrix()
    print("Copy: ", time.time() - start)
    self.assertTrue(c.flags.writeable)
    self.assertTrue(c.flags.owndata)


if __name__ == "__main__":
  absltest.main()
