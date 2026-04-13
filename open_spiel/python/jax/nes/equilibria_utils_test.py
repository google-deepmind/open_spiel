from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

"""Tests for open_spiel.python.jax.nes.equilibria_utils.py"""



RPS = np.asarray(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

UNIFORM = np.zeros((2, 3), dtype=np.float32)
ONE_TWO = np.asarray([[1e9, 0.0, 0.0], [0.0, 1e9, 0.0]], dtype=np.float32)
ONE_ONE = np.asarray(
    [[0.0, -np.inf, -np.inf], [0.0, -np.inf, -np.inf]], dtype=np.float32
)

UNIFORM_JOINT = np.zeros((3, 3), dtype=np.float32)
ROCK_PAPER_SCISSORS_JOINT = np.asarray(
    [[1e9, 0.0, 0.0], [0.0, 1e9, 0.0], [0.0, 0.0, 1e9]], dtype=np.float32
)
ROCK_PAPER_JOINT = np.asarray(
    [[0.0, 1e9, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
)

NO_MASK = np.ones((3, 3), dtype=np.float32)
ONE_ONE_MASK = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
)
ONE_TWO_MASK = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
)


class EquilibriaTest(parameterized.TestCase):

  @parameterized.parameters([
      (RPS, UNIFORM, 0.0),
      (RPS, ONE_TWO, 2.0),
  ])
  def test_nash_approx(self, payoffs, logits, expected):
      pass


if __name__ == "__main__":
  absltest.main()