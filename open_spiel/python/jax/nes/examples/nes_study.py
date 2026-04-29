from absl import flags
from absl import logging
from absl.testing import absltest

from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import games

# | # | Test                                 | What It Proves                                   |
# | - | ------------------------------------ | ------------------------------------------------ |
# | 1 | Training convergence on `8×8`        | The unsupervised loss works                      |
# | 2 | In-distribution accuracy (`8×8`)     | NES matches iterative solvers                    |
# | 3 | Zero-shot on `4×4`, `16×16`, `32×32` | Shape-independent generalization                 |
# | 4 | MWME vs. MRE vs. ε variants          | Flexible selection framework                     |
# | 5 | CCE vs. CE duals                     | Both solution concepts are learnable             |
# | 6 | 2×2 canonical games                  | Qualitative correctness & polytope visualization |


class PaperStudy(absltest.TestCase):
  pass

if __name__ == "__main__":
  absltest.main()