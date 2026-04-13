import itertools

import jax
import pyspiel
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import samplers, utils

test_games = {
  # === Classic 2-player cubic games ===
  "rps": pyspiel.load_game("matrix_rps"),  # Rock-Paper-Scissors (zero-sum)
  "mp": pyspiel.load_game("matrix_mp"),  # Matching Pennies (zero-sum)
  "pd": pyspiel.load_game("matrix_pd"),  # Prisoner's Dilemma (general-sum)
  "bos": pyspiel.load_game("matrix_bos"),  # Battle of the Sexes (coordination)
  "cd": pyspiel.load_game("matrix_cd"),  # Chicken / Hawk-Dove
  "coordination": pyspiel.load_game(
    "matrix_coordination"
  ),  # Pure coordination game
}


class SamplerTest(parameterized.TestCase):
  # different games and objective
  # m, z_m
  @parameterized.parameters(
    itertools.product(
      test_games.values(),
      (1, 2, 4),
      (1, 4, 16),
    )
  )
  def test_open_spiel_sampler(self, game, m, batch_size):
    sampler = samplers.OpenSpielGameSampler(
      game, samplers.Objective.EPS_MWME, m=m, z_m=None
    )
    batch = sampler.sample_random(batch_size=5, rng=jax.random.key(0))
    self.assertTrue(
      samplers.stack(jax.vmap(samplers.broadcast)(batch)).shape, (batch_size, 4, *sampler.payoff_tensor.shape)
    )

    action_sizes = tuple(
      game.num_distinct_actions() for _ in range(game.num_players())
    )
    batch2 = samplers.Data(
      **utils.dummy_nes_batch(
        batch_size, game.num_players(), action_sizes, jax.random.key(0)
      )
    )
    batch_normalised = sampler.normalise_batch(batch2)
    self.assertTrue(
      samplers.stack(jax.vmap(samplers.broadcast)(batch_normalised)).shape,
      (batch_size, 4, *sampler.payoff_tensor.shape),
    )


if __name__ == "__main__":
  absltest.main()
