import itertools

import jax
import jax.numpy as jnp
import pyspiel
from absl.testing import absltest, parameterized

from open_spiel.python.jax.nes import samplers
from open_spiel.python.jax.nes import utils
from open_spiel.python.jax.nes import games

"""Tests for open_spiel.python.jax.nes.samplers.py"""


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


test_configs = {
  "open_spiel": [
    samplers.MixedGameConfig(open_spiel_name="matrix_pd"),
    samplers.MixedGameConfig(open_spiel_name="matrix_mp"),
  ],
  "random": [
    samplers.MixedGameConfig(
      game_type=games.Game.EMPIRICAL_DISC_GAME, num_strategies=(3, 3)
    ),
    samplers.MixedGameConfig(
      game_type=games.Game.L2_INVARIANT, num_strategies=(4, 3)
    ),
  ],
  "mixed": [
    samplers.MixedGameConfig(open_spiel_name="matrix_pd"),
    samplers.MixedGameConfig(
      game_type=games.Game.L2_INVARIANT, num_strategies=(4, 3)
    ),
  ],
}

BASE_OBJECTIVE = samplers.Objective.EPS_MWMRE


class SamplerTest(parameterized.TestCase):
  @parameterized.parameters(
    itertools.product(
      test_games.values(),
      (1, 2, 4),
      (1, 4, 16),
    )
  )
  def test_open_spiel_sampler(self, game, m, batch_size):
    sampler = samplers.OpenSpielGameSampler(game, BASE_OBJECTIVE, m=m, z_m=None)
    batch = sampler.sample_random(batch_size=batch_size, rng=jax.random.key(0))
    self.assertTrue(
      samplers.stack(jax.vmap(samplers.broadcast)(batch)).shape,
      (batch_size, 4, *sampler.payoff_tensor.shape),
    )

  @parameterized.parameters(
    itertools.product(
      test_games.values(),
      (1, 2, 4),
      (1, 4, 16),
    )
  )
  def test_normalisation(self, game, m, batch_size):
    sampler = samplers.OpenSpielGameSampler(game, BASE_OBJECTIVE, m=m, z_m=None)
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

  @parameterized.parameters(
    itertools.product(
      (
        games.Game.L1_INVARIANT,
        games.Game.L2_INVARIANT,
        games.Game.LINF_INVARIANT,
      ),
      ((3, 4), (3, 4, 5), (3, 3)),
      (1, 2, 4),
      (1, 4, 16),
    )
  )
  def test_random_games_sampler(self, game, num_strategies, m, batch_size):
    sampler = samplers.RandomGameSampler(
      game, num_strategies, {}, BASE_OBJECTIVE, m=m, z_m=None
    )
    batch = sampler.sample_random(batch_size=batch_size, rng=jax.random.key(0))
    num_players = len(num_strategies)

    self.assertEqual(
      batch.reward.shape, (batch_size, num_players) + num_strategies
    )
    self.assertEqual(batch.mask.shape, (batch_size,) + num_strategies)

  @parameterized.parameters(itertools.product((1, 2, 8), test_configs.values()))
  def test_multi_game_sampler(self, batch_size, configs):
    sampler = samplers.MultiGameSampler(
      game_configs=configs,
      max_actions=8,
      obj=samplers.Objective.EPS_MWMRE,
      m=2,
    )

    batch = sampler.sample_random(batch_size=batch_size, rng=jax.random.key(0))

    # All padded to max
    self.assertEqual(batch.reward.shape, (batch_size, 2, 8, 8))
    self.assertEqual(batch.mask.shape, (batch_size, 8, 8))

    # Variable valid regions
    # TODO: add test, expose the strategies number
    valid_counts = jnp.sum(batch.mask, axis=(1, 2))  # noqa: F841

    # # Sigma valid
    for i in range(batch_size):
      valid = batch.mask[i]
      self.assertTrue(jnp.all(batch.sigma_hat[i][valid] >= 0))
      self.assertTrue(
        jnp.allclose(jnp.sum(batch.sigma_hat[i][valid]), 1.0, atol=1e-4)
      )
      self.assertTrue(jnp.all(batch.sigma_hat[i][~valid] == 0))


if __name__ == "__main__":
  absltest.main()
