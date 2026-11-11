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

"""AlphaZero on Connect Four.

SearchPolicy self-play at 256 simulations with PUCT and Dirichlet root noise
(alpha 1.43, about 10 / 7 actions, weight 0.25), temperature 1 for the first
8 plies of a game and 1e-3 after, a 256x4 MLP over the 126-wide encoding,
256 games per iteration and 4 epochs of 1024-position minibatches over a
replay of the last 4 iterations, for 200 iterations.

  python -m open_spiel.python.scale.train --config=az_connect_four
"""

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import az
from open_spiel.python.scale.networks import catalog

ENCODER = "connect_four_flat_planes"
NETWORK = "mlp_256x4"


def az_config() -> az.AZConfig:
  """The AlphaZero settings of this config.

  Returns:
    The az.AZConfig.
  """
  return az.AZConfig(
      env_name="connect_four",
      encoder=ENCODER,
      obs_shape=[126],
      network=az.jsonable(catalog.get(NETWORK)),
      sampler="perfect_information",
      num_worlds=1,
      seed=0,
      num_simulations=256,
      dirichlet_alpha=1.43,
      dirichlet_epsilon=0.25,
      temperature=1.0,
      temperature_drop_move=8,
      temperature_post_drop=1e-3,
      batch_size=256,
      num_iterations=200,
      num_epochs=4,
      minibatch_size=1024,
      replay_buffer_size=4,
      lr=1e-3,
      lr_end=1e-4,
      weight_decay=1e-4,
      max_grad_norm=1.0,
      value_coef=0.5,
  )


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  return recipes.alphazero("az_connect_four", az_config(), keep_every=10)
