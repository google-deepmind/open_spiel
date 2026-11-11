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

r"""Reinforcement learning for games in JAX (it needs jax_enable_x64 on).

Self-play, population methods (PSRO and its relatives), behaviour cloning and
AlphaZero-style search over OpenSpiel games implemented in JAX, with one
rollout driver, one decision rule, one checkpoint format and one trainer.

  python -m open_spiel.python.scale.train --config=mmd_kuhn_poker_jax \
      --checkpoint_dir=/tmp/kuhn

The package computes counters in int64 and decides in f64, so it needs
jax_enable_x64 on: a frame counter then never wraps and a float64 leaf restores
as float64. Importing the package changes nothing in the process (other JAX code
beside it keeps its own setting). `enable_x64()` turns it on; the entry points
(the trainer, the league runner's run(), the launcher, the policy server) call
it, and code that uses the components directly calls it once, before it creates
arrays. Components assert it (`core.x64.assert_x64()`) and raise if it is off.

Components live in subpackages: core, configs, envs, encoders, networks,
policies, belief, learners, sources, league, ckpt, interop, serve, eval,
launch, parallel and testing.
"""

from open_spiel.python.scale.core import frameworks as _frameworks

# Defined in core.frameworks (which core.x64 re-exports too), so no
# subpackage imports this package's __init__ for it.
enable_x64 = _frameworks.enable_x64
