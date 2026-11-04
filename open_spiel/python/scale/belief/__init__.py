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

"""Belief samplers: complete game states consistent with what one seat has seen.

Search in an imperfect-information game needs concrete worlds to expand. A
belief sampler turns a seat's InfoState history into K complete env states,
which SearchPolicy uses as roots and averages over (perfect-information
Monte Carlo).

  sampler = belief.registry.build("enumeration", env)
  params = sampler.table_for(opponent, opponent_params, core.types.SAMPLE)
  carry = sampler.init_carry(params, batch, seat, key)
  carry = sampler.observe(params, carry, step)          # once per SeatStep
  worlds, logw = sampler.sample_worlds(params, carry, step.info, k=32, key=key)

The protocol (core.types.BeliefSampler):

  init_params(key) -> params
  init_carry(params, batch, seat, key) -> carry
  observe(params, carry, step) -> carry
  sample_worlds(params, carry, info, k, key) -> (worlds, logw)
      worlds: env states with leaves [B, K, ...]; logw: log-weights [B, K]
      (0 for exact draws).

Every method takes `params`, so a learned sampler has the same shape as a
Policy; a sampler without parameters returns {} from init_params. Samplers
read only the seat's InfoStates, never the true state. Attributes:

  is_perfect_information  True if the InfoState determines the world, in
      which case search values carry no strategy-fusion bias.
  params_source           where `params` come from: "none", "own" (the
      searching policy's params) or "opponent" (an opponent policy's params
      or table, set by the caller).
  kwargs                  JSON constructor arguments besides `env`;
      `registry.build(registry.name_of(s), env, **s.kwargs)` rebuilds s.
  stateful                bool: the sampler keeps a carry across plies (its
      observe changes it); a search over a stateful sampler is stateful.
  capability              core.types.Capability; a SearchPolicy needs
      NATIVE_JAX.

registry.check refuses a sampler that lacks any of these.

Shipped samplers:

  perfect_information  K copies of Env.world_from_info: perfect-information
                       games.
  enumeration          the exact posterior for small games, from the full game
                       tree and an opponent policy table.
  particle             sequential importance resampling for games too large
                       to enumerate; approximate.

Game-specific samplers live with their game and are declared in its
GameSpec.components:

  leduc                (envs/leduc/belief.py) the exact posterior over leduc's
                       hidden card under an opponent model.
  gin                  (envs/gin/belief.py) uniform-consistent determinized
                       gin rummy deals.

A new sampler is a class with the attributes and methods above, registered
with `belief.registry.register(name, cls)`.
"""
