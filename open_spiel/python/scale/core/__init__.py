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

"""The contracts every component implements, and the rollout drivers.

  types        Env, Policy, Learner, SeatStep, TrajectoryBatch, Manifest, ...
  decide       logits and a legal mask to an action
  arena        the in-graph rollout driver; host_arena, the host-driven one
  slots        Slot, SlotTable, TemperatureDrop
  chance, plies, sequence, x64, rng, frameworks, aux

`types`, `slots`, `frameworks` and `aux` import no array framework; `rng`
imports numpy only; every other module imports jax. Import each submodule
explicitly (`core.decide`, not `core`).
"""
