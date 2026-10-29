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

"""Encoders: InfoState -> network input, one module per game.

An encoder reads an InfoState pytree only, never env state, so a policy
built on it cannot see what its seat does not know. Each module registers
its encoders with `encoders.registry`; a policy names one by its obs_format.

  enc = registry.get("connect_four_planes")       # f32[6, 7, 2]

Names are `<game>_<what>`: `*_observation_tensor` and `*_info_state` encode
what the game's observation and information-state tensors hold (OpenSpiel's,
or the JAX engine's where the two layouts differ; each Encoder's
`reference` names which), `*_planes` are board planes for convolutional
networks.
"""
