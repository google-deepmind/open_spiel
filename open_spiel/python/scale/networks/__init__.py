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

"""Networks: a Flax network library and the Policy over it.

  torsos        MLPTorso, CNNTorso, RecurrentTorso: features from encoded inputs
  heads         DensePolicyHead, ValueHead (scalar or HL-Gauss), AuxHead
  actor_critic  ActorCritic and build_network: a network from a JSON config
  catalog       named network configs and their builders (mlp, cnn)
  policy        NetworkPolicy: an encoder plus a network, as a Policy
  hl_gauss      the HL-Gauss categorical value transform
  layers        the dtype policy, activations and initialisers

register_torso and register_policy_head (actor_critic) add network kinds,
catalog.register adds named configs.
"""
