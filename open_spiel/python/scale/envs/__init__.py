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

"""Envs: one Env per game, and the game table (envs.registry).

An Env wraps one game's dynamics behind the core.types.Env contract: a raw
layer that exposes chance nodes (root, apply, apply_chance), a resolved layer
that samples them (init, step), and the InfoState of each seat. Opponents,
encoders and networks live elsewhere.
"""
