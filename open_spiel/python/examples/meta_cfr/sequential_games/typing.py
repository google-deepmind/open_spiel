# Copyright 2022 DeepMind Technologies Limited
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

"""Typing definitions."""

from typing import Any, Dict, Callable
import jax.numpy as jnp
import optax
from open_spiel.python.examples.meta_cfr.sequential_games import game_tree_utils

PyTree = Any
Params = PyTree
ApplyFn = Callable[..., jnp.ndarray]
OptState = optax.OptState

GameTree = game_tree_utils.GameTree
InfostateNode = game_tree_utils.InfoState
InfostateMapping = Dict[str, InfostateNode]
HistoryNode = game_tree_utils.HistoryTreeNode

