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

