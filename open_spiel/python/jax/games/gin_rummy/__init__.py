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

"""Gin rummy, OpenSpiel's `gin_rummy`, in JAX.

  game = gin_rummy.GinRummy(oklahoma=True)        # this module
  state = game.new_initial_state()

Modules:
- `cards_and_melds`: constants, card encoding, meld enumeration and masks.
- `lookup_tables`: run-score lookup tables.
- `deadwood`: deadwood and optimal-meld computation.
- `layoffs`: layoffs onto the knocker's melds.
- `game`: the state machine (`init_state`, `step`, `legal_actions_mask`,
  `chance_outcomes`, `get_returns`) and `GinRummy`.
- `dense_cache_game`, `dense_cache`: the dense-cache backend,
  `GinRummy(backend="dense_cache")`.
- `simple_bot`: OpenSpiel's simple gin rummy bot as a JAX policy.

This package re-exports the commonly used names of those modules.
"""

from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_STOCK,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_MELD_BASE,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_PASS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ALL_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import card_rank
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import card_str
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import card_suit
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import CHANCE_PLAYER
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import DP_STATE_SIZE
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    generate_all_melds,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_card_points,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_card_values,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_cards_bool,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_overlap,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_overlap_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_points,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_points_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_sizes
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import GIN_BONUS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import HAND_SIZE
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    KNOCK_THRESHOLD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import make_card
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import MAX_HAND_SIZE
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    MAX_NUM_DRAW_UPCARD_ACTIONS,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    MAX_VALID_MELDS,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import meld_to_mask
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_ACTIONS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_CARDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_RANKS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_SUITS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DEAL
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DISCARD
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DRAW
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_FIRST_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_GAME_OVER,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_LAYOFF
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_WALL
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import STATE_DTYPES
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    TERMINAL_PLAYER,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import UNDERCUT_BONUS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    WALL_STOCK_SIZE,
)
from open_spiel.python.jax.games.gin_rummy.game import chance_outcomes
from open_spiel.python.jax.games.gin_rummy.game import chance_outcomes_mask
from open_spiel.python.jax.games.gin_rummy.game import DEFAULT_RULES
from open_spiel.python.jax.games.gin_rummy.game import get_hand
from open_spiel.python.jax.games.gin_rummy.game import get_returns
from open_spiel.python.jax.games.gin_rummy.game import GinRummy
from open_spiel.python.jax.games.gin_rummy.game import init_dealt_state
from open_spiel.python.jax.games.gin_rummy.game import init_state
from open_spiel.python.jax.games.gin_rummy.game import is_chance_node
from open_spiel.python.jax.games.gin_rummy.game import legal_actions_mask
from open_spiel.python.jax.games.gin_rummy.game import OBSERVATION_TENSOR_SIZE
from open_spiel.python.jax.games.gin_rummy.game import observation_tensor_size
from open_spiel.python.jax.games.gin_rummy.game import oklahoma_knock_card
from open_spiel.python.jax.games.gin_rummy.game import RULE_NAMES
from open_spiel.python.jax.games.gin_rummy.game import (
    SCORE_MULTIPLIER_PLANE_SIZE,
)
from open_spiel.python.jax.games.gin_rummy.game import step
from open_spiel.python.jax.games.gin_rummy.game import UPCARD_TAKEN_PLANE_SIZE
from open_spiel.python.jax.games.gin_rummy.layoffs import compute_layoff_cards
from open_spiel.python.jax.games.gin_rummy.layoffs import init_lazy_constants
from open_spiel.python.jax.games.gin_rummy.layoffs import (
    lazy_constants_initialized,
)

__all__ = [
    "ACTION_DRAW_STOCK",
    "ACTION_DRAW_UPCARD",
    "ACTION_KNOCK",
    "ACTION_MELD_BASE",
    "ACTION_PASS",
    "ALL_MELDS",
    "CHANCE_PLAYER",
    "DEFAULT_RULES",
    "DP_STATE_SIZE",
    "GIN_BONUS",
    "GinRummy",
    "HAND_SIZE",
    "KNOCK_THRESHOLD",
    "MAX_HAND_SIZE",
    "MAX_NUM_DRAW_UPCARD_ACTIONS",
    "MAX_VALID_MELDS",
    "NUM_ACTIONS",
    "NUM_CARDS",
    "NUM_MELDS",
    "NUM_RANKS",
    "NUM_SUITS",
    "OBSERVATION_TENSOR_SIZE",
    "PHASE_DEAL",
    "PHASE_DISCARD",
    "PHASE_DRAW",
    "PHASE_FIRST_UPCARD",
    "PHASE_GAME_OVER",
    "PHASE_KNOCK",
    "PHASE_LAYOFF",
    "PHASE_WALL",
    "RULE_NAMES",
    "SCORE_MULTIPLIER_PLANE_SIZE",
    "STATE_DTYPES",
    "TERMINAL_PLAYER",
    "UNDERCUT_BONUS",
    "UPCARD_TAKEN_PLANE_SIZE",
    "WALL_STOCK_SIZE",
    "card_rank",
    "card_str",
    "card_suit",
    "chance_outcomes",
    "chance_outcomes_mask",
    "compute_layoff_cards",
    "generate_all_melds",
    "get_card_points",
    "get_card_values",
    "get_hand",
    "get_meld_cards_bool",
    "get_meld_masks",
    "get_meld_overlap",
    "get_meld_overlap_padded",
    "get_meld_points",
    "get_meld_points_padded",
    "get_meld_sizes",
    "get_returns",
    "init_dealt_state",
    "init_lazy_constants",
    "lazy_constants_initialized",
    "init_state",
    "is_chance_node",
    "legal_actions_mask",
    "make_card",
    "observation_tensor_size",
    "oklahoma_knock_card",
    "meld_to_mask",
    "step",
]
