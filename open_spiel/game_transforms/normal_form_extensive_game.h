// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_NORMAL_FORM_EXTENSIVE_GAME_H
#define OPEN_SPIEL_GAME_TRANSFORMS_NORMAL_FORM_EXTENSIVE_GAME_H

#include <memory>

#include "open_spiel/spiel.h"
#include "open_spiel/tensor_game.h"

namespace open_spiel {

// Creates an extensive-form game (EFG)'s equivalent tensor game.
//
// Note that this tensor game will have a row (or axis-1 slice, etc) for each
// deterministic policy in the extensive-form game. As such, it will be
// exponentially larger than the extensive-form game. In particular, if S_i is
// number of information states for player i, and A(s_i) for s_i in S_i is the
// set of legal actions at s_i, then the number of deterministic policies is
// the product \Prod_{s_i in S_i) |A(s_i)|, and can include many redundant
// policies that differ, e.g., only in unreachable states. See Chapter 5 of
// (Shoham and Leyton-Brown, Multiagent Systems Algorithmic, Game-Theoretic, and
// Logical Foundations, 2009, http://masfoundations.org/) for more detail,
// including examples of the transformations.
//
// Hence, this method should only be used for  small games! For example, Kuhn
// poker has 64 deterministic policies, resulting in a 64-by-64 matrix.

std::shared_ptr<const tensor_game::TensorGame> ExtensiveToTensorGame(
    const Game& game);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_NORMAL_FORM_EXTENSIVE_GAME_H_
