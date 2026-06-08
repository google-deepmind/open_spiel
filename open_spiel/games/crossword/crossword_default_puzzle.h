// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_DEFAULT_PUZZLE_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_DEFAULT_PUZZLE_H_

namespace open_spiel {
namespace crossword {

inline constexpr const char* kDefaultPuzzleContents = R"xd(
Title: Default crossword puzzle.
Author: OpenSpiel authors
Copyright: © 2026 OpenSpiel authors
Date: 2025-06-05


GO#A
ARID
M#TA
EDAM

A1. Ancient board game with black and white stones ~ GO
A4. Dry, like a desert ~ ARID
A7. Teaching assistant ~ TA
A8. Dutch cheese ~ EDAM

D1. Chess is an example of a board ____ ~ GAME
D2. Disjunction (logical operator) ~ OR
D3. Popular optimizer ~ ADAM
D6. Invitation to Apply, abbreviated ~ ITA

)xd";

}  // namespace crossword
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_DEFAULT_PUZZLE_H_
