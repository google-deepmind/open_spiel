// Copyright 2023 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_ABALONE_ABALONE_BOARDS_H_
#define OPEN_SPIEL_GAMES_ABALONE_ABALONE_BOARDS_H_

#include <string>
#include <vector>

#include "open_spiel/games/abalone/abalone_core.h"

// Starting positions (other than the standard "classic" one, which lives in
// abalone_core.cpp alongside VALID_BOARD).  Each array is a complete 9x9
// board in the same coordinate system as abalone_core::core_state::board_.
//
// Sources for the layouts:
//   - Belgian/German/Dutch/Swiss Daisy, Snake, The Wall: abalonegames.com
//     and Wikimedia Commons SVGs (public domain by François Haffner).
//   - All other layouts: gym-abalone (github.com/towzeur/gym-abalone),
//     variants.json, MIT licence.

namespace abalone_core {

extern const CellState ABALONE_INIT_BELGIAN_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_GERMAN_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_DUTCH_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_SWISS_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_SNAKE[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_WALL[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_6_9[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_ALIEN_ATTACK[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_ALIENING[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_ALITRATION[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_ANGLATTACK[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_APRIL_FISH[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_BUTTONHOLE[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_CENTRIFUGE[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_CHECKERBOARD[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_CROSS[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_CROSSED_HEARTS[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_CROWN[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_DEF_OR_ATAK[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_DOMINATION[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_DOMINATION_VS_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_DUEL[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FACE_2_FACE[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FOOD_MILL[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FORCING[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FRACTURE[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FRENCH_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FRENCH_KING_CAKE1[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_FUJIYAMA1[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_HORSESHOES[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_I_M_A_ROBOT[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_INFILTRALIEN[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_INFILTRATION[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_INTIMIDATION[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_IRISH_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_LABYRINTH[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_MOSAIC[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_MQUORRIC_1[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_NEKO[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_OATH[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_OCTOPUS[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_PYRAMID[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_SCOTTISH_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_SLALOMS[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_SNOWFLAKE[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_SPIDER[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_STAR[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_TINSELS[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_TRENCH[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_TRUE_DAISY[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_WEDDING_RINGS[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_WHEEL[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_WINDMILL[kNumRows][kNumCols];
extern const CellState ABALONE_INIT_XMAS_STAR[kNumRows][kNumCols];

// Maps a board name (as used in the "board" game parameter) to its layout.
// Returns nullptr if the name is not a fixed layout (e.g. "classical" and
// "random-symmetric" are handled by the caller) or is unknown.
const CellState (*BoardForName(const std::string& name))[kNumCols];

// Returns the names of all fixed-layout boards registered in BoardForName.
std::vector<std::string> BoardNames();

}  // namespace abalone_core

#endif  // OPEN_SPIEL_GAMES_ABALONE_ABALONE_BOARDS_H_
