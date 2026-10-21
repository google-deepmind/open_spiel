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

#include "open_spiel/games/abalone/abalone_boards.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace abalone_core {

// Moved from abalone_core.cpp.  The Belgian Daisy (Marguerite Belge) is the
// tournament standard adopted in 1999.
const CellState ABALONE_INIT_BELGIAN_DAISY[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},    // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},      // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},        // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},    // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player0},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player0},    // i
};

// Decoded from Abalone_german.svg (Wikimedia Commons, public domain by
// François Haffner).  Two clusters of 7 marbles per player, diagonally
// opposite, but with a more compact diamond shape than the Belgian Daisy.
const CellState ABALONE_INIT_GERMAN_DAISY[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Player0},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player0},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// Decoded from Abalone_holland.svg.  Similar to the Belgian Daisy but with
// an intruder marble of the opposite colour in each cluster.
const CellState ABALONE_INIT_DUTCH_DAISY[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player0},  // i
};

// Decoded from Abalone_swiss.svg.  Similar to the German Daisy but with
// an intruder marble of the opposite colour in each cluster.
const CellState ABALONE_INIT_SWISS_DAISY[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Player0},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player0},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// Decoded from Abalone_snakes.svg.  Each player's marbles form a diagonal
// "snake" from one corner to the opposite, forcing early confrontation.
const CellState ABALONE_INIT_SNAKE[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Player1, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0},  // i
};

// Decoded from gym-abalone variants.json (the-wall). 14B + 14W.
// Each player forms a wall across the board, forcing early confrontation.
const CellState ABALONE_INIT_WALL[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// 6-9 (from gym-abalone variants.json)
const CellState ABALONE_INIT_6_9[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// alien-attack (from gym-abalone variants.json)
const CellState ABALONE_INIT_ALIEN_ATTACK[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Invalid, CellState::Player0,
     CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// aliening (from gym-abalone variants.json)
const CellState ABALONE_INIT_ALIENING[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// alitration (from gym-abalone variants.json)
const CellState ABALONE_INIT_ALITRATION[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Invalid, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// anglattack (from gym-abalone variants.json)
const CellState ABALONE_INIT_ANGLATTACK[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Player0, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player1},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Player1, CellState::Invalid,
     CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// april-fish (from gym-abalone variants.json)
const CellState ABALONE_INIT_APRIL_FISH[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// buttonhole (from gym-abalone variants.json)
const CellState ABALONE_INIT_BUTTONHOLE[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0},  // e
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// centrifuge (from gym-abalone variants.json)
const CellState ABALONE_INIT_CENTRIFUGE[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty},  // f
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// checkerboard (from gym-abalone variants.json)
const CellState ABALONE_INIT_CHECKERBOARD[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// cross (from gym-abalone variants.json)
const CellState ABALONE_INIT_CROSS[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Invalid, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// crossed-hearts (from gym-abalone variants.json)
const CellState ABALONE_INIT_CROSSED_HEARTS[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// crown (from gym-abalone variants.json)
const CellState ABALONE_INIT_CROWN[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// def-or-atak (from gym-abalone variants.json)
const CellState ABALONE_INIT_DEF_OR_ATAK[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// domination (from gym-abalone variants.json)
const CellState ABALONE_INIT_DOMINATION[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty},  // f
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// domination-vs-daisy (from gym-abalone variants.json)
const CellState ABALONE_INIT_DOMINATION_VS_DAISY[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// duel (from gym-abalone variants.json)
const CellState ABALONE_INIT_DUEL[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player0},  // e
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player0, CellState::Invalid,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// face-2-face (from gym-abalone variants.json)
const CellState ABALONE_INIT_FACE_2_FACE[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Invalid},  // d
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1},  // e
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty},  // f
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// food-mill (from gym-abalone variants.json)
const CellState ABALONE_INIT_FOOD_MILL[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Player1, CellState::Player0, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player0},  // e
    {CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// forcing (from gym-abalone variants.json)
const CellState ABALONE_INIT_FORCING[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player1, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player0, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player0},  // e
    {CellState::Player1, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// fracture (from gym-abalone variants.json)
const CellState ABALONE_INIT_FRACTURE[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player0, CellState::Empty},  // e
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// french-daisy (from gym-abalone variants.json)
const CellState ABALONE_INIT_FRENCH_DAISY[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// french-king-cake1 (from gym-abalone variants.json)
const CellState ABALONE_INIT_FRENCH_KING_CAKE1[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player0, CellState::Invalid},  // d
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0},  // e
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// fujiyama1 (from gym-abalone variants.json)
const CellState ABALONE_INIT_FUJIYAMA1[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// horseshoes (from gym-abalone variants.json)
const CellState ABALONE_INIT_HORSESHOES[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// i_m-a-robot (from gym-abalone variants.json)
const CellState ABALONE_INIT_I_M_A_ROBOT[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player0, CellState::Invalid,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// infiltralien (from gym-abalone variants.json)
const CellState ABALONE_INIT_INFILTRALIEN[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// infiltration (from gym-abalone variants.json)
const CellState ABALONE_INIT_INFILTRATION[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// intimidation (from gym-abalone variants.json)
const CellState ABALONE_INIT_INTIMIDATION[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// irish-daisy (from gym-abalone variants.json)
const CellState ABALONE_INIT_IRISH_DAISY[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// labyrinth (from gym-abalone variants.json)
const CellState ABALONE_INIT_LABYRINTH[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty},  // e
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Player0, CellState::Invalid, CellState::Empty,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// mosaic (from gym-abalone variants.json)
const CellState ABALONE_INIT_MOSAIC[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// mquorric-1 (from gym-abalone variants.json)
const CellState ABALONE_INIT_MQUORRIC_1[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// neko (from gym-abalone variants.json)
const CellState ABALONE_INIT_NEKO[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Invalid, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// oath (from gym-abalone variants.json)
const CellState ABALONE_INIT_OATH[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Player0},  // e
    {CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// octopus (from gym-abalone variants.json)
const CellState ABALONE_INIT_OCTOPUS[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// pyramid (from gym-abalone variants.json)
const CellState ABALONE_INIT_PYRAMID[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// scottish-daisy (from gym-abalone variants.json)
const CellState ABALONE_INIT_SCOTTISH_DAISY[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player0, CellState::Player1, CellState::Player1,
     CellState::Player0, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player0, CellState::Invalid,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// slaloms (from gym-abalone variants.json)
const CellState ABALONE_INIT_SLALOMS[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty},  // e
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// snowflake (from gym-abalone variants.json)
const CellState ABALONE_INIT_SNOWFLAKE[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty},  // e
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Player1, CellState::Invalid,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// spider (from gym-abalone variants.json)
const CellState ABALONE_INIT_SPIDER[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player1, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// star (from gym-abalone variants.json)
const CellState ABALONE_INIT_STAR[kNumRows][kNumCols] = {
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player0, CellState::Invalid, CellState::Player1,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// tinsels (from gym-abalone variants.json)
const CellState ABALONE_INIT_TINSELS[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Player1,
     CellState::Player0, CellState::Player1, CellState::Empty},  // e
    {CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player0, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// trench (from gym-abalone variants.json)
const CellState ABALONE_INIT_TRENCH[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// true-daisy (from gym-abalone variants.json)
const CellState ABALONE_INIT_TRUE_DAISY[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player0, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Player1, CellState::Invalid, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Player1, CellState::Invalid, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// wedding-rings (from gym-abalone variants.json)
const CellState ABALONE_INIT_WEDDING_RINGS[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player0, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// wheel (from gym-abalone variants.json)
const CellState ABALONE_INIT_WHEEL[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Player1, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player0},  // e
    {CellState::Invalid, CellState::Empty, CellState::Player0,
     CellState::Player1, CellState::Player0, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player1, CellState::Invalid,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player1, CellState::Invalid, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// windmill (from gym-abalone variants.json)
const CellState ABALONE_INIT_WINDMILL[kNumRows][kNumCols] = {
    {CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Player0, CellState::Player0, CellState::Empty,
     CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Empty, CellState::Player0, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Player0, CellState::Invalid},  // d
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Player0, CellState::Player0},  // e
    {CellState::Player1, CellState::Player1, CellState::Empty,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // f
    {CellState::Invalid, CellState::Invalid, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Player0, CellState::Player0, CellState::Invalid,
     CellState::Invalid, CellState::Player1, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

// xmas-star (from gym-abalone variants.json)
const CellState ABALONE_INIT_XMAS_STAR[kNumRows][kNumCols] = {
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Invalid,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // a
    {CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty,
     CellState::Invalid, CellState::Invalid, CellState::Invalid},  // b
    {CellState::Player1, CellState::Player1, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Player1, CellState::Invalid, CellState::Invalid},  // c
    {CellState::Empty, CellState::Player1, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Invalid},  // d
    {CellState::Empty, CellState::Empty, CellState::Player1,
     CellState::Empty, CellState::Empty, CellState::Empty,
     CellState::Player0, CellState::Empty, CellState::Empty},  // e
    {CellState::Invalid, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Empty, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // f
    {CellState::Player0, CellState::Player0, CellState::Player0,
     CellState::Empty, CellState::Player1, CellState::Player0,
     CellState::Player0, CellState::Empty, CellState::Empty},  // g
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Player1, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // h
    {CellState::Invalid, CellState::Invalid, CellState::Player1,
     CellState::Invalid, CellState::Empty, CellState::Empty,
     CellState::Empty, CellState::Empty, CellState::Empty},  // i
};

namespace {

// Single source of truth for all fixed-layout board names and their arrays.
const std::unordered_map<std::string, const CellState (*)[kNumCols]>&
BoardRegistry() {
  static const std::unordered_map<std::string,
                                  const CellState (*)[kNumCols]>
      kBoards = {
          {"belgian-daisy", ABALONE_INIT_BELGIAN_DAISY},
          {"german-daisy", ABALONE_INIT_GERMAN_DAISY},
          {"dutch-daisy", ABALONE_INIT_DUTCH_DAISY},
          {"swiss-daisy", ABALONE_INIT_SWISS_DAISY},
          {"snakes", ABALONE_INIT_SNAKE},
          {"the-wall", ABALONE_INIT_WALL},
          {"six-nine", ABALONE_INIT_6_9},
          {"alien-attack", ABALONE_INIT_ALIEN_ATTACK},
          {"aliening", ABALONE_INIT_ALIENING},
          {"alitration", ABALONE_INIT_ALITRATION},
          {"anglattack", ABALONE_INIT_ANGLATTACK},
          {"april-fish", ABALONE_INIT_APRIL_FISH},
          {"buttonhole", ABALONE_INIT_BUTTONHOLE},
          {"centrifuge", ABALONE_INIT_CENTRIFUGE},
          {"checkerboard", ABALONE_INIT_CHECKERBOARD},
          {"cross", ABALONE_INIT_CROSS},
          {"crossed-hearts", ABALONE_INIT_CROSSED_HEARTS},
          {"crown", ABALONE_INIT_CROWN},
          {"def-or-atak", ABALONE_INIT_DEF_OR_ATAK},
          {"domination", ABALONE_INIT_DOMINATION},
          {"domination-vs-daisy", ABALONE_INIT_DOMINATION_VS_DAISY},
          {"duel", ABALONE_INIT_DUEL},
          {"face-2-face", ABALONE_INIT_FACE_2_FACE},
          {"food-mill", ABALONE_INIT_FOOD_MILL},
          {"forcing", ABALONE_INIT_FORCING},
          {"fracture", ABALONE_INIT_FRACTURE},
          {"french-daisy", ABALONE_INIT_FRENCH_DAISY},
          {"french-king-cake1", ABALONE_INIT_FRENCH_KING_CAKE1},
          {"fujiyama1", ABALONE_INIT_FUJIYAMA1},
          {"horseshoes", ABALONE_INIT_HORSESHOES},
          {"i_m-a-robot", ABALONE_INIT_I_M_A_ROBOT},
          {"infiltralien", ABALONE_INIT_INFILTRALIEN},
          {"infiltration", ABALONE_INIT_INFILTRATION},
          {"intimidation", ABALONE_INIT_INTIMIDATION},
          {"irish-daisy", ABALONE_INIT_IRISH_DAISY},
          {"labyrinth", ABALONE_INIT_LABYRINTH},
          {"mosaic", ABALONE_INIT_MOSAIC},
          {"mquorric-1", ABALONE_INIT_MQUORRIC_1},
          {"neko", ABALONE_INIT_NEKO},
          {"oath", ABALONE_INIT_OATH},
          {"octopus", ABALONE_INIT_OCTOPUS},
          {"pyramid", ABALONE_INIT_PYRAMID},
          {"scottish-daisy", ABALONE_INIT_SCOTTISH_DAISY},
          {"slaloms", ABALONE_INIT_SLALOMS},
          {"snowflake", ABALONE_INIT_SNOWFLAKE},
          {"spider", ABALONE_INIT_SPIDER},
          {"star", ABALONE_INIT_STAR},
          {"tinsels", ABALONE_INIT_TINSELS},
          {"trench", ABALONE_INIT_TRENCH},
          {"true-daisy", ABALONE_INIT_TRUE_DAISY},
          {"wedding-rings", ABALONE_INIT_WEDDING_RINGS},
          {"wheel", ABALONE_INIT_WHEEL},
          {"windmill", ABALONE_INIT_WINDMILL},
          {"xmas-star", ABALONE_INIT_XMAS_STAR},
      };
  return kBoards;
}

}  // namespace

const CellState (*BoardForName(const std::string& name))[kNumCols] {
  const auto& boards = BoardRegistry();
  auto it = boards.find(name);
  if (it == boards.end()) return nullptr;
  return it->second;
}

std::vector<std::string> BoardNames() {
  const auto& boards = BoardRegistry();
  std::vector<std::string> names;
  names.reserve(boards.size());
  for (const auto& [name, _] : boards) names.push_back(name);
  return names;
}

}  // namespace abalone_core
