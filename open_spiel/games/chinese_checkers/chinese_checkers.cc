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

#include "open_spiel/games/chinese_checkers/chinese_checkers.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chinese_checkers {
namespace {

const GameType kGameType{
    /*short_name=*/"chinese_checkers",
    /*long_name=*/"Chinese Checkers",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kMaxNumPlayers,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {
        {"players", GameParameter(kDefaultNumPlayers)},
        {"max_moves", GameParameter(kDefaultMaxMoves)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ChineseCheckersGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// --- Precomputed board topology ---

// Player slot assignments per player count.
const std::vector<int> kSlots2 = {0, 3};
const std::vector<int> kSlots3 = {0, 2, 4};
const std::vector<int> kSlots4 = {0, 1, 3, 4};
const std::vector<int> kSlots6 = {0, 1, 2, 3, 4, 5};

}  // namespace

// Row coordinates for each position (0-120).
const int kCellRow[kNumPositions] = {
    0, 1, 1, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 13, 13, 13, 13, 14, 14, 14, 15, 15,
    16
};

// Column coordinates (doubled) for each position.
const int kCellCol[kNumPositions] = {
    12, 11, 13, 10, 12, 14, 9, 11, 13, 15,
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
    20, 22, 24, 1, 3, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 2, 4, 6, 8, 10,
    12, 14, 16, 18, 20, 22, 3, 5, 7, 9,
    11, 13, 15, 17, 19, 21, 4, 6, 8, 10,
    12, 14, 16, 18, 20, 3, 5, 7, 9, 11,
    13, 15, 17, 19, 21, 2, 4, 6, 8, 10,
    12, 14, 16, 18, 20, 22, 1, 3, 5, 7,
    9, 11, 13, 15, 17, 19, 21, 23, 0, 2,
    4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
    24, 9, 11, 13, 15, 10, 12, 14, 11, 13,
    12
};

// Neighbor in each of 6 directions (-1 = off-board).
// Directions:
//   0=UL(-1,-1) 1=UR(-1,+1) 2=L(0,-2) 3=R(0,+2) 4=DL(+1,-1) 5=DR(+1,+1)
const int kNeighbor[kNumPositions][kNumDirections] = {
    { -1,  -1,  -1,  -1,   1,   2},
    { -1,   0,  -1,   2,   3,   4},
    {  0,  -1,   1,  -1,   4,   5},
    { -1,   1,  -1,   4,   6,   7},
    {  1,   2,   3,   5,   7,   8},
    {  2,  -1,   4,  -1,   8,   9},
    { -1,   3,  -1,   7,  14,  15},
    {  3,   4,   6,   8,  15,  16},
    {  4,   5,   7,   9,  16,  17},
    {  5,  -1,   8,  -1,  17,  18},
    { -1,  -1,  -1,  11,  -1,  23},
    { -1,  -1,  10,  12,  23,  24},
    { -1,  -1,  11,  13,  24,  25},
    { -1,  -1,  12,  14,  25,  26},
    { -1,   6,  13,  15,  26,  27},
    {  6,   7,  14,  16,  27,  28},
    {  7,   8,  15,  17,  28,  29},
    {  8,   9,  16,  18,  29,  30},
    {  9,  -1,  17,  19,  30,  31},
    { -1,  -1,  18,  20,  31,  32},
    { -1,  -1,  19,  21,  32,  33},
    { -1,  -1,  20,  22,  33,  34},
    { -1,  -1,  21,  -1,  34,  -1},
    { 10,  11,  -1,  24,  -1,  35},
    { 11,  12,  23,  25,  35,  36},
    { 12,  13,  24,  26,  36,  37},
    { 13,  14,  25,  27,  37,  38},
    { 14,  15,  26,  28,  38,  39},
    { 15,  16,  27,  29,  39,  40},
    { 16,  17,  28,  30,  40,  41},
    { 17,  18,  29,  31,  41,  42},
    { 18,  19,  30,  32,  42,  43},
    { 19,  20,  31,  33,  43,  44},
    { 20,  21,  32,  34,  44,  45},
    { 21,  22,  33,  -1,  45,  -1},
    { 23,  24,  -1,  36,  -1,  46},
    { 24,  25,  35,  37,  46,  47},
    { 25,  26,  36,  38,  47,  48},
    { 26,  27,  37,  39,  48,  49},
    { 27,  28,  38,  40,  49,  50},
    { 28,  29,  39,  41,  50,  51},
    { 29,  30,  40,  42,  51,  52},
    { 30,  31,  41,  43,  52,  53},
    { 31,  32,  42,  44,  53,  54},
    { 32,  33,  43,  45,  54,  55},
    { 33,  34,  44,  -1,  55,  -1},
    { 35,  36,  -1,  47,  -1,  56},
    { 36,  37,  46,  48,  56,  57},
    { 37,  38,  47,  49,  57,  58},
    { 38,  39,  48,  50,  58,  59},
    { 39,  40,  49,  51,  59,  60},
    { 40,  41,  50,  52,  60,  61},
    { 41,  42,  51,  53,  61,  62},
    { 42,  43,  52,  54,  62,  63},
    { 43,  44,  53,  55,  63,  64},
    { 44,  45,  54,  -1,  64,  -1},
    { 46,  47,  -1,  57,  65,  66},
    { 47,  48,  56,  58,  66,  67},
    { 48,  49,  57,  59,  67,  68},
    { 49,  50,  58,  60,  68,  69},
    { 50,  51,  59,  61,  69,  70},
    { 51,  52,  60,  62,  70,  71},
    { 52,  53,  61,  63,  71,  72},
    { 53,  54,  62,  64,  72,  73},
    { 54,  55,  63,  -1,  73,  74},
    { -1,  56,  -1,  66,  75,  76},
    { 56,  57,  65,  67,  76,  77},
    { 57,  58,  66,  68,  77,  78},
    { 58,  59,  67,  69,  78,  79},
    { 59,  60,  68,  70,  79,  80},
    { 60,  61,  69,  71,  80,  81},
    { 61,  62,  70,  72,  81,  82},
    { 62,  63,  71,  73,  82,  83},
    { 63,  64,  72,  74,  83,  84},
    { 64,  -1,  73,  -1,  84,  85},
    { -1,  65,  -1,  76,  86,  87},
    { 65,  66,  75,  77,  87,  88},
    { 66,  67,  76,  78,  88,  89},
    { 67,  68,  77,  79,  89,  90},
    { 68,  69,  78,  80,  90,  91},
    { 69,  70,  79,  81,  91,  92},
    { 70,  71,  80,  82,  92,  93},
    { 71,  72,  81,  83,  93,  94},
    { 72,  73,  82,  84,  94,  95},
    { 73,  74,  83,  85,  95,  96},
    { 74,  -1,  84,  -1,  96,  97},
    { -1,  75,  -1,  87,  98,  99},
    { 75,  76,  86,  88,  99, 100},
    { 76,  77,  87,  89, 100, 101},
    { 77,  78,  88,  90, 101, 102},
    { 78,  79,  89,  91, 102, 103},
    { 79,  80,  90,  92, 103, 104},
    { 80,  81,  91,  93, 104, 105},
    { 81,  82,  92,  94, 105, 106},
    { 82,  83,  93,  95, 106, 107},
    { 83,  84,  94,  96, 107, 108},
    { 84,  85,  95,  97, 108, 109},
    { 85,  -1,  96,  -1, 109, 110},
    { -1,  86,  -1,  99,  -1,  -1},
    { 86,  87,  98, 100,  -1,  -1},
    { 87,  88,  99, 101,  -1,  -1},
    { 88,  89, 100, 102,  -1,  -1},
    { 89,  90, 101, 103,  -1, 111},
    { 90,  91, 102, 104, 111, 112},
    { 91,  92, 103, 105, 112, 113},
    { 92,  93, 104, 106, 113, 114},
    { 93,  94, 105, 107, 114,  -1},
    { 94,  95, 106, 108,  -1,  -1},
    { 95,  96, 107, 109,  -1,  -1},
    { 96,  97, 108, 110,  -1,  -1},
    { 97,  -1, 109,  -1,  -1,  -1},
    {102, 103,  -1, 112,  -1, 115},
    {103, 104, 111, 113, 115, 116},
    {104, 105, 112, 114, 116, 117},
    {105, 106, 113,  -1, 117,  -1},
    {111, 112,  -1, 116,  -1, 118},
    {112, 113, 115, 117, 118, 119},
    {113, 114, 116,  -1, 119,  -1},
    {115, 116,  -1, 119,  -1, 120},
    {116, 117, 118,  -1, 120,  -1},
    {118, 119,  -1,  -1,  -1,  -1}
};

// Hop landing in each of 6 directions (-1 = off-board).
const int kHopDest[kNumPositions][kNumDirections] = {
    { -1,  -1,  -1,  -1,   3,   5},
    { -1,  -1,  -1,  -1,   6,   8},
    { -1,  -1,  -1,  -1,   7,   9},
    { -1,   0,  -1,   5,  14,  16},
    { -1,  -1,  -1,  -1,  15,  17},
    {  0,  -1,   3,  -1,  16,  18},
    { -1,   1,  -1,   8,  26,  28},
    { -1,   2,  -1,   9,  27,  29},
    {  1,  -1,   6,  -1,  28,  30},
    {  2,  -1,   7,  -1,  29,  31},
    { -1,  -1,  -1,  12,  -1,  35},
    { -1,  -1,  -1,  13,  -1,  36},
    { -1,  -1,  10,  14,  35,  37},
    { -1,  -1,  11,  15,  36,  38},
    { -1,   3,  12,  16,  37,  39},
    { -1,   4,  13,  17,  38,  40},
    {  3,   5,  14,  18,  39,  41},
    {  4,  -1,  15,  19,  40,  42},
    {  5,  -1,  16,  20,  41,  43},
    { -1,  -1,  17,  21,  42,  44},
    { -1,  -1,  18,  22,  43,  45},
    { -1,  -1,  19,  -1,  44,  -1},
    { -1,  -1,  20,  -1,  45,  -1},
    { -1,  -1,  -1,  25,  -1,  46},
    { -1,  -1,  -1,  26,  -1,  47},
    { -1,  -1,  23,  27,  46,  48},
    { -1,   6,  24,  28,  47,  49},
    { -1,   7,  25,  29,  48,  50},
    {  6,   8,  26,  30,  49,  51},
    {  7,   9,  27,  31,  50,  52},
    {  8,  -1,  28,  32,  51,  53},
    {  9,  -1,  29,  33,  52,  54},
    { -1,  -1,  30,  34,  53,  55},
    { -1,  -1,  31,  -1,  54,  -1},
    { -1,  -1,  32,  -1,  55,  -1},
    { 10,  12,  -1,  37,  -1,  56},
    { 11,  13,  -1,  38,  -1,  57},
    { 12,  14,  35,  39,  56,  58},
    { 13,  15,  36,  40,  57,  59},
    { 14,  16,  37,  41,  58,  60},
    { 15,  17,  38,  42,  59,  61},
    { 16,  18,  39,  43,  60,  62},
    { 17,  19,  40,  44,  61,  63},
    { 18,  20,  41,  45,  62,  64},
    { 19,  21,  42,  -1,  63,  -1},
    { 20,  22,  43,  -1,  64,  -1},
    { 23,  25,  -1,  48,  -1,  66},
    { 24,  26,  -1,  49,  65,  67},
    { 25,  27,  46,  50,  66,  68},
    { 26,  28,  47,  51,  67,  69},
    { 27,  29,  48,  52,  68,  70},
    { 28,  30,  49,  53,  69,  71},
    { 29,  31,  50,  54,  70,  72},
    { 30,  32,  51,  55,  71,  73},
    { 31,  33,  52,  -1,  72,  74},
    { 32,  34,  53,  -1,  73,  -1},
    { 35,  37,  -1,  58,  75,  77},
    { 36,  38,  -1,  59,  76,  78},
    { 37,  39,  56,  60,  77,  79},
    { 38,  40,  57,  61,  78,  80},
    { 39,  41,  58,  62,  79,  81},
    { 40,  42,  59,  63,  80,  82},
    { 41,  43,  60,  64,  81,  83},
    { 42,  44,  61,  -1,  82,  84},
    { 43,  45,  62,  -1,  83,  85},
    { -1,  47,  -1,  67,  86,  88},
    { 46,  48,  -1,  68,  87,  89},
    { 47,  49,  65,  69,  88,  90},
    { 48,  50,  66,  70,  89,  91},
    { 49,  51,  67,  71,  90,  92},
    { 50,  52,  68,  72,  91,  93},
    { 51,  53,  69,  73,  92,  94},
    { 52,  54,  70,  74,  93,  95},
    { 53,  55,  71,  -1,  94,  96},
    { 54,  -1,  72,  -1,  95,  97},
    { -1,  56,  -1,  77,  98, 100},
    { -1,  57,  -1,  78,  99, 101},
    { 56,  58,  75,  79, 100, 102},
    { 57,  59,  76,  80, 101, 103},
    { 58,  60,  77,  81, 102, 104},
    { 59,  61,  78,  82, 103, 105},
    { 60,  62,  79,  83, 104, 106},
    { 61,  63,  80,  84, 105, 107},
    { 62,  64,  81,  85, 106, 108},
    { 63,  -1,  82,  -1, 107, 109},
    { 64,  -1,  83,  -1, 108, 110},
    { -1,  65,  -1,  88,  -1,  -1},
    { -1,  66,  -1,  89,  -1,  -1},
    { 65,  67,  86,  90,  -1,  -1},
    { 66,  68,  87,  91,  -1, 111},
    { 67,  69,  88,  92,  -1, 112},
    { 68,  70,  89,  93, 111, 113},
    { 69,  71,  90,  94, 112, 114},
    { 70,  72,  91,  95, 113,  -1},
    { 71,  73,  92,  96, 114,  -1},
    { 72,  74,  93,  97,  -1,  -1},
    { 73,  -1,  94,  -1,  -1,  -1},
    { 74,  -1,  95,  -1,  -1,  -1},
    { -1,  75,  -1, 100,  -1,  -1},
    { -1,  76,  -1, 101,  -1,  -1},
    { 75,  77,  98, 102,  -1,  -1},
    { 76,  78,  99, 103,  -1,  -1},
    { 77,  79, 100, 104,  -1, 115},
    { 78,  80, 101, 105,  -1, 116},
    { 79,  81, 102, 106, 115, 117},
    { 80,  82, 103, 107, 116,  -1},
    { 81,  83, 104, 108, 117,  -1},
    { 82,  84, 105, 109,  -1,  -1},
    { 83,  85, 106, 110,  -1,  -1},
    { 84,  -1, 107,  -1,  -1,  -1},
    { 85,  -1, 108,  -1,  -1,  -1},
    { 89,  91,  -1, 113,  -1, 118},
    { 90,  92,  -1, 114,  -1, 119},
    { 91,  93, 111,  -1, 118,  -1},
    { 92,  94, 112,  -1, 119,  -1},
    {102, 104,  -1, 117,  -1, 120},
    {103, 105,  -1,  -1,  -1,  -1},
    {104, 106, 115,  -1, 120,  -1},
    {111, 113,  -1,  -1,  -1,  -1},
    {112, 114,  -1,  -1,  -1,  -1},
    {115, 117,  -1,  -1,  -1,  -1}
};

// Triangle cells (6 triangles of 10 cells each).
// 0=North 1=NE 2=SE 3=South 4=SW 5=NW
const int kTriangleCells[kNumTriangles][kTriangleSize] = {
    {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9},
    { 19,  20,  21,  22,  32,  33,  34,  44,  45,  55},
    { 74,  84,  85,  95,  96,  97, 107, 108, 109, 110},
    {111, 112, 113, 114, 115, 116, 117, 118, 119, 120},
    { 65,  75,  76,  86,  87,  88,  98,  99, 100, 101},
    { 10,  11,  12,  13,  23,  24,  25,  35,  36,  46}
};

const std::vector<int>& PlayerSlots(int num_players) {
  switch (num_players) {
    case 2: return kSlots2;
    case 3: return kSlots3;
    case 4: return kSlots4;
    case 6: return kSlots6;
    default:
      SpielFatalError(absl::StrCat("Invalid number of players: ", num_players,
                                   ". Must be 2, 3, 4, or 6."));
  }
}

// --- ChineseCheckersState ---

ChineseCheckersState::ChineseCheckersState(std::shared_ptr<const Game> game,
                                           int num_players, int max_moves)
    : State(game),
      num_players_(num_players),
      max_moves_(max_moves),
      player_slots_(PlayerSlots(num_players)) {
  board_.fill(kEmpty);
  // Place pieces in home triangles.
  for (int p = 0; p < num_players_; ++p) {
    int tri = player_slots_[p];
    for (int i = 0; i < kTriangleSize; ++i) {
      board_[kTriangleCells[tri][i]] = p;
    }
  }
}

Player ChineseCheckersState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  return current_player_;
}

bool ChineseCheckersState::HasContinuationHops(int pos) const {
  for (int d = 0; d < kNumDirections; ++d) {
    int neighbor = kNeighbor[pos][d];
    if (neighbor < 0 || board_[neighbor] == kEmpty) continue;
    int landing = kHopDest[pos][d];
    if (landing < 0 || board_[landing] != kEmpty) continue;
    if (visited_.count(landing)) continue;
    return true;
  }
  return false;
}

std::vector<Action> ChineseCheckersState::LegalActions() const {
  if (IsTerminal()) return {};

  std::vector<Action> actions;

  if (hop_from_ >= 0) {
    // In a hop chain: only continuation hops from hop_from_ plus pass.
    for (int d = 0; d < kNumDirections; ++d) {
      int neighbor = kNeighbor[hop_from_][d];
      if (neighbor < 0 || board_[neighbor] == kEmpty) continue;
      int landing = kHopDest[hop_from_][d];
      if (landing < 0 || board_[landing] != kEmpty) continue;
      if (visited_.count(landing)) continue;
      actions.push_back(hop_from_ * kNumDirections + d);
    }
    actions.push_back(kPassAction);
    return actions;
  }

  // Normal turn: any piece of current player can step or hop.
  for (int pos = 0; pos < kNumPositions; ++pos) {
    if (board_[pos] != current_player_) continue;
    for (int d = 0; d < kNumDirections; ++d) {
      int neighbor = kNeighbor[pos][d];
      if (neighbor < 0) continue;
      if (board_[neighbor] == kEmpty) {
        // Step move.
        actions.push_back(pos * kNumDirections + d);
      } else {
        // Potential hop.
        int landing = kHopDest[pos][d];
        if (landing >= 0 && board_[landing] == kEmpty) {
          actions.push_back(pos * kNumDirections + d);
        }
      }
    }
  }

  // If completely blocked (extremely rare), allow pass.
  if (actions.empty()) {
    actions.push_back(kPassAction);
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

void ChineseCheckersState::AdvanceTurn() {
  Player mover = current_player_;
  hop_from_ = -1;
  visited_.clear();
  current_player_ = (current_player_ + 1) % num_players_;
  total_moves_++;
  if (CheckWinner(mover)) {
    outcome_ = mover;
  }
}

void ChineseCheckersState::DoApplyAction(Action action) {
  // Save undo info.
  UndoInfo info;
  info.prev_hop_from = hop_from_;
  info.prev_current_player = current_player_;
  info.prev_total_moves = total_moves_;
  info.prev_outcome = outcome_;
  info.prev_visited = visited_;

  if (action == kPassAction) {
    info.piece_origin = -1;
    info.piece_dest = -1;
    undo_stack_.push_back(std::move(info));
    AdvanceTurn();
    return;
  }

  int source = action / kNumDirections;
  int dir = action % kNumDirections;
  int neighbor = kNeighbor[source][dir];

  if (hop_from_ >= 0) {
    // Continuing hop chain.
    SPIEL_CHECK_EQ(source, hop_from_);
    int landing = kHopDest[source][dir];
    SPIEL_CHECK_GE(landing, 0);
    SPIEL_CHECK_EQ(board_[landing], kEmpty);

    info.piece_origin = source;
    info.piece_dest = landing;
    undo_stack_.push_back(std::move(info));

    board_[landing] = current_player_;
    board_[source] = kEmpty;
    visited_.insert(landing);
    hop_from_ = landing;

    if (!HasContinuationHops(landing)) {
      AdvanceTurn();
    }
  } else if (board_[neighbor] == kEmpty) {
    // Step move.
    info.piece_origin = source;
    info.piece_dest = neighbor;
    undo_stack_.push_back(std::move(info));

    board_[neighbor] = current_player_;
    board_[source] = kEmpty;
    AdvanceTurn();
  } else {
    // Hop (start of potential chain).
    int landing = kHopDest[source][dir];
    SPIEL_CHECK_GE(landing, 0);
    SPIEL_CHECK_EQ(board_[landing], kEmpty);

    info.piece_origin = source;
    info.piece_dest = landing;
    undo_stack_.push_back(std::move(info));

    board_[landing] = current_player_;
    board_[source] = kEmpty;
    visited_.clear();
    visited_.insert(source);
    visited_.insert(landing);
    hop_from_ = landing;

    if (!HasContinuationHops(landing)) {
      AdvanceTurn();
    }
  }
}

void ChineseCheckersState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_FALSE(undo_stack_.empty());
  UndoInfo info = std::move(undo_stack_.back());
  undo_stack_.pop_back();

  if (info.piece_origin >= 0) {
    board_[info.piece_origin] = player;
    board_[info.piece_dest] = kEmpty;
  }

  hop_from_ = info.prev_hop_from;
  current_player_ = info.prev_current_player;
  total_moves_ = info.prev_total_moves;
  outcome_ = info.prev_outcome;
  visited_ = std::move(info.prev_visited);

  history_.pop_back();
  --move_number_;
}

bool ChineseCheckersState::CheckWinner(Player player) const {
  int target_tri = TargetTriangle(player_slots_[player]);
  for (int i = 0; i < kTriangleSize; ++i) {
    if (board_[kTriangleCells[target_tri][i]] != player) return false;
  }
  return true;
}

bool ChineseCheckersState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || total_moves_ >= max_moves_;
}

std::vector<double> ChineseCheckersState::Returns() const {
  std::vector<double> result(num_players_, 0.0);
  if (outcome_ != kInvalidPlayer) {
    for (int p = 0; p < num_players_; ++p) {
      result[p] = (p == outcome_) ? (num_players_ - 1.0) : -1.0;
    }
  }
  return result;
}

std::string ChineseCheckersState::ActionToString(Player player,
                                                  Action action_id) const {
  if (action_id == kPassAction) return "Pass";
  int source = action_id / kNumDirections;
  int dir = action_id % kNumDirections;
  const char* dir_names[] = {"UL", "UR", "L", "R", "DL", "DR"};
  int neighbor = kNeighbor[source][dir];
  if (neighbor >= 0 && board_[neighbor] == kEmpty) {
    return absl::StrCat(source, "->", neighbor);
  }
  int landing = kHopDest[source][dir];
  if (landing >= 0) {
    return absl::StrCat(source, "=>", landing);
  }
  return absl::StrCat(source, "-", dir_names[dir]);
}

std::string ChineseCheckersState::ToString() const {
  // Render the star board row by row.
  // Each row's cells are rendered at their doubled column coordinate.
  std::string result;
  int idx = 0;
  for (int r = 0; r < kNumRows; ++r) {
    std::string line(25, ' ');
    while (idx < kNumPositions && kCellRow[idx] == r) {
      int col = kCellCol[idx];
      if (board_[idx] == kEmpty) {
        line[col] = '.';
      } else {
        line[col] = '1' + board_[idx];  // Players shown as 1-6.
      }
      ++idx;
    }
    // Trim trailing spaces.
    while (!line.empty() && line.back() == ' ') line.pop_back();
    absl::StrAppend(&result, line, "\n");
  }
  absl::StrAppend(&result, "Player: ", current_player_);
  if (hop_from_ >= 0) {
    absl::StrAppend(&result, " (hopping from ", hop_from_, ")");
  }
  return result;
}

std::string ChineseCheckersState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ChineseCheckersState::ObservationTensor(Player player,
                                              absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  int tensor_size = (num_players_ + 1) * kNumPositions + num_players_;
  SPIEL_CHECK_EQ(static_cast<int>(values.size()), tensor_size);
  std::fill(values.begin(), values.end(), 0.0f);

  // Board planes: num_players channels for pieces + 1 channel for empty.
  // Egocentric rotation: observing player is channel 0.
  for (int pos = 0; pos < kNumPositions; ++pos) {
    if (board_[pos] == kEmpty) {
      values[num_players_ * kNumPositions + pos] = 1.0f;
    } else {
      int rotated = (board_[pos] - player + num_players_) % num_players_;
      values[rotated * kNumPositions + pos] = 1.0f;
    }
  }

  // Current player one-hot (rotated).
  int offset = (num_players_ + 1) * kNumPositions;
  int rotated_current = (
      current_player_ - player + num_players_) % num_players_;
  values[offset + rotated_current] = 1.0f;
}

std::unique_ptr<State> ChineseCheckersState::Clone() const {
  return std::unique_ptr<State>(new ChineseCheckersState(*this));
}

// --- ChineseCheckersGame ---

ChineseCheckersGame::ChineseCheckersGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      max_moves_(ParameterValue<int>("max_moves")) {
  SPIEL_CHECK_TRUE(num_players_ == 2 || num_players_ == 3 ||
                   num_players_ == 4 || num_players_ == 6);
}

}  // namespace chinese_checkers
}  // namespace open_spiel
