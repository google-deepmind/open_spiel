// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_GIN_RUMMY_UTILS_H_
#define OPEN_SPIEL_GAMES_GIN_RUMMY_UTILS_H_

#include <map>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"

namespace open_spiel {
namespace gin_rummy {

inline constexpr int kNumSuits = 4;
inline constexpr int kNumRanks = 13;
inline constexpr int kNumCards = kNumSuits * kNumRanks;
inline constexpr int kMaxHandSize = 11;

using VecInt = std::vector<int>;
using VecVecInt = std::vector<std::vector<int>>;
using VecVecVecInt = std::vector<std::vector<std::vector<int>>>;

std::string CardString(absl::optional<int> card);
std::string HandToString(const VecInt &cards);

int CardInt(std::string card);

std::vector<std::string> CardIntsToCardStrings(const VecInt &cards);
VecInt CardStringsToCardInts(const std::vector<std::string> &cards);

int CardValue(int card_index);
int TotalCardValue(const VecInt &cards);
int TotalCardValue(const VecVecInt &meld_group);
int CardRank(const int card_index);
int CardSuit(const int card_index);

bool CompareRanks(int card_1, int card_2);
bool CompareSuits(int card_1, int card_2);

bool IsRankMeld(const VecInt &cards);
bool IsSuitMeld(const VecInt &cards);

VecVecInt RankMelds(VecInt cards);
VecVecInt SuitMelds(VecInt cards);
VecVecInt AllMelds(const VecInt &cards);

bool VectorsIntersect(VecInt *v1, VecInt *v2);

VecVecInt NonOverlappingMelds(VecInt *meld, VecVecInt *melds);

void AllPaths(VecInt *meld, VecVecInt *all_melds, VecVecInt *path,
              VecVecVecInt *all_paths);

VecVecVecInt AllMeldGroups(const VecInt &cards);

VecVecInt BestMeldGroup(const VecInt &cards);

int MinDeadwood(VecInt hand, absl::optional<int> card);
int MinDeadwood(const VecInt &hand);

int RankMeldLayoff(const VecInt &meld);
VecInt SuitMeldLayoffs(const VecInt &meld);

VecInt LegalMelds(const VecInt &hand, int knock_card);
VecInt LegalDiscards(const VecInt &hand, int knock_card);

VecInt AllLayoffs(const VecInt &layed_melds, const VecInt &previous_layoffs);

int MeldToInt(VecInt meld);

std::map<VecInt, int> BuildMeldToIntMap();
std::map<int, VecInt> BuildIntToMeldMap();

static const std::map<int, VecInt> int_to_meld = BuildIntToMeldMap();
static const std::map<VecInt, int> meld_to_int = BuildMeldToIntMap();

}  // namespace gin_rummy
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GIN_RUMMY_UTILS_H_
