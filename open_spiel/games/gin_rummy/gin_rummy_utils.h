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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_GIN_RUMMY_UTILS_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_GIN_RUMMY_UTILS_H_

#include <vector>
#include <string>
#include <map>

namespace open_spiel {
namespace gin_rummy {

std::string CardString(int card);
std::string HandToString(const std::vector<int> &cards);

int CardInt(std::string card);

std::vector<std::string> CardIntsToCardStrings(const std::vector<int> &cards);
std::vector<int> CardStringsToCardInts(const std::vector<std::string> &cards);

int CardValue(int card_index);
int TotalCardValue(const std::vector<int> &cards);
int TotalCardValue(const std::vector<std::vector<int>> &meld_group);
int CardRank(const int card_index);
int CardSuit(const int card_index);

bool CompareRanks(int card_1, int card_2);
bool CompareSuits(int card_1, int card_2);

bool IsRankMeld(const std::vector<int> &cards);
bool IsSuitMeld(const std::vector<int> &cards);

std::vector<std::vector<int>> RankMelds(std::vector<int> cards);
std::vector<std::vector<int>> SuitMelds(std::vector<int> cards);
std::vector<std::vector<int>> AllMelds(const std::vector<int> &cards);

bool VectorsIntersect(std::vector<int> *v1, std::vector<int> *v2);

std::vector<std::vector<int>> NonOverlappingMelds(
    std::vector<int> *meld, std::vector<std::vector<int>> *melds);

void AllPaths(std::vector<int> *meld,
              std::vector<std::vector<int>> *all_melds,
              std::vector<std::vector<int>> *path,
              std::vector<std::vector<std::vector<int>>> *all_paths);

std::vector<std::vector<std::vector<int>>> AllMeldGroups(
    const std::vector<int> &cards);

std::vector<std::vector<int>> BestMeldGroup(const std::vector<int> &cards);

int MinDeadwood(const std::vector<int> &hand, int card);
int MinDeadwood(const std::vector<int> &hand);

int RankMeldLayoff(const std::vector<int> &meld);
std::vector<int> SuitMeldLayoffs(const std::vector<int> &meld);

std::vector<int> LegalMelds(const std::vector<int> &hand, int knock_card);
std::vector<int> LegalDiscards(const std::vector<int> &hand, int knock_card);

std::vector<int> AllLayoffs(const std::vector<int> &layed_melds,
                            const std::vector<int> &previous_layoffs);


int MeldToInt(std::vector<int> meld);

std::map<std::vector<int>, int> BuildMeldToIntMap();
std::map<int, std::vector<int>> BuildIntToMeldMap();

static const std::map<int, std::vector<int>> int_to_meld = BuildIntToMeldMap();
static const std::map<std::vector<int>, int> meld_to_int = BuildMeldToIntMap();

}  // namespace gin_rummy
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_GIN_RUMMY_UTILS_H_

