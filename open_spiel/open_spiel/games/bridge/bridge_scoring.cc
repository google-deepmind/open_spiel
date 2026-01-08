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

#include "open_spiel/games/bridge/bridge_scoring.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"

namespace open_spiel {
namespace bridge {
namespace {
constexpr int kBaseTrickScores[] = {20, 20, 30, 30, 30};

int ScoreContract(Contract contract, DoubleStatus double_status) {
  int score = contract.level * kBaseTrickScores[contract.trumps];
  if (contract.trumps == kNoTrump) score += 10;
  return score * double_status;
}

// Score for failing to make the contract (will be negative).
int ScoreUndertricks(int undertricks, bool is_vulnerable,
                     DoubleStatus double_status) {
  if (double_status == kUndoubled) {
    return (is_vulnerable ? -100 : -50) * undertricks;
  }
  int score = 0;
  if (is_vulnerable) {
    score = -200 - 300 * (undertricks - 1);
  } else {
    if (undertricks == 1) {
      score = -100;
    } else if (undertricks == 2) {
      score = -300;
    } else {
      // This takes into account the -100 for the fourth and subsequent tricks.
      score = -500 - 300 * (undertricks - 3);
    }
  }
  return score * (double_status / 2);
}

// Score for tricks made in excess of the bid.
int ScoreOvertricks(Denomination trump_suit, int overtricks, bool is_vulnerable,
                    DoubleStatus double_status) {
  if (double_status == kUndoubled) {
    return overtricks * kBaseTrickScores[trump_suit];
  } else {
    return (is_vulnerable ? 100 : 50) * overtricks * double_status;
  }
}

// Bonus for making a doubled or redoubled contract.
int ScoreDoubledBonus(DoubleStatus double_status) {
  return 50 * (double_status / 2);
}

// Bonuses for partscore, game, or slam.
int ScoreBonuses(int level, int contract_score, bool is_vulnerable) {
  if (level == 7) {  // 1500/1000 for grand slam + 500/300 for game
    return is_vulnerable ? 2000 : 1300;
  } else if (level == 6) {  // 750/500 for small slam + 500/300 for game
    return is_vulnerable ? 1250 : 800;
  } else if (contract_score >= 100) {  // game bonus
    return is_vulnerable ? 500 : 300;
  } else {  // partscore bonus
    return 50;
  }
}
}  // namespace

int Score(Contract contract, int declarer_tricks, bool is_vulnerable) {
  if (contract.level == 0) return 0;
  int contracted_tricks = 6 + contract.level;
  int contract_result = declarer_tricks - contracted_tricks;
  if (contract_result < 0) {
    return ScoreUndertricks(-contract_result, is_vulnerable,
                            contract.double_status);
  } else {
    int contract_score = ScoreContract(contract, contract.double_status);
    int bonuses = ScoreBonuses(contract.level, contract_score, is_vulnerable) +
                  ScoreDoubledBonus(contract.double_status) +
                  ScoreOvertricks(contract.trumps, contract_result,
                                  is_vulnerable, contract.double_status);
    return contract_score + bonuses;
  }
}

std::string Contract::ToString() const {
  if (level == 0) return "Passed Out";
  std::string str = absl::StrCat(level, std::string{kDenominationChar[trumps]});
  if (double_status == kDoubled) absl::StrAppend(&str, "X");
  if (double_status == kRedoubled) absl::StrAppend(&str, "XX");
  absl::StrAppend(&str, " ", std::string{kPlayerChar[declarer]});
  return str;
}

int Contract::Index() const {
  if (level == 0) return 0;
  int index = level - 1;
  index *= kNumDenominations;
  index += static_cast<int>(trumps);
  index *= kNumPlayers;
  index += static_cast<int>(declarer);
  index *= kNumDoubleStates;
  if (double_status == kRedoubled) index += 2;
  if (double_status == kDoubled) index += 1;
  return index + 1;
}

}  // namespace bridge
}  // namespace open_spiel
