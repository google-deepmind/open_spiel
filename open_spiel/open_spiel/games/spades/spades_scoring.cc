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

#include "open_spiel/games/spades/spades_scoring.h"

#include <array>
namespace open_spiel {
namespace spades {
namespace {

// Score from contract is 10 times the bid (make contract arg negative if
// failed)
int ScoreContract(int contract) { return contract * 10; }

// Penalty for accumulating 10 bags (-100 per instance)
int ScoreBagPenalties(int current_score, int overtricks) {
  int current_bags = current_score % 10;
  current_bags += overtricks;
  return -100 * (current_bags / 10);
}

// Bonus/penalty for succeeding/failing a Nil bid
int ScoreNil(int tricks) { return (tricks > 0) ? -100 : 100; }
}  // namespace

std::array<int, kNumPartnerships> Score(
    const std::array<int, kNumPlayers> contracts,
    const std::array<int, kNumPlayers> taken_tricks,
    const std::array<int, kNumPartnerships> current_scores) {
  std::array<int, kNumPartnerships> round_scores = {0, 0};

  for (int pship = 0; pship < kNumPartnerships; ++pship) {
    int contract = contracts[pship] + contracts[pship + 2];
    int contract_result =
        (taken_tricks[pship] + taken_tricks[pship + 2]) - contract;
    int bonuses = 0;
    int contract_score = 0;

    // Score any nils
    if (contracts[pship] == 0) {
      bonuses += ScoreNil(taken_tricks[pship]);
    }
    if (contracts[pship + 2] == 0) {
      bonuses += ScoreNil(taken_tricks[pship + 2]);
    }

    // Score contracts and check for bag penalties
    if (contract_result < 0) {
      contract_score = ScoreContract(-contract);
    } else {
      contract_score = ScoreContract(contract);

      bonuses += contract_result +  // Each overtrick (bag) is worth 1 point
                 ScoreBagPenalties(current_scores[pship], contract_result);
    }

    round_scores[pship] = contract_score + bonuses;
  }

  return round_scores;
}

}  // namespace spades
}  // namespace open_spiel
