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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_BRIDGE_BRIDGE_SCORING_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_BRIDGE_BRIDGE_SCORING_H_

// Scoring for (duplicate) contract bridge.
// See Law 77 of the Laws of Bridge, 2017:
// http://www.worldbridge.org/wp-content/uploads/2017/03/2017LawsofDuplicateBridge-paginated.pdf

namespace open_spiel {
namespace bridge {

enum Suit { kClubs = 0, kDiamonds, kHearts, kSpades, kNone };
enum DoubleStatus { kUndoubled = 1, kDoubled = 2, kRedoubled = 4 };

struct Contract {
  int level;
  Suit trumps;
  DoubleStatus double_status;
  int declarer;
};

int Score(Contract contract, int declarer_tricks, bool is_vulnerable);

}  // namespace bridge
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_BRIDGE_BRIDGE_SCORING_H_
