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

#ifndef OPEN_SPIEL_GAMES_TAROK_CONTRACTS_H_
#define OPEN_SPIEL_GAMES_TAROK_CONTRACTS_H_

#include <array>
#include <iostream>
#include <string>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tarok {

// a subset of bidding actions that are used throughout the codebase and add to
// readability, for more info see TarokState::LegalActionsInBidding()
inline constexpr int kInvalidBidAction = -1;
inline constexpr int kBidPassAction = 0;
inline constexpr int kBidKlopAction = 1;
inline constexpr int kBidThreeAction = 2;
inline constexpr int kBidSoloThreeAction = 5;
inline constexpr int kBidSoloOneAction = 7;

enum class ContractName {
  kKlop,
  kThree,
  kTwo,
  kOne,
  kSoloThree,
  kSoloTwo,
  kSoloOne,
  kBeggar,
  kSoloWithout,
  kOpenBeggar,
  kColourValatWithout,
  kValatWithout,
  kNotSelected
};

struct Contract {
  Contract(ContractName name, int score, int num_talon_exchanges,
           bool needs_king_calling, bool declarer_starts, bool is_negative);

  bool NeedsTalonExchange() const;

  const ContractName name;
  const int score;
  const int num_talon_exchanges;
  const bool needs_king_calling;
  const bool declarer_starts;
  const bool is_negative;
};

const std::array<Contract, 12> InitializeContracts();

std::ostream& operator<<(std::ostream& os, const ContractName& contract_name);

std::string ContractNameToString(const ContractName& contract_name);

}  // namespace tarok
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TAROK_CONTRACTS_H_
