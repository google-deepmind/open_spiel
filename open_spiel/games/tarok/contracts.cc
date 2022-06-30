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

#include "open_spiel/games/tarok/contracts.h"

namespace open_spiel {
namespace tarok {

Contract::Contract(ContractName name, int score, int num_talon_exchanges,
                   bool needs_king_calling, bool declarer_starts,
                   bool is_negative)
    : name(name),
      score(score),
      num_talon_exchanges(num_talon_exchanges),
      needs_king_calling(needs_king_calling),
      declarer_starts(declarer_starts),
      is_negative(is_negative) {}

bool Contract::NeedsTalonExchange() const { return num_talon_exchanges > 0; }

const std::array<Contract, 12> InitializeContracts() {
  return {
      Contract(ContractName::kKlop, 70, 0, false, false, true),
      Contract(ContractName::kThree, 10, 3, true, false, false),
      Contract(ContractName::kTwo, 20, 2, true, false, false),
      Contract(ContractName::kOne, 30, 1, true, false, false),
      Contract(ContractName::kSoloThree, 40, 3, false, false, false),
      Contract(ContractName::kSoloTwo, 50, 2, false, false, false),
      Contract(ContractName::kSoloOne, 60, 1, false, false, false),
      Contract(ContractName::kBeggar, 70, 0, false, true, true),
      Contract(ContractName::kSoloWithout, 80, 0, false, true, false),
      Contract(ContractName::kOpenBeggar, 90, 0, false, true, true),
      Contract(ContractName::kColourValatWithout, 125, 0, false, true, false),
      Contract(ContractName::kValatWithout, 500, 0, false, true, false)};
}

std::ostream& operator<<(std::ostream& os, const ContractName& contract_name) {
  os << ContractNameToString(contract_name);
  return os;
}

std::string ContractNameToString(const ContractName& contract_name) {
  switch (contract_name) {
    case ContractName::kKlop:
      return "Klop";
    case ContractName::kThree:
      return "Three";
    case ContractName::kTwo:
      return "Two";
    case ContractName::kOne:
      return "One";
    case ContractName::kSoloThree:
      return "Solo three";
    case ContractName::kSoloTwo:
      return "Solo two";
    case ContractName::kSoloOne:
      return "Solo one";
    case ContractName::kBeggar:
      return "Beggar";
    case ContractName::kSoloWithout:
      return "Solo without";
    case ContractName::kOpenBeggar:
      return "Open beggar";
    case ContractName::kColourValatWithout:
      return "Colour valat without";
    case ContractName::kValatWithout:
      return "Valat without";
    case ContractName::kNotSelected:
      return "Not selected";
  }
}

}  // namespace tarok
}  // namespace open_spiel
