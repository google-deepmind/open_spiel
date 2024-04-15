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

#include "open_spiel/games/nim/nim.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace nim {
namespace {

constexpr char kDefaultPileSizes[] = "1;3;5;7";

std::vector<int> ParsePilesString(const std::string &str) {
  std::vector<std::string> sizes = absl::StrSplit(str, ';');
  std::vector<int> pile_sizes;
  for (const auto &sz : sizes) {
    int val;
    if (!absl::SimpleAtoi(sz, &val)) {
      SpielFatalError(absl::StrCat("Could not parse size '", sz,
                                   "' of pile_sizes string '", str,
                                   "' as an integer"));
    }
    pile_sizes.push_back(val);
  }
  return pile_sizes;
}

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"nim",
    /*long_name=*/"Nim",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {
        {"pile_sizes", GameParameter(std::string(kDefaultPileSizes))},
        {"is_misere", GameParameter(kDefaultIsMisere)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new NimGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

NimGame::NimGame(const GameParameters &params)
    : Game(kGameType, params),
      piles_(ParsePilesString(ParameterValue<std::string>("pile_sizes"))),
      is_misere_(ParameterValue<bool>("is_misere")) {
  num_piles_ = piles_.size();
  max_num_per_pile_ = *std::max_element(piles_.begin(), piles_.end());
}

int NimGame::NumDistinctActions() const {
  if (piles_.empty()) {
    return 0;
  }
  // action_id = (take - 1) * num_piles_ + pile_idx < (max_take - 1) *
  // num_piles_ + num_piles = max_take * num_piles_
  return num_piles_ * max_num_per_pile_ + 1;
}

int NimGame::MaxGameLength() const {
  // players can take only 1 object at every step
  return std::accumulate(piles_.begin(), piles_.end(), 0);
}

std::pair<int, int> NimState::UnpackAction(Action action_id) const {
  // action_id = (take - 1) * num_piles_ + pile_idx
  int pile_idx = action_id % num_piles_;
  int take = (action_id - pile_idx) / num_piles_ + 1;
  return {pile_idx, take};
}

bool NimState::IsEmpty() const {
  return std::accumulate(piles_.begin(), piles_.end(), 0) == 0;
}

void NimState::DoApplyAction(Action move) {
  SPIEL_CHECK_FALSE(IsTerminal());
  std::pair<int, int> action = UnpackAction(move);
  int pile_idx = action.first, take = action.second;

  SPIEL_CHECK_LT(pile_idx, piles_.size());
  SPIEL_CHECK_GT(take, 0);
  SPIEL_CHECK_LE(take, piles_[pile_idx]);

  piles_[pile_idx] -= take;
  if (IsEmpty()) {
    outcome_ = is_misere_ ? 1 - current_player_ : current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> NimState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  for (std::size_t pile_idx = 0; pile_idx < piles_.size(); pile_idx++) {
    // the player has to take at least one object from a pile
    for (int take = 1; take <= piles_[pile_idx]; take++) {
      moves.push_back((take - 1) * num_piles_ + (int)pile_idx);
    }
  }
  std::sort(moves.begin(), moves.end());
  return moves;
}

std::string NimState::ActionToString(Player player, Action action_id) const {
  std::pair<int, int> action = UnpackAction(action_id);
  int pile_idx = action.first, take = action.second;
  return absl::StrCat("pile:", pile_idx + 1, ", take:", take, ";");
}

NimState::NimState(std::shared_ptr<const Game> game, int num_piles,
                   std::vector<int> piles, bool is_misere,
                   int max_num_per_pile)
    : State(game),
      num_piles_(num_piles),
      piles_(piles),
      is_misere_(is_misere),
      max_num_per_pile_(max_num_per_pile) {}

std::string NimState::ToString() const {
  std::string str;
  absl::StrAppend(&str, "(", current_player_, "): ");
  for (std::size_t pile_idx = 0; pile_idx < piles_.size(); pile_idx++) {
    absl::StrAppend(&str, piles_[pile_idx]);
    if (pile_idx != piles_.size() - 1) {
      absl::StrAppend(&str, " ");
    }
  }
  return str;
}

bool NimState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsEmpty();
}

std::vector<double> NimState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string NimState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string NimState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void NimState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  // [one-hot player] + [IsTerminal()] + [binary representation of num_piles] +
  // [binary representation of every pile]
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0);

  int offset = 0;
  values[current_player_] = 1;
  offset += 2;
  values[offset] = IsTerminal() ? 1 : 0;
  offset += 1;

  // num_piles (which is >= 1)
  values[offset + num_piles_ - 1] = 1;
  offset += num_piles_;

  for (std::size_t pile_idx = 0; pile_idx < piles_.size(); pile_idx++) {
    values[offset + piles_[pile_idx]] = 1;
    offset += max_num_per_pile_ + 1;
  }

  SPIEL_CHECK_EQ(offset, values.size());
}

void NimState::UndoAction(Player player, Action move) {
  std::pair<int, int> action = UnpackAction(move);
  int pile_idx = action.first, take = action.second;
  piles_[pile_idx] += take;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> NimState::Clone() const {
  return std::unique_ptr<State>(new NimState(*this));
}

}  // namespace nim
}  // namespace open_spiel
