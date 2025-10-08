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

#include "open_spiel/games/universal_poker/repeated_poker.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/strip.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace repeated_poker {
namespace {

const GameType kGameType{
    /*short_name=*/"repeated_poker",
    /*long_name=*/"Repeated Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {
      {"universal_poker_game_string",
       GameParameter(GameParameter::Type::kGame, true)},
      {"max_num_hands", GameParameter(GameParameter::Type::kInt, true)},
      {"reset_stacks", GameParameter(GameParameter::Type::kBool, true)},
      {"rotate_dealer", GameParameter(true)},
      {"blind_schedule", GameParameter(GameParameter::Type::kString)},
    },
    /*default_loadable=*/false
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new RepeatedPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Parses blind schedule string of the form
// <blind_level_1>;...;<blind_level_n>
// where each blind level is of the form
// <num_hands>:<small_blind>/<big_blind>
std::vector<BlindLevel> ParseBlindSchedule(
    const std::string& blind_schedule_str) {
  std::vector<BlindLevel> blind_levels;
  if (blind_schedule_str.empty()) {
    return blind_levels;
  }
  std::vector<std::string> levels = absl::StrSplit(
      absl::StripSuffix(blind_schedule_str, ";"), ';');
  for (const auto& level : levels) {
    std::vector<std::string> parts = absl::StrSplit(level, ':');
    SPIEL_CHECK_EQ(parts.size(), 2);
    std::vector<std::string> blinds = absl::StrSplit(parts[1], '/');
    SPIEL_CHECK_EQ(parts.size(), 2);
    blind_levels.push_back({
        .num_hands = std::stoi(parts[0]),
        .small_blind = std::stoi(blinds[0]),
        .big_blind = std::stoi(blinds[1]),
    });
  }
  return blind_levels;
}

}  // namespace

RepeatedPokerState::RepeatedPokerState(
    std::shared_ptr<const Game> game,
    std::string universal_poker_game_string,
    int max_num_hands,
    bool reset_stacks,
    bool rotate_dealer,
    std::string blind_schedule)
    : State(game),
      universal_poker_game_string_(universal_poker_game_string),
      max_num_hands_(max_num_hands),
      reset_stacks_(reset_stacks),
      rotate_dealer_(rotate_dealer),
      blind_schedule_str_(blind_schedule) {
  // Load the underlying universal_poker game and retrieve the multiple ACPC
  // wrappers.
  std::shared_ptr<const Game> universal_poker_game =
      LoadGame(universal_poker_game_string_);
  universal_poker_game_params_ = universal_poker_game->GetParameters();
  universal_poker_state_ = std::make_unique<UniversalPokerState>(
      *dynamic_cast<UniversalPokerState*>(
          universal_poker_game->NewInitialState().get()));
  const acpc_cpp::ACPCState& acpc_state = universal_poker_state_->acpc_state();
  const acpc_cpp::ACPCGame* acpc_game = acpc_state.game();
  const project_acpc_server::Game& raw_acpc_game = acpc_game->Game();
  // Initial setup logic.
  num_active_players_ = universal_poker_game->NumPlayers();
  hand_number_ = 0;
  SPIEL_CHECK_GE(max_num_hands_, 1);
  for (Player i = 0; i < num_players_; ++i) {
    stacks_.push_back(acpc_game->StackSize(i));
    player_to_seat_[i] = i;
    seat_to_player_[i] = i;
  }
  // Action always begins to the left of the dealer after the flop.
  SPIEL_CHECK_GE(acpc_game->NumRounds(), 2);
  dealer_ = ((raw_acpc_game.firstPlayer[1] - 1) % num_players_ +
             num_players_) % num_players_;
  for (int round_num = 2; round_num < acpc_game->NumRounds(); ++round_num) {
    // ACPC does not enforce a consistent dealer position for all rounds after
    // the flop, but we do, as this is always the case in practice.
    SPIEL_CHECK_EQ(raw_acpc_game.firstPlayer[1],
                   raw_acpc_game.firstPlayer[round_num]);
  }
  blind_schedule_levels_ = ParseBlindSchedule(blind_schedule_str_);
  if (!blind_schedule_str_.empty() && blind_schedule_levels_.empty()) {
    SpielFatalError("Failed to parse blind schedule.");
  }
  if (blind_schedule_levels_.empty()) {
    // Identify the small and big blinds from the underlying ACPC game.
    std::vector<int> blinds{acpc_game->blinds().begin(),
                            acpc_game->blinds().end()};
    int num_blinds = 0;
    for (int blind : blinds) {
      if (blind > 0) {
        num_blinds++;
        if (small_blind_ == kInvalidBlindValue) {
          small_blind_ = blind;
        } else {
          big_blind_ = std::max(small_blind_, blind);
          small_blind_ = std::min(small_blind_, blind);
        }
      }
    }
    SPIEL_CHECK_EQ(num_blinds, 2);
    SPIEL_CHECK_GT(small_blind_, 0);
    SPIEL_CHECK_GT(big_blind_, 0);
  }
  UpdateBlinds();
  UpdateUniversalPoker();
}

// Must manually define copy constructor because we're storing a unique_ptr to
// the underlying UniversalPokerState, and Clone() only returns a pointer to the
// State interface.
RepeatedPokerState::RepeatedPokerState(const RepeatedPokerState& other)
    : State(other),
      universal_poker_game_string_(other.universal_poker_game_string_),
      universal_poker_game_params_(other.universal_poker_game_params_),
      universal_poker_state_(std::unique_ptr<UniversalPokerState>(
          dynamic_cast<UniversalPokerState*>(
              other.universal_poker_state_->Clone().release()))),
      hand_number_(other.hand_number_),
      max_num_hands_(other.max_num_hands_),
      is_terminal_(other.is_terminal_),
      stacks_(other.stacks_),
      reset_stacks_(other.reset_stacks_),
      rotate_dealer_(other.rotate_dealer_),
      blind_schedule_str_(other.blind_schedule_str_),
      blind_schedule_levels_(other.blind_schedule_levels_),
      player_to_seat_(other.player_to_seat_),
      dealer_(other.dealer_),
      seat_to_player_(other.seat_to_player_),
      num_active_players_(other.num_active_players_),
      small_blind_(other.small_blind_),
      big_blind_(other.big_blind_),
      small_blind_seat_(other.small_blind_seat_),
      big_blind_seat_(other.big_blind_seat_),
      acpc_hand_histories_(other.acpc_hand_histories_),
      hand_returns_(other.hand_returns_) {}

void RepeatedPokerState::UpdateStacks() {
  if (reset_stacks_) return;
  for (Player player_id = 0; player_id < num_players_; ++player_id) {
    int seat = player_to_seat_.at(player_id);
    if (seat != kInactivePlayerSeat) {
      stacks_[player_id] += universal_poker_state_->GetTotalReward(seat);
    }
  }
}

void RepeatedPokerState::UpdateSeatAssignments() {
  if (reset_stacks_) return;
  player_to_seat_.clear();
  seat_to_player_.clear();
  int next_open_seat = 0;
  for (Player player_id = 0; player_id < num_players_; ++player_id) {
    if (stacks_[player_id] < big_blind_) {
      player_to_seat_[player_id] = kInactivePlayerSeat;
    } else {
      player_to_seat_[player_id] = next_open_seat;
      seat_to_player_[next_open_seat] = player_id;
      next_open_seat++;
    }
  }
  num_active_players_ = next_open_seat;
}

void RepeatedPokerState::UpdateDealer() {
  if (!rotate_dealer_) return;
  dealer_ = (dealer_ + 1) % num_players_;
  while (player_to_seat_.at(dealer_) == kInactivePlayerSeat) {
    dealer_ = (dealer_ + 1) % num_players_;
  }
}

void RepeatedPokerState::UpdateBlinds() {
  // Update seat assignments for blinds.
  if (num_active_players_ == 2) {
    small_blind_seat_ = DealerSeat();
    big_blind_seat_ = 1 - DealerSeat();
  } else {
    small_blind_seat_ = (DealerSeat() + 1) % num_active_players_;
    big_blind_seat_ = (DealerSeat() + 2) % num_active_players_;
  }
  // Update value of blinds based on the blind schedule.
  if (!blind_schedule_levels_.empty()) {
    int num_hands = 0;
    for (const auto& level : blind_schedule_levels_) {
      if (hand_number_ < num_hands + level.num_hands) {
        small_blind_ = level.small_blind;
        big_blind_ = level.big_blind;
        return;
      }
      num_hands += level.num_hands;
    }
    // If we've exceeded the schedule, use the last level.
    small_blind_ = blind_schedule_levels_.back().small_blind;
    big_blind_ = blind_schedule_levels_.back().big_blind;
  }
}

void RepeatedPokerState::UpdateUniversalPoker() {
  auto& acpc_state = universal_poker_state_->acpc_state();
  auto* acpc_game = acpc_state.game();

  universal_poker_game_params_["numPlayers"] = GameParameter(
      num_active_players_);

  std::vector<int> stacks;
  for (Player player_id = 0; player_id < num_players_; ++player_id) {
    if (player_to_seat_.at(player_id) != kInactivePlayerSeat) {
      stacks.push_back(stacks_[player_id]);
    }
  }
  universal_poker_game_params_["stack"] = GameParameter(
      absl::StrJoin(stacks, " "));
  std::vector<int> blinds(num_active_players_, 0);
  blinds[small_blind_seat_] = small_blind_;
  blinds[big_blind_seat_] = big_blind_;
  universal_poker_game_params_["blind"] = GameParameter(
      absl::StrJoin(blinds, " "));
  std::vector<int> first_player_per_round;
  for (int round_num = 0; round_num < acpc_game->NumRounds(); ++round_num) {
    if (round_num == 0) {   // Pre-flop differs from all other rounds.
      if (num_active_players_ == 2) {
        first_player_per_round.push_back(DealerSeat());
      } else {
        first_player_per_round.push_back(
            (big_blind_seat_ + 1) % num_active_players_);
      }
    } else {
      first_player_per_round.push_back(
          (DealerSeat() + 1) % num_active_players_);
    }
  }
  // Note that first player is 0-indexed in the ACPC code, but is 1-indexed in
  // the ACPC gamedef, so we increment here before updating the game param.
  for (int i = 0; i < first_player_per_round.size(); ++i) {
    first_player_per_round[i] += 1;
  }
  universal_poker_game_params_["firstPlayer"] = GameParameter(
      absl::StrJoin(first_player_per_round, " "));
  // Finally, load a new universal_poker game and state with the updated params.
  std::shared_ptr<const Game> universal_poker_game = LoadGame(
      "universal_poker", universal_poker_game_params_);
  universal_poker_state_ = std::make_unique<UniversalPokerState>(
      *dynamic_cast<UniversalPokerState*>(
          universal_poker_game->NewInitialState().get()));
}

void RepeatedPokerState::DoApplyAction(Action action) {
  universal_poker_state_->ApplyAction(action);
  if (!universal_poker_state_->IsTerminal()) {
    return;
  }
  // Record hand-level information.
  for (int i = 0; i < universal_poker_state_->Returns().size(); ++i) {
    Player p = seat_to_player_.at(i);
    hand_returns_.back()[p] = universal_poker_state_->Returns()[i];
  }
  auto& acpc_state = universal_poker_state_->acpc_state();
  acpc_hand_histories_.push_back(acpc_state.ToString());
  // Terminate or start a new hand.
  if (hand_number_ + 1 == max_num_hands_) {
    is_terminal_ = true;
    return;
  }
  hand_number_++;
  hand_returns_.push_back(std::vector<double>(num_players_, 0.0));
  UpdateStacks();
  UpdateSeatAssignments();
  if (num_active_players_ == 1) {
    is_terminal_ = true;  // We're playing a tournament and we have a winner.
    return;
  }
  UpdateDealer();
  UpdateBlinds();
  UpdateUniversalPoker();
}

Player RepeatedPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else if (universal_poker_state_->IsChanceNode()) {
    return kChancePlayerId;
  } else {
    return seat_to_player_.at(universal_poker_state_->CurrentPlayer());
  }
}

// TODO(jhtschultz): Switch to rewards?
std::vector<double> RepeatedPokerState::Returns() const {
  SPIEL_CHECK_EQ(hand_number_ + 1, hand_returns_.size());
  std::vector<double> returns(num_players_, 0.0);
  if (!IsTerminal()) {
    return returns;
  }
  for (const auto& hand_returns : hand_returns_) {
    for (int i = 0; i < num_players_; ++i) {
      returns[i] += hand_returns[i];
    }
  }
  return returns;
}

std::string RepeatedPokerState::ToString() const {
  return absl::StrCat(
      "Hand number: ", hand_number_, "\n",
      universal_poker_state_->ToString()
  );
}

std::string RepeatedPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  int seat_id = player_to_seat_.at(player);
  if (seat_id == kInactivePlayerSeat) {
    // TODO(jhtschultz): Consider adding an observer to universal_poker and
    // returning the public information here. This would allow players who have
    // been eliminated to continue watching the game.
    return "Game over.\n";
  }
  return universal_poker_state_->ObservationString(seat_id);
}

std::unique_ptr<State> RepeatedPokerState::Clone() const {
  return std::unique_ptr<State>(new RepeatedPokerState(*this));
}

RepeatedPokerGame::RepeatedPokerGame(const GameParameters& params)
    : Game(kGameType, params),
      max_num_hands_(ParameterValue<int>("max_num_hands")),
      reset_stacks_(ParameterValue<bool>("reset_stacks")),
      rotate_dealer_(ParameterValue<bool>("rotate_dealer")),
      blind_schedule_(ParameterValue<std::string>("blind_schedule")) {
  SPIEL_CHECK_GE(max_num_hands_, 1);
  GameParameter universal_poker_game_param =
      params.at("universal_poker_game_string");
  SPIEL_CHECK_TRUE(universal_poker_game_param.has_game_value());
  universal_poker_game_string_ = universal_poker_game_param.ToString();
  std::shared_ptr<const Game> base_game = LoadGame(
      universal_poker_game_string_);
  base_game_ = std::make_shared<UniversalPokerGame>(base_game->GetParameters());
}

std::unique_ptr<State> RepeatedPokerGame::NewInitialState() const {
  return std::unique_ptr<State>(new RepeatedPokerState(
      shared_from_this(), universal_poker_game_string_, max_num_hands_,
      reset_stacks_, rotate_dealer_, blind_schedule_));
}

// The action space must be able to support all legal bet sizes. This remains
// constant in the case of resetting stacks, but otherwise will increase as
// stack sizes increases. Players can bet more than their opponent's stack size,
// so the upper bound is set to the total number of chips. This isn't a tight
// upper bound, since the opponent must have had at least one big blind to start
// hand, but it's very close and should not matter for practical purposes.
int RepeatedPokerGame::NumDistinctActions() const {
    if (reset_stacks_) {
      return base_game_->NumDistinctActions();
    } else {
      const acpc_cpp::ACPCGame* acpc_game = base_game_->GetACPCGame();
      return acpc_game->TotalMoney();
    }
  }

double RepeatedPokerGame::MinUtility() const {
  const acpc_cpp::ACPCGame* acpc_game = base_game_->GetACPCGame();
  if (reset_stacks_ || acpc_game->IsLimitGame()) {
    return base_game_->MinUtility() * max_num_hands_;
  } else {
    std::vector<int> stacks(base_game_->NumPlayers());
    for (int i = 0; i < base_game_->NumPlayers(); ++i) {
      stacks[i] = acpc_game->StackSize(i);
    }
    return *std::max_element(stacks.begin(), stacks.end());
  }
}

double RepeatedPokerGame::MaxUtility() const {
  const acpc_cpp::ACPCGame* acpc_game = base_game_->GetACPCGame();
  if (reset_stacks_ || acpc_game->IsLimitGame()) {
    return base_game_->MaxUtility() * max_num_hands_;
  } else {
    return acpc_game->TotalMoney();
  }
}


}  // namespace repeated_poker
}  // namespace universal_poker
}  // namespace open_spiel
