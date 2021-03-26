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

#include "open_spiel/games/k_gmp.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/fog/fog_constants.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/simultaneous_move_game.h"

namespace open_spiel {
namespace k_gmp {
namespace {

constexpr double kAnte = 1;

// Facts about the game
const GameType kGameType{/*short_name=*/"k_gmp",
                         /*long_name=*/"K General Matching Pennies",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation, //kImperfectInformation
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                          {"k", GameParameter(9)},
                           {"num_actions", GameParameter(3)}
                         }
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new KGMPGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

KGMPState::KGMPState(std::shared_ptr<const Game> game)
    : SimMoveState(game),
      parent_game_(static_cast<const KGMPGame&>(*game)),
      current_k_gmp_game_num_selected_(-1),
      winner_(kInvalidPlayer),
      k_(parent_game_.GetK()),
      n_actions_(parent_game_.GetNActions())
      {}

int KGMPState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
      if (current_k_gmp_game_num_selected_ == -1) {
          return 0;
      } else {
          return kSimultaneousPlayerId;
      }
  }
}

void KGMPState::DoApplyActions(const std::vector<Action>& actions) {
  if (current_k_gmp_game_num_selected_ == -1) {
      SPIEL_CHECK_NE(actions[0], kInvalidAction);
      SPIEL_CHECK_EQ(actions[1], kInvalidAction);
      current_k_gmp_game_num_selected_ = actions[0];
  } else {
      SPIEL_CHECK_NE(actions[0], kInvalidAction);
      SPIEL_CHECK_NE(actions[1], kInvalidAction);
      if (actions[0] == actions[1]) {
//          printf("winner is 0, actions are %d %d ", actions[0], actions[1]);
          winner_ = 0;
      } else {
//          printf("winner is 1, actions are %d %d ", actions[0], actions[1]);
          winner_ = 1;
      }

  }
}

void KGMPState::DoApplyAction(Action move) {
    SPIEL_CHECK_EQ(CurrentPlayer(), 0);
    SPIEL_CHECK_NE(move, kInvalidAction);
    current_k_gmp_game_num_selected_ = move;
}

std::vector<Action> KGMPState::LegalActions(Player player) const {
    if (IsTerminal()) return {};
    if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
    SPIEL_CHECK_FALSE(IsChanceNode());
    SPIEL_CHECK_TRUE(player == Player{0} || player == Player{1});

    std::vector<Action> movelist;

    if (current_k_gmp_game_num_selected_ == -1) {
        if (player == 0) {
            for (int i = 0; i < k_; i++) {
                movelist.push_back(i);
            }
        }
    } else {
        for (int i = 0; i < n_actions_; i++) {
            movelist.push_back(i);
        }
    }
    return movelist;
}

std::string KGMPState::ActionToString(Player player, Action move) const {
  if (current_k_gmp_game_num_selected_ == -1)
    return absl::StrCat("Chose game: ", move);
  else
    return absl::StrCat("GMP action: ", move);
}

std::string KGMPState::ToString() const {
  // The deal: space separated card per player
  std::string str;
  for (int i = 0; i < history_.size(); ++i) {
    if (!str.empty()) str.push_back(' ');
    absl::StrAppend(&str, history_[i].action);
  }

  return str;
}

bool KGMPState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> KGMPState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(2, 0.0);
  }
//  printf(" winner in Returns is %d ", winner_ );
  std::vector<double> returns(2);
  if (winner_ == 0) {
      returns[0] = n_actions_ - 1.0;
      returns[1] = -1.0 * (n_actions_ - 1.0);
  } else {
      SPIEL_CHECK_NE(winner_, kInvalidPlayer);
      SPIEL_CHECK_EQ(winner_, 1);
      returns[0] = -1.0;
      returns[1] = 1.0;
  }

  return returns;
}

std::string KGMPState::InformationStateString(Player player) const {
  return HistoryString();
}

std::string KGMPState::ObservationString(Player player) const {
  return ToString();
}

//void KGMPState::InformationStateTensor(Player player,
//                                       absl::Span<float> values) const {
//  ContiguousAllocator allocator(values);
//  const KuhnGame& game = open_spiel::down_cast<const KuhnGame&>(*game_);
//  game.info_state_observer_->WriteTensor(*this, player, &allocator);
//}

void KGMPState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
    std::fill(values.begin(), values.end(), 0.);

    if (current_k_gmp_game_num_selected_ != -1) {
        values[current_k_gmp_game_num_selected_] = 1.0;
    }
}

std::unique_ptr<State> KGMPState::Clone() const {
  return std::unique_ptr<State>(new KGMPState(*this));
}



//void KGMPState::UndoAction(Player player, Action move) {
//  if (history_.size() <= num_players_) {
//    // Undoing a deal move.
//    card_dealt_[move] = kInvalidPlayer;
//  } else {
//    // Undoing a bet / pass.
//    if (move == ActionType::kBet) {
//      pot_ -= 1;
//      if (player == first_bettor_) first_bettor_ = kInvalidPlayer;
//    }
//    winner_ = kInvalidPlayer;
//  }
//  history_.pop_back();
//}

//std::vector<std::pair<Action, double>> KGMPState::ChanceOutcomes() const {
//  SPIEL_CHECK_TRUE(IsChanceNode());
//  std::vector<std::pair<Action, double>> outcomes;
//  const double p = 1.0 / (num_players_ + 1 - history_.size());
//  for (int card = 0; card < card_dealt_.size(); ++card) {
//    if (card_dealt_[card] == kInvalidPlayer) outcomes.push_back({card, p});
//  }
//  return outcomes;
//}


//
//std::unique_ptr<State> KGMPState::ResampleFromInfostate(
//    int player_id, std::function<double()> rng) const {
//  std::unique_ptr<State> state = game_->NewInitialState();
//  Action player_chance = history_.at(player_id).action;
//  for (int p = 0; p < game_->NumPlayers(); ++p) {
//    if (p == history_.size()) return state;
//    if (p == player_id) {
//      state->ApplyAction(player_chance);
//    } else {
//      Action other_chance = player_chance;
//      while (other_chance == player_chance) {
//        other_chance = SampleAction(state->ChanceOutcomes(), rng()).first;
//      }
//      state->ApplyAction(other_chance);
//    }
//  }
//  SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
//  if (game_->NumPlayers() == history_.size()) return state;
//  for (int i = game_->NumPlayers(); i < history_.size(); ++i) {
//    state->ApplyAction(history_.at(i).action);
//  }
//  return state;
//}

KGMPGame::KGMPGame(const GameParameters& params)
    : Game(kGameType, params),
    k_(ParameterValue<int>("k")),
    num_actions_(ParameterValue<int>("num_actions")) {
}

std::unique_ptr<State> KGMPGame::NewInitialState() const {
  return std::unique_ptr<State>(new KGMPState(shared_from_this()));
}

//std::vector<int> KuhnGame::InformationStateTensorShape() const {
//  // One-hot for whose turn it is.
//  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
//  // Followed by 2 (n - 1 + n) bits for betting sequence (longest sequence:
//  // everyone except one player can pass and then everyone can bet/pass).
//  // n + n + 1 + 2 (n-1 + n) = 6n - 1.
//  return {6 * num_players_ - 1};
//}

//std::vector<int> KuhnGame::ObservationTensorShape() const {
//  // One-hot for whose turn it is.
//  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
//  // Followed by the contribution of each player to the pot (n).
//  // n + n + 1 + n = 3n + 1.
//  return k_;
//}

//double KuhnGame::MaxUtility() const {
//
//  return (num_players_ - 1) * 2;
//}
//
//double KuhnGame::MinUtility() const {
//
//  return -2;
//}

}  // namespace k_gmp
}  // namespace open_spiel
