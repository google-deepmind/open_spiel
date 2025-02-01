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

#include "open_spiel/games/phantom_ttt/phantom_ttt.h"

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace phantom_ttt {
namespace {

using tic_tac_toe::kCellStates;

using tic_tac_toe::CellState;

using tic_tac_toe::PlayerToState;
using tic_tac_toe::StateToString;

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"phantom_ttt",
    /*long_name=*/"Phantom Tic Tac Toe",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"obstype", GameParameter(std::string(kDefaultObsType))}}};

const GameType kImperfectRecallGameType{
    /*short_name=*/"phantom_ttt_ir",
    /*long_name=*/"Phantom Tic Tac Toe with Imperfect Recall",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"obstype", GameParameter(std::string(kDefaultObsType))}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PhantomTTTGame(params, kGameType));
}

std::shared_ptr<const Game> ImperfectRecallFactory(
    const GameParameters& params) {
  return std::shared_ptr<const Game>(new ImperfectRecallPTTTGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

REGISTER_SPIEL_GAME(kImperfectRecallGameType, ImperfectRecallFactory);
RegisterSingleTensorObserver single_tensor_imperfect_recall(
    kImperfectRecallGameType.short_name);

}  // namespace

ImperfectRecallPTTTGame::ImperfectRecallPTTTGame(const GameParameters& params)
    : PhantomTTTGame(params, kImperfectRecallGameType) {}

PhantomTTTState::PhantomTTTState(std::shared_ptr<const Game> game,
                                 ObservationType obs_type, size_t rows,
                                 size_t cols)
    : State(game), state_(game, rows, cols), obs_type_(obs_type),
      x_view_(rows, cols), o_view_(rows, cols) {
  if (obs_type_ == ObservationType::kRevealNumTurns) {
    // Reserve 1 bit to select the player and another bit as "I don't know."
    bits_per_action_ = NumDistinctActions() + 2;
    // The longest sequence happens when each player perfectly mimics
    // the moves of the other player, e.g. 0011223344556677889. Once
    // the last possible action is taken by any of the players, the
    // game ends (i.e., this action is not mimicable by the other player).
    longest_sequence_ = 2 * NumDistinctActions() - 1;
  } else {
    SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    bits_per_action_ = NumDistinctActions();
    longest_sequence_ = NumDistinctActions();
  }
}

void PhantomTTTState::DoApplyAction(Action move) {
  // Current player's view.
  Player cur_player = CurrentPlayer();
  auto& cur_view = cur_player == 0 ? x_view_ : o_view_;

  // Two cases: either there is a mark already there, or not.
  if (state_.BoardAt(move) == CellState::kEmpty) {
    // No mark on board, so play this normally.
    state_.ApplyAction(move);
  }

  // Update current player's view, and action sequence.
  SPIEL_CHECK_EQ(cur_view.At(move), CellState::kEmpty);
  cur_view.At(move) = state_.BoardAt(move);
  action_sequence_.push_back(std::pair<int, Action>(cur_player, move));

  // Note: do not modify player's turn here, it will have been done above
  // if necessary.
}

std::vector<Action> PhantomTTTState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  const Player player = CurrentPlayer();
  const auto& cur_view = player == 0 ? x_view_ : o_view_;

  for (Action move = 0; move < cur_view.Size(); ++move) {
    if (cur_view.At(move) == CellState::kEmpty) {
      moves.push_back(move);
    }
  }

  return moves;
}

std::string PhantomTTTState::ViewToString(Player player) const {
  const auto& cur_view = player == 0 ? x_view_ : o_view_;
  std::string str;
  for (int r = 0; r < cur_view.Rows(); ++r) {
    for (int c = 0; c < cur_view.Cols(); ++c) {
      absl::StrAppend(&str,
                      StateToString(cur_view.At(r * cur_view.Cols() + c)));
    }
    if (r < (cur_view.Rows() - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

std::string PhantomTTTState::ActionSequenceToString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string str;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      // Always include the observing player's actions.
      absl::StrAppend(&str, player_with_action.first, ",");
      absl::StrAppend(&str, player_with_action.second, " ");
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      // If the number of turns are revealed, then each of the other player's
      // actions will show up as unknowns.
      absl::StrAppend(&str, player_with_action.first, ",? ");
    } else {
      // Do not reveal anything about the number of actions taken by opponent.
      SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    }
  }
  return str;
}

std::string PhantomTTTState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string str;
  absl::StrAppend(&str, ViewToString(player), "\n");
  if (obs_type_ != ObservationType::kRevealNothing) {
    absl::StrAppend(&str, history_.size(), "\n");
  }
  absl::StrAppend(&str, ActionSequenceToString(player));
  return str;
}

void PhantomTTTState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // First bits encodes the player's view in the same way as TicTacToe.
  // Then the action sequence follows (one-hot encoded, per action).
  // Encoded in the same way as InformationStateAsString, so full sequences
  // which may contain action value 10 to represent "I don't know."
  const auto& player_view = player == 0 ? x_view_ : o_view_;
  SPIEL_CHECK_EQ(values.size(), NumDistinctActions() * kCellStates +
                                    longest_sequence_ * bits_per_action_);
  std::fill(values.begin(), values.end(), 0.);
  for (int cell = 0; cell < player_view.Size(); ++cell) {
    values[NumDistinctActions() * static_cast<int>(player_view.At(cell)) +
           cell] = 1.0;
  }

  // Now encode the sequence. Each (player, action) pair uses 11 bits:
  //   - first bit is the player taking the action (0 or 1)
  //   - next 10 bits is the one-hot encoded action (10 = "I don't know")
  int offset = NumDistinctActions() * kCellStates;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      // Always include the observing player's actions.
      if (obs_type_ == ObservationType::kRevealNumTurns) {
        values[offset] = player_with_action.first;  // Player 0 or 1
        values[offset + 1 + player_with_action.second] = 1.0;
      } else {
        // Here we don't need to encode the player since we won't see opponent
        // moves.
        SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
        values[offset + player_with_action.second] = 1.0;
      }
      offset += bits_per_action_;
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      // If the number of turns are revealed, then each of the other player's
      // actions will show up as unknowns.
      values[offset] = player_with_action.first;
      values[offset + 1 + NumDistinctActions()] = 1.0;  // I don't know.
      offset += bits_per_action_;
    } else {
      // Do not reveal anything about the number of actions taken by opponent.
      SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    }
  }
}

std::string PhantomTTTState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string observation = ViewToString(player);
  if (obs_type_ == ObservationType::kRevealNumTurns) {
    absl::StrAppend(&observation, "\nTotal turns: ", action_sequence_.size());
  }
  return observation;
}

void PhantomTTTState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0.);

  // First bits encodes the player's view in the same way as TicTacToe.
  const auto& player_view = player == 0 ? x_view_ : o_view_;
  for (int cell = 0; cell < player_view.Size(); ++cell) {
    values[NumDistinctActions() * static_cast<int>(player_view.At(cell)) +
           cell] = 1.0;
  }

  // Then a one-hot to represent total number of turns.
  if (obs_type_ == ObservationType::kRevealNumTurns) {
    values[NumDistinctActions() * kCellStates + action_sequence_.size()] = 1.0;
  }
}

std::unique_ptr<State> PhantomTTTState::Clone() const {
  return std::unique_ptr<State>(new PhantomTTTState(*this));
}

void PhantomTTTState::UndoAction(Player player, Action move) {
  Action last_move = action_sequence_.back().second;
  SPIEL_CHECK_EQ(last_move, move);

  if (state_.BoardAt(move) == PlayerToState(player)) {
    // If the board has a mark that is the undoing player, then this was
    // a successful move. Undo as normal.
    state_.UndoAction(player, move);
  }

  // Undo the action from that player's view, and pop from the action seq
  auto& player_view = player == 0 ? x_view_ : o_view_;
  player_view.At(move) = CellState::kEmpty;
  action_sequence_.pop_back();

  history_.pop_back();
  --move_number_;
  // Note, do not change the player.. this will already have been done above
  // if necessary.
}

PhantomTTTGame::PhantomTTTGame(const GameParameters& params, GameType game_type)
    : Game(game_type, params),
      game_(std::static_pointer_cast<const tic_tac_toe::TicTacToeGame>(
          LoadGame("tic_tac_toe"))) {
  std::string obs_type = ParameterValue<std::string>("obstype");
  if (obs_type == "reveal-nothing") {
    obs_type_ = ObservationType::kRevealNothing;
    bits_per_action_ = game_->NumDistinctActions();
    longest_sequence_ = game_->NumDistinctActions();
  } else if (obs_type == "reveal-numturns") {
    obs_type_ = ObservationType::kRevealNumTurns;
    // Reserve 1 bit to select the player and another bit as "I don't know."
    bits_per_action_ = game_->NumDistinctActions() + 2;
    // The longest sequence happens when each player perfectly mimics
    // the moves of the other player, e.g. 0011223344556677889. Once
    // the last possible action is taken by any of the players, the
    // game ends (i.e., this action is not mimicable by the other player).
    longest_sequence_ = 2 * game_->NumDistinctActions() - 1;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized observation type: ", obs_type));
  }
}

std::vector<int> PhantomTTTGame::InformationStateTensorShape() const {
  // Enc
  return {1, game_->NumDistinctActions() * kCellStates +
      longest_sequence_ * bits_per_action_};
}

std::vector<int> PhantomTTTGame::ObservationTensorShape() const {
  if (obs_type_ == ObservationType::kRevealNothing) {
    return {game_->NumDistinctActions() * kCellStates};
  } else if (obs_type_ == ObservationType::kRevealNumTurns) {
    return {game_->NumDistinctActions() * kCellStates + longest_sequence_};
  } else {
    SpielFatalError("Unknown observation type");
  }
}

}  // namespace phantom_ttt
}  // namespace open_spiel
