// Copyright 2023 DeepMind Technologies Limited
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

#include "open_spiel/games/abalone/abalone.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"
#include "open_spiel/games/abalone/abalone_core_ab.h"


namespace open_spiel {
namespace abalone {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"abalone",
    /*long_name=*/"Abalone",
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
    /*parameter_specification=*/{
      {"marbles_to_win", GameParameter(abalone_core::kMarblesToWin)},
      {"marble_reward", GameParameter(abalone_core::kMarbleReward)},
      {"marble_advantage", GameParameter(abalone_core::kMarbleAdvantage)},
      {"draw_penalty", GameParameter(abalone_core::kDrawPenalty)},
      {"board", GameParameter(abalone_core::kDefaultBoard)},
      {"invert", GameParameter(abalone_core::kInvertBoard)}
    }  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AbaloneGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

void AbaloneState::UndoAction(Player player, Action action) {
  // We don't have direct undo functionality, so we replay the whole
  // history instead.
  history_.pop_back();
  --move_number_;
  ResetBoard();
  for (const auto& [history_player, history_action] : history_) {
    DoApplyAction(history_action);
  }
}

void AbaloneState::DoApplyAction(Action action) {
  // SPIEL_CHECK_EQ(board_[move], CellState::Empty);
  const auto &up_game = static_cast<const AbaloneGame&>(*this->game_);

  // update board state
  abalone_core::Move move = abalone_core::Move::ActionToMove(action);
  if (move.IsValid(this->core_state_)) {
    move.Apply(this->core_state_, up_game.m_marbles_to_win,
               abalone_core::kHistoryMax, up_game.m_marble_advantage);
  } else {
    core_state_.outcome_ =
        static_cast<abalone_core::CellState>(1 - core_state_.ToPlay());
  }

  auto new_returns = Returns();
  rewards_ = {new_returns[0] - prev_returns_[0],
              new_returns[1] - prev_returns_[1]};
  prev_returns_ = new_returns;
}

std::vector<Action> AbaloneState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (auto i = abalone_core::kActionMin; i < abalone_core::kActionMax; ++i) {
    auto move = abalone_core::Move::ActionToMove(i);
    if (move.IsValid(this->core_state_)) {
      moves.push_back(i);
    }
  }
  return moves;
}

std::string AbaloneState::ActionToString(Player player,
                                           Action action_id) const {
  return game_->ActionToString(player, action_id);
}

void AbaloneState::ResetBoard() {
  const auto &up_game = static_cast<const AbaloneGame&>(*this->game_);
  auto init_board = abalone_core::ABALONE_INIT_CLASSIC;
  if (up_game.m_init_board.compare("classic") == 0) {
    init_board = abalone_core::ABALONE_INIT_CLASSIC;
  } else if (up_game.m_init_board.compare("belgian-daisy") == 0) {
    init_board = abalone_core::ABALONE_INIT_BELGIAN_DAISY;
  } else {
    SpielFatalError(
        absl::StrCat("board init not found: ", up_game.m_init_board));
  }

  auto invert_board = [invert = up_game.m_init_invert](
      abalone_core::CellState c) {
    if (invert && c == abalone_core::CellState::Player0)
      return abalone_core::CellState::Player1;
    if (invert && c == abalone_core::CellState::Player1)
      return abalone_core::CellState::Player0;
    return c;
  };

  core_state_.Reset();

  // We need to reset the board, applying the inversion if requested.
  for (int r = 0; r < abalone_core::kNumRows; r++) {
    for (int c = 0; c < abalone_core::kNumCols; c++) {
      core_state_.board_[r][c] = invert_board(init_board[r][c]);
    }
  }

  rewards_ = {0.0, 0.0};
  prev_returns_ = {0.0, 0.0};
}

Action AbaloneState::StringToAction(Player player,
                                    const std::string& action_str) const {
  auto maybe_move = abalone_core::Move::FromString(action_str);
  if (std::get<0>(maybe_move)) {
    return abalone_core::Move::MoveToAction(std::get<1>(maybe_move));
  }

  SpielFatalError(
      absl::StrCat("Couldn't find an action matching ", action_str));
}

AbaloneState::AbaloneState(std::shared_ptr<const Game> game) : State(game) {
  ResetBoard();
}

std::string AbaloneState::ToString() const {
  std::string str;
  absl::StrAppend(&str, "board = \n");
  auto display_line = [&](std::string prefix, int line, int start, int end,
                          std::string postfix) {
    absl::StrAppend(&str, prefix);
    for (auto i = start; i < end; ++i) {
      absl::StrAppend(&str, "   ");
      absl::StrAppend(&str, StateToString(core_state_.board_[line][i]));
    }
    absl::StrAppend(&str, postfix);
    absl::StrAppend(&str, "\n");
  };
  display_line("<i>        ", 8, 4, 9, "");
  display_line("<h>      ", 7, 3, 9, "");
  display_line("<g>    ", 6, 2, 9, "");
  display_line("<f>  ", 5, 1, 9, "");
  display_line("<e>", 4, 0, 9, "");
  display_line("<d>  ", 3, 0, 8, "  <9>");
  display_line("<c>    ", 2, 0, 7, "  <8>");
  display_line("<b>      ", 1, 0, 6, "  <7>");
  display_line("<a>        ", 0, 0, 5, "  <6>");

  absl::StrAppend(&str, "               <1> <2> <3> <4> <5>\n");

  absl::StrAppend(&str, "move_number_ = ");
  absl::StrAppend(&str, move_number_);
  absl::StrAppend(&str, "\n");

  auto rewards = Rewards();
  absl::StrAppend(&str, "rewards = ");
  absl::StrAppend(&str, rewards[0]);
  absl::StrAppend(&str, ", ");
  absl::StrAppend(&str, rewards[1]);
  absl::StrAppend(&str, "\n");

  auto returns = Returns();
  absl::StrAppend(&str, "returns = ");
  absl::StrAppend(&str, returns[0]);
  absl::StrAppend(&str, ", ");
  absl::StrAppend(&str, returns[1]);
  absl::StrAppend(&str, "\n");

  absl::StrAppend(&str, "winner = ");
  absl::StrAppend(&str, StateToString(core_state_.outcome_));
  absl::StrAppend(&str, "\n");

  absl::StrAppend(&str, "done = ");
  absl::StrAppend(&str, IsTerminal());
  absl::StrAppend(&str, "\n");

  return str;
}

bool AbaloneState::IsTerminal() const {
  return core_state_.outcome_ != abalone_core::CellState::Invalid ||
         move_number_ >= abalone_core::kHistoryMax;
}

std::vector<double> AbaloneState::Rewards() const { return rewards_; }

std::vector<double> AbaloneState::Returns() const {
  // Set by an invalid move.
  if (core_state_.outcome_ != abalone_core::CellState::Invalid) {
    if (core_state_.outcome_ == abalone_core::CellState::Player0)
      return {1.0, -1.0};
    if (core_state_.outcome_ == abalone_core::CellState::Player1)
      return {-1.0, 1.0};
  }
  int ballCount[2] = { 0, 0 };
  for (int line = 0; line < abalone_core::kNumRows; ++line) {
    for (int column = 0; column < abalone_core::kNumCols; ++column) {
      auto slot = core_state_.board_[line][column];
      if (slot == abalone_core::CellState::Player0) {
        ballCount[0]++;
      } else if (slot == abalone_core::CellState::Player1) {
        ballCount[1]++;
      }
    }
  }

  const auto &up_game = static_cast<const AbaloneGame&>(*GetGame());
  if (ballCount[0] <= 14 - up_game.m_marbles_to_win) {
    return {-1.0, 1.0};
  }
  if (ballCount[1] <= 14 - up_game.m_marbles_to_win) {
    return {1.0, -1.0};
    // (ballCount[0]>ballCount[1])
    // if (ballCount[1]>ballCount[0])
    // return {0.0, 0.0};
  }
  // return {0.0, 0.0};
  const double marble_reward = up_game.m_marble_reward;
  auto marble_balance = (14-ballCount[1])-(14-ballCount[0]);
  double base = marble_balance * marble_reward;

  // Draw penalty: applied only on an actual draw (terminal via move limit
  // with no winner). Kept zero-sum: the player with fewer marbles is
  // penalized. No penalty on equal marble counts (cannot break a true tie
  // in a zero-sum fashion) or on non-terminal states.
  double penalty = 0.0;
  if (IsTerminal() && up_game.m_draw_penalty != 0.0) {
    if (ballCount[0] > ballCount[1])
      penalty = up_game.m_draw_penalty;
    else if (ballCount[1] > ballCount[0])
      penalty = -up_game.m_draw_penalty;
  }
  return {base + penalty, -base - penalty};
}

std::string AbaloneState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string AbaloneState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void AbaloneState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 3-d tensor.
  TensorView<3> view(values,
                     {abalone_core::kNumPlayers + 1, abalone_core::kNumRows,
                      abalone_core::kNumCols},
                     true);

  // Encode so the current player's marbles are always on the same layer.
  auto player1_index = 0;
  auto player2_index = 0;
  switch (player) {
    case abalone_core::CellState::Player0:
      player1_index = 1;
      player2_index = 2;
      break;
    case abalone_core::CellState::Player1:
      player1_index = 2;
      player2_index = 1;
      break;
  }

  for (int row = 0; row < abalone_core::kNumRows; ++row) {
    for (int col = 0; col < abalone_core::kNumCols; ++col) {
      auto index = 0;
      switch (core_state_.board_[row][col]) {
        case abalone_core::CellState::Invalid:
          continue;
        case abalone_core::CellState::Empty:
          index = 0;
          break;
        case abalone_core::CellState::Player0:
          index = player1_index;
          break;
        case abalone_core::CellState::Player1:
          index = player2_index;
          break;
      }
      view[{index, row, col}] = 1.f;
    }
  }
}

std::unique_ptr<State> AbaloneState::Clone() const {
  return std::unique_ptr<State>(new AbaloneState(*this));
}

std::unique_ptr<State> AbaloneGame::NewInitialState() const {
  return std::unique_ptr<State>(new AbaloneState(shared_from_this()));
}

std::string AbaloneGame::ActionToString(Player player,
                                          Action action_id) const {
  auto move = abalone_core::Move::ActionToMove(action_id);
  return absl::StrCat(move.ToString());
}

AbaloneGame::AbaloneGame(const GameParameters& params)
    : Game(kGameType, params) {
  m_marbles_to_win = ParameterValue<int>("marbles_to_win");
  m_marble_reward = ParameterValue<double>("marble_reward");
  m_draw_penalty = ParameterValue<double>("draw_penalty");
  m_init_board = ParameterValue<std::string>("board");
  m_init_invert = ParameterValue<bool>("invert");
  m_marble_advantage = ParameterValue<bool>("marble_advantage");
}

std::pair<open_spiel::Action, float> AllAbaloneMoves_ABSpiel(
    const std::unique_ptr<State>& _state, int _depth,
    std::vector<std::pair<open_spiel::Action, float>>* all_moves) {
  auto game = _state->GetGame();
  Player player = _state->CurrentPlayer();
  auto best_value = -1.001f;
  open_spiel::Action best_action = -1;
  std::vector<std::pair<open_spiel::Action, float>> children;
  for (auto action : _state->LegalActions()) {
    auto childstate = _state->Child(action);
    if (childstate.get()->IsTerminal()) {
      auto q_value = childstate->Returns()[player];
      if (q_value > best_value) {
        best_action = action;
        best_value = q_value;
      }
      if (all_moves) {
        all_moves->push_back(std::make_pair(action, q_value));
      }
      continue;
    }

    auto core_player = down_cast<const abalone::AbaloneState*>(
        childstate.get())->core_state_.ToPlay();
    std::pair<double, Action> value_action = algorithms::AlphaBetaSearch(
      *game,
      childstate.get(),
      [core_player](const State& state) {
          const auto& abalone_state =
              down_cast<const abalone::AbaloneState&>(state);
          return abalone_core::Heuristic(
              abalone_state.core_state_, core_player) * 0.001;
          },
      _depth-1,
      childstate.get()->CurrentPlayer());
    float q_value = -value_action.first;
    if (q_value > best_value) {
      best_action = action;
      best_value = q_value;
    }
    if (all_moves) {
      all_moves->push_back(std::make_pair(action, q_value));
    }
  }
  // auto max_move = *std::max_element(
  //     all_moves->begin(), all_moves->end(),
  //     [](const auto& m1, const auto& m2) {
  //       return m1.second < m2.second;
  //     });
  return std::make_pair(best_action, best_value);
}

std::vector<double> AbaloneEvaluator::Evaluate(const State& state) {
  auto abalone_state = down_cast<const abalone::AbaloneState&>(state);
  // auto core_player = abalone_state.core_state_.ToPlay();
  auto core_player = abalone_core::CellState::Player0;
  auto score =
      abalone_core::Heuristic(abalone_state.core_state_, core_player)
      * 0.001;
  std::vector<double> returns = {score, -score};
  return returns;
}

ActionsAndProbs AbaloneEvaluator::Prior(const State& state) {
  // Returns equal probability for all actions.
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    std::vector<Action> legal_actions = state.LegalActions();
    ActionsAndProbs prior;
    prior.reserve(legal_actions.size());
    for (const Action& action : legal_actions) {
      prior.emplace_back(action, 1.0 / legal_actions.size());
    }
    return prior;
  }
}

std::pair<Action, std::vector<std::pair<Action, float>>> AbaloneAB(
    const State& state, int depth) {
  if (const auto* stt = dynamic_cast<const abalone::AbaloneState*>(&state)) {
    std::vector<std::pair<abalone_core::core_Action, float>> core_all_moves;
    auto best_move = abalone_core::AlphaBeta(
        stt->core_state_, depth, -1.f, 1.f, &core_all_moves);
    std::vector<std::pair<Action, float>> all_moves;
    all_moves.reserve(core_all_moves.size());
    for (const auto& [action, value] : core_all_moves) {
      all_moves.emplace_back(static_cast<Action>(action), value);
    }
    return {static_cast<Action>(best_move.first), std::move(all_moves)};
  }
  SpielFatalError("state is not AbaloneState");
  return {static_cast<Action>(-1), {}};
}

}  // namespace abalone
}  // namespace open_spiel
