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

#include "open_spiel/games/ultimate_tic_tac_toe/ultimate_tic_tac_toe.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace ultimate_tic_tac_toe {
namespace {

namespace ttt = tic_tac_toe;

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"ultimate_tic_tac_toe",
    /*long_name=*/"Ultimate Tic-Tac-Toe",
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new UltimateTTTGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

bool UltimateTTTState::AllLocalStatesTerminal() const {
  return std::any_of(
      local_states_.begin(), local_states_.end(),
      [](const std::unique_ptr<State>& state) { return state->IsTerminal(); });
}

void UltimateTTTState::DoApplyAction(Action move) {
  if (current_state_ < 0) {
    // Choosing a board.
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, kNumSubgames);
    current_state_ = move;
  } else {
    // Apply action to local state, then apply that move.
    SPIEL_CHECK_FALSE(local_states_[current_state_]->IsTerminal());
    local_states_[current_state_]->ApplyAction(move);
    // Check if it's terminal and mark the outcome in the meta-game.
    if (local_states_[current_state_]->IsTerminal()) {
      Player local_outcome = local_state(current_state_)->outcome();
      if (local_outcome >= 0) {
        meta_board_.At(current_state_).component_ =
          ttt::PlayerToComponent(local_outcome);
      }
    }
    // Set the next potential local state.
    current_state_ = move;
    // Check for a win or no more moves in the meta-game.
    if (ttt::BoardHasLine(meta_board_, current_player_)) {
      outcome_ = current_player_;
    } else if (AllLocalStatesTerminal()) {
      outcome_ = kInvalidPlayer;  // draw.
    } else {
      // Does the next board done? If not, set current_state_ to less than 0 to
      // indicate that the next board is a choice.
      if (local_states_[current_state_]->IsTerminal()) {
        current_state_ = -1;
      }
      current_player_ = NextPlayerRoundRobin(current_player_, ttt::kNumPlayers);
      // Need to set the current player in the local board.
      if (current_state_ >= 0) {
        local_state(current_state_)->SetCurrentPlayer(current_player_);
      }
    }
  }
}

std::vector<Action> UltimateTTTState::LegalActions() const {
  if (IsTerminal()) return {};
  if (current_state_ < 0) {
    std::vector<Action> actions;
    // Choosing the next state to play: any one that is not finished.
    for (int i = 0; i < local_states_.size(); ++i) {
      if (!local_states_[i]->IsTerminal()) {
        actions.push_back(i);
      }
    }
    return actions;
  } else {
    return local_states_[current_state_]->LegalActions();
  }
}

std::string UltimateTTTState::ActionToString(Player player,
                                             Action action_id) const {
  if (current_state_ < 0) {
    return absl::StrCat("Choose local board ", action_id);
  } else {
    return absl::StrCat(
        "Local board ", current_state_, ": ",
        local_states_[current_state_]->ActionToString(player, action_id));
  }
}

UltimateTTTState::UltimateTTTState(std::shared_ptr<const Game> game)
    : State(game),
      ttt_game_(
          static_cast<const UltimateTTTGame*>(game.get())->TicTacToeGame()),
      meta_board_(kNumSubRows, kNumSubCols),
      current_state_(-1) {
  for (int i = 0; i < local_states_.size(); ++i) {
    local_states_[i] = ttt_game_->NewInitialState();
  }
}

Display::Display(std::string str) {
  // Convert the line from newline-separated lines to actual tracked lines
  std::stringstream ss(str);
  std::string line;
  while (std::getline(ss, line, '\n')) {
    display_.push_back(line + "\n");
  }
}

void Display::Add(size_t dest_line, const Display other) {
  // If the line number is within the current display vertical range the lines
  // are appended to the end of the current line; otherwise, we need to expand
  // current vertical range
  if (dest_line + other.display_.size() > display_.size()) {
    display_.resize(dest_line + other.display_.size());
  }

  // Merge the lines at the line number
  for (size_t l = 0; l < other.display_.size(); ++l) {
    auto &line = display_[dest_line + l];

    // Remove the newline, if any, from the end of the current line, so
    // that both lines can be merged
    if (!line.empty() && (line.back() == '\n')) {
      line = line.substr(0, line.size() - 1);
    }
    line += other.display_[l];
  }
}

size_t Display::Height() const {
  return display_.size();
}

std::string Display::ToString() const {
  std::string str;
  for (const auto &line : display_) {
    str += line;
  }
  return str;
}

std::string UltimateTTTState::ToString() const {
  Display display;

  // Each sub TTT is separated from the other sub TTTs in the same row by
  // a space, and is separated from the other sub TTTs in the same col by
  // a new line
  for (int meta_row = 0; meta_row < kNumSubRows; ++meta_row) {
    for (int meta_col = 0; meta_col < kNumSubCols; ++meta_col) {
      int state_idx = meta_row * kNumSubCols + meta_col;
      SPIEL_CHECK_GE(state_idx, 0);
      SPIEL_CHECK_LT(state_idx, local_states_.size());

      // Get the corresponding subgame as a string so that we can add it
      // to the current display
      auto str = local_state(state_idx)->ToString();

      // Add this subgame to the display. This assumes that all sub-games
      // in this row have the same height
      Display sub_display(str);
      display.Add(display.Height() -
                  ((meta_col == 0) ? 0 : sub_display.Height()), sub_display);

      // If this is not the last subgame in this row we have to add a space
      // to separate this subgame from the next one
      if (meta_col != (kNumSubCols - 1)) {
        Display space_display(" ");
        const auto start_row = display.Height() - sub_display.Height();
        for (size_t row = 0; row < sub_display.Height(); ++row) {
          display.Add(start_row + row, space_display);
        }
      }
    }

    // By the end of every row of subgames we add a space, so that each meta
    // row is visually separated from the others
    if (meta_row < (kNumSubRows - 1)) {
      Display space_display("\n");
      display.Add(display.Height(), space_display);
    }
  }

  return display.ToString();
}

bool UltimateTTTState::IsTerminal() const { return outcome_ != kUnfinished; }

std::vector<double> UltimateTTTState::Returns() const {
  std::vector<double> returns = {0.0, 0.0};
  if (outcome_ >= 0) {
    returns[outcome_] = 1.0;
    returns[1 - outcome_] = -1.0;
  }
  return returns;
}

std::string UltimateTTTState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string UltimateTTTState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void UltimateTTTState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_GT(local_states_.size(), 0);
  const auto num_cells = local_state(0)->board_.Size();

  // Treat `values` as a 3-d tensor:
  // num meta cells x number of states x num sub cells:
  //   - local state index, then
  //   - empty versus x versus o, then
  //   - then position within the local board.
  TensorView<3> view(values, {kNumSubgames, ttt::kCellStates, num_cells},
                     /*reset*/true);
  for (int state = 0; state < kNumSubgames; ++state) {
    SPIEL_CHECK_EQ(local_state(state)->board_.Size(), num_cells);
    for (int cell = 0; cell < num_cells; ++cell) {
      view[{state, TileToState(local_state(state)->BoardAt(cell)),
            cell}] = 1.0;
    }
  }
}

void UltimateTTTState::UndoAction(Player player, Action move) {}

UltimateTTTState::UltimateTTTState(const UltimateTTTState& other)
    : State(other),
      current_player_(other.current_player_),
      outcome_(other.outcome_),
      ttt_game_(other.ttt_game_),
      meta_board_(other.meta_board_),
      current_state_(other.current_state_) {
  SPIEL_CHECK_EQ(local_states_.size(), other.local_states_.size());
  for (int i = 0; i < local_states_.size(); ++i) {
    local_states_[i] = other.local_states_[i]->Clone();
  }
}

std::unique_ptr<State> UltimateTTTState::Clone() const {
  return std::unique_ptr<State>(new UltimateTTTState(*this));
}

UltimateTTTGame::UltimateTTTGame(const GameParameters& params)
    : Game(kGameType, params), ttt_game_(LoadGame("tic_tac_toe")) {}

}  // namespace ultimate_tic_tac_toe
}  // namespace open_spiel
