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

#include "open_spiel/games/snake/snake.h"

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace snake {
namespace {

const GameType kGameType{/*short_name=*/"snake",
                         /*long_name=*/"Snake",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kMaxPlayers,
                         /*min_num_players=*/kMinPlayers,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"rows", GameParameter(kDefaultRows)},
                          {"columns", GameParameter(kDefaultColumns)},
                          {"players", GameParameter(kDefaultPlayers)},
                          {"horizon", GameParameter(kDefaultHorizon)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SnakeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

constexpr std::array<int, 4> kRowDelta = {{-1, 0, 1, 0}};
constexpr std::array<int, 4> kColDelta = {{0, 1, 0, -1}};

std::string DirectionString(Action a) {
  switch (a) {
    case kNorth:
      return "North";
    case kEast:
      return "East";
    case kSouth:
      return "South";
    case kWest:
      return "West";
    default:
      return absl::StrCat("Unknown(", a, ")");
  }
}

}  // namespace

SnakeState::SnakeState(std::shared_ptr<const Game> game, int rows, int cols,
                       int num_players, int horizon)
    : SimMoveState(game),
      num_rows_(rows),
      num_cols_(cols),
      num_players_total_(num_players),
      horizon_(horizon),
      snakes_(num_players) {
  PlaceInitialSnakes();
}

void SnakeState::PlaceInitialSnakes() {
  // 2 players: one on the left edge, one on the right edge, both centered
  // vertically.
  // 4 players: corners (one per corner), each one cell in from the corner.
  if (num_players_total_ == 2) {
    snakes_[0].body.push_back({num_rows_ / 2, 1});
    snakes_[1].body.push_back({num_rows_ / 2, num_cols_ - 2});
  } else {
    SPIEL_CHECK_EQ(num_players_total_, 4);
    snakes_[0].body.push_back({1, 1});
    snakes_[1].body.push_back({1, num_cols_ - 2});
    snakes_[2].body.push_back({num_rows_ - 2, 1});
    snakes_[3].body.push_back({num_rows_ - 2, num_cols_ - 2});
  }
}

bool SnakeState::InBounds(const Cell& c) const {
  return c.row >= 0 && c.row < num_rows_ && c.col >= 0 && c.col < num_cols_;
}

int SnakeState::NumAlive() const {
  int n = 0;
  for (const auto& s : snakes_) {
    if (s.alive) ++n;
  }
  return n;
}

std::vector<Cell> SnakeState::EmptyCells() const {
  std::vector<std::vector<bool>> occupied(num_rows_,
                                          std::vector<bool>(num_cols_, false));
  for (const auto& s : snakes_) {
    if (!s.alive) continue;
    for (const auto& c : s.body) occupied[c.row][c.col] = true;
  }
  if (has_fruit_) occupied[fruit_.row][fruit_.col] = true;
  std::vector<Cell> empties;
  empties.reserve(num_rows_ * num_cols_);
  for (int r = 0; r < num_rows_; ++r) {
    for (int c = 0; c < num_cols_; ++c) {
      if (!occupied[r][c]) empties.push_back({r, c});
    }
  }
  return empties;
}

void SnakeState::PlaceFruit(int empty_cell_index) {
  std::vector<Cell> empties = EmptyCells();
  SPIEL_CHECK_GE(empty_cell_index, 0);
  SPIEL_CHECK_LT(empty_cell_index, empties.size());
  fruit_ = empties[empty_cell_index];
  has_fruit_ = true;
}

Player SnakeState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (chance_pending_) return kChancePlayerId;
  return kSimultaneousPlayerId;
}

ActionsAndProbs SnakeState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(chance_pending_);
  std::vector<Cell> empties = EmptyCells();
  ActionsAndProbs outcomes;
  outcomes.reserve(empties.size());
  const double p = 1.0 / empties.size();
  for (int i = 0; i < empties.size(); ++i) {
    outcomes.emplace_back(i, p);
  }
  return outcomes;
}

std::vector<Action> SnakeState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (player == kChancePlayerId) {
    SPIEL_CHECK_TRUE(chance_pending_);
    std::vector<Cell> empties = EmptyCells();
    std::vector<Action> actions;
    actions.reserve(empties.size());
    for (int i = 0; i < empties.size(); ++i) actions.push_back(i);
    return actions;
  }
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_total_);
  // All four directions are always legal. Dead snakes are also given the four
  // actions; they have no effect (their state remains dead).
  return {kNorth, kEast, kSouth, kWest};
}

std::string SnakeState::ActionToString(Player player, Action action_id) const {
  if (player == kSimultaneousPlayerId) {
    return FlatJointActionToString(action_id);
  }
  if (player == kChancePlayerId) {
    std::vector<Cell> empties = EmptyCells();
    SPIEL_CHECK_GE(action_id, 0);
    SPIEL_CHECK_LT(action_id, empties.size());
    return absl::StrCat("Place fruit at (", empties[action_id].row, ",",
                        empties[action_id].col, ")");
  }
  return absl::StrCat("p", player, ":", DirectionString(action_id));
}

void SnakeState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(chance_pending_);
  PlaceFruit(action_id);
  chance_pending_ = false;
}

void SnakeState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_FALSE(chance_pending_);
  SPIEL_CHECK_EQ(actions.size(), num_players_total_);

  // 1) Compute new heads for alive snakes.
  std::vector<Cell> new_heads(num_players_total_, {-1, -1});
  std::vector<bool> died_now(num_players_total_, false);
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive) continue;
    Action a = actions[p];
    SPIEL_CHECK_GE(a, 0);
    SPIEL_CHECK_LT(a, kNumMovementActions);
    int dir = static_cast<int>(a);
    Cell head = snakes_[p].body.front();
    new_heads[p] = {head.row + kRowDelta[dir], head.col + kColDelta[dir]};
  }

  // 2) Out-of-bounds → dies.
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive) continue;
    if (!InBounds(new_heads[p])) died_now[p] = true;
  }

  // 3) Head-to-head collisions: any two alive snakes with same new head die.
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive || died_now[p]) continue;
    for (Player q = p + 1; q < num_players_total_; ++q) {
      if (!snakes_[q].alive || died_now[q]) continue;
      if (new_heads[p] == new_heads[q]) {
        died_now[p] = true;
        died_now[q] = true;
      }
    }
  }

  // 4) Determine fruit eaters and build post-move bodies for snakes that
  //    survived steps 2 and 3.
  std::vector<bool> ate_fruit(num_players_total_, false);
  std::vector<std::deque<Cell>> new_bodies(num_players_total_);
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive || died_now[p]) continue;
    new_bodies[p] = snakes_[p].body;
    new_bodies[p].push_front(new_heads[p]);
    if (has_fruit_ && new_heads[p] == fruit_) {
      ate_fruit[p] = true;
    } else {
      new_bodies[p].pop_back();
    }
  }

  // If multiple surviving snakes targeted the fruit, head-to-head collision in
  // step 3 already killed them. So at most one snake ate the fruit.

  // 5) Head-to-body collisions among surviving snakes.
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive || died_now[p]) continue;
    const Cell& h = new_heads[p];
    bool collided = false;
    for (Player q = 0; q < num_players_total_; ++q) {
      if (!snakes_[q].alive || died_now[q]) continue;
      const auto& body = new_bodies[q];
      // Skip this snake's own head (index 0).
      int start = (p == q) ? 1 : 0;
      for (int i = start; i < static_cast<int>(body.size()); ++i) {
        if (body[i] == h) {
          collided = true;
          break;
        }
      }
      if (collided) break;
    }
    if (collided) died_now[p] = true;
  }

  // 6) Apply: surviving snakes adopt their new bodies; dead snakes lose body.
  bool fruit_eaten = false;
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive) continue;
    if (died_now[p]) {
      snakes_[p].alive = false;
      snakes_[p].body.clear();
    } else {
      snakes_[p].body = new_bodies[p];
      if (ate_fruit[p]) {
        snakes_[p].score += 1;
        fruit_eaten = true;
      }
    }
  }
  if (fruit_eaten) {
    has_fruit_ = false;
    fruit_ = {-1, -1};
  }

  ++total_moves_;

  // 7) If the game is not over and the fruit was eaten (and there is room for
  //    a new one), schedule a chance node to place a new fruit.
  if (!IsTerminal() && !has_fruit_) {
    if (!EmptyCells().empty()) {
      chance_pending_ = true;
    }
    // If there are no empty cells, simply leave has_fruit_ false; the game
    // will continue (or terminate by horizon / death).
  }
}

bool SnakeState::IsTerminal() const {
  if (total_moves_ >= horizon_) return true;
  return NumAlive() < 2;
}

std::vector<double> SnakeState::Returns() const {
  std::vector<double> returns(num_players_total_, 0.0);
  if (!IsTerminal()) return returns;
  for (Player p = 0; p < num_players_total_; ++p) {
    returns[p] = static_cast<double>(snakes_[p].score);
  }
  return returns;
}

std::string SnakeState::ToString() const {
  // Render the board with characters:
  //   '.' empty
  //   '*' fruit
  //   digit '0'..'3' for snake head
  //   lowercase 'a'..'d' for snake body
  std::vector<std::string> grid(num_rows_, std::string(num_cols_, '.'));
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive) continue;
    const auto& body = snakes_[p].body;
    for (int i = 0; i < static_cast<int>(body.size()); ++i) {
      char ch =
          (i == 0) ? static_cast<char>('0' + p) : static_cast<char>('a' + p);
      grid[body[i].row][body[i].col] = ch;
    }
  }
  if (has_fruit_) grid[fruit_.row][fruit_.col] = '*';
  std::string out;
  for (int r = 0; r < num_rows_; ++r) {
    absl::StrAppend(&out, grid[r], "\n");
  }
  absl::StrAppend(&out, "Scores:");
  for (Player p = 0; p < num_players_total_; ++p) {
    absl::StrAppend(&out, " p", p, "=", snakes_[p].score,
                    snakes_[p].alive ? "" : "(dead)");
  }
  absl::StrAppend(&out, "\n");
  return out;
}

std::string SnakeState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_total_);
  return ToString();
}

void SnakeState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_total_);
  TensorView<3> view(values, {2 * num_players_total_ + 1, num_rows_, num_cols_},
                     true);
  for (Player p = 0; p < num_players_total_; ++p) {
    if (!snakes_[p].alive) continue;
    const auto& body = snakes_[p].body;
    for (int i = 0; i < static_cast<int>(body.size()); ++i) {
      int plane = (i == 0) ? (2 * p) : (2 * p + 1);
      view[{plane, body[i].row, body[i].col}] = 1.0;
    }
  }
  if (has_fruit_) {
    view[{2 * num_players_total_, fruit_.row, fruit_.col}] = 1.0;
  }
}

std::unique_ptr<State> SnakeState::Clone() const {
  return std::unique_ptr<State>(new SnakeState(*this));
}

SnakeGame::SnakeGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      num_rows_(ParameterValue<int>("rows")),
      num_cols_(ParameterValue<int>("columns")),
      num_players_(ParameterValue<int>("players")),
      horizon_(ParameterValue<int>("horizon")) {
  SPIEL_CHECK_GE(num_rows_, 4);
  SPIEL_CHECK_GE(num_cols_, 4);
  SPIEL_CHECK_GE(horizon_, 1);
  if (num_players_ != 2 && num_players_ != 4) {
    SpielFatalError(
        absl::StrCat("Snake supports 2 or 4 players, got ", num_players_));
  }
}

}  // namespace snake
}  // namespace open_spiel
