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

#include "open_spiel/games/colored_trails/colored_trails.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace colored_trails {

namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"colored_trails",
                         /*long_name=*/"Colored Trails",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/3,
                         /*min_num_players=*/3,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"boards_file", GameParameter("")},
                          {"board_size", GameParameter(kDefaultBoardSize)},
                          {"num_colors", GameParameter(kDefaultNumColors)},
                          {"players", GameParameter(kDefaultNumPlayers)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ColoredTrailsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

bool IsLegalTrade(
    const Board& board, const Trade& trade,
    const std::vector<int>& proposer_chips,
    const std::vector<int>& responder_chips) {
  if (trade.giving.empty() || trade.receiving.empty()) {
    // pass trade is always legal.
    return true;
  }

  for (int i = 0; i < board.num_colors; ++i) {
    if (trade.giving[i] > proposer_chips[i]) {
      return false;
    }

    if (trade.receiving[i] > responder_chips[i]) {
      return false;
    }
  }

  // Try to reduce the trade, if it's not valid or not equal to the same trade
  // then not a legal trade.
  Trade copy = trade;
  bool valid = copy.reduce();
  return (valid && copy == trade);
}


std::vector<Action> GenerateLegalActionsForChips(
    const ColoredTrailsGame* game,
    const Board& board,
    const std::vector<int>& player_chips,
    const std::vector<int>& responder_chips) {
  std::vector<Action> actions;
  ChipComboIterator proposer_iter(player_chips);
  while (!proposer_iter.IsFinished()) {
    std::vector<int> proposer_chips = proposer_iter.Next();
    ChipComboIterator receiver_iter(responder_chips);
    while (!receiver_iter.IsFinished()) {
      std::vector<int> receiver_chips = receiver_iter.Next();
      Trade trade(proposer_chips, receiver_chips);
      if (IsLegalTrade(board, trade, proposer_chips, responder_chips)) {
        int trade_id = game->LookupTradeId(trade.ToString());
        actions.push_back(trade_id);
      }
    }
  }
  // Sort and remove duplicates.
  absl::c_sort(actions);
  auto last = std::unique(actions.begin(), actions.end());
  actions.erase(last, actions.end());

  // Add pass trade.
  actions.push_back(game->PassAction());
  return actions;
}

}  // namespace

Board::Board()
    : board(size * size, -1),
      num_chips(num_players, -1),
      positions(num_players + 1, -1) {
  init();
}

Board::Board(int _size, int _num_colors, int _num_players)
    : size(_size),
      num_colors(_num_colors),
      num_players(_num_players),
      board(size * size, -1),
      num_chips(num_players, -1),
      positions(num_players + 1, -1) {
  init();
}

Board Board::Clone() const {
  Board clone(size, num_colors, num_players);
  clone.board = board;
  clone.num_chips = num_chips;
  clone.chips = chips;
  clone.positions = positions;
  return clone;
}


void Board::init() {
  chips.reserve(num_players);
  for (int p = 0; p < num_players; ++p) {
    chips.push_back(std::vector<int>(num_colors, 0));
  }
}

bool Board::InBounds(int row, int col) const {
  return (row >= 0 && row < size && col >= 0 && col < size);
}

void Board::ApplyTrade(std::pair<int, int> players, const Trade& trade) {
  if (trade.giving.empty()) {
    // This is a pass, so don't change the board.
    return;
  }
  SPIEL_CHECK_EQ(trade.giving.size(), num_colors);
  SPIEL_CHECK_EQ(trade.receiving.size(), num_colors);
  for (int i = 0; i < num_colors; ++i) {
    SPIEL_CHECK_LE(trade.giving[i], chips[players.first][i]);
    SPIEL_CHECK_LE(trade.receiving[i], chips[players.second][i]);
    chips[players.first][i] -= trade.giving[i];
    chips[players.second][i] += trade.giving[i];
    chips[players.first][i] += trade.receiving[i];
    chips[players.second][i] -= trade.receiving[i];
  }
}

std::string Board::ToString() const {
  std::string str = absl::StrCat(size, " ", num_colors, " ", num_players, " ");
  for (int i = 0; i < board.size(); ++i) {
    str.push_back(ColorToChar(board[i]));
  }
  absl::StrAppend(&str, " ");
  for (Player p = 0; p < num_players; ++p) {
    absl::StrAppend(&str, ComboToString(chips[p]), " ");
  }
  absl::StrAppend(&str, absl::StrJoin(positions, " "));
  return str;
}

std::string Board::PrettyBoardString() const {
  std::string str;
  for (int r = 0; r < size; ++r) {
    for (int c = 0; c < size; ++c) {
      str.push_back(ColorToChar(board[r * size + c]));
    }
    str.push_back('\n');
  }
  return str;
}

void Board::ParseFromLine(const std::string& line) {
  // Example: 4 5 3 AAEDCABCDAAABBEE AACCCD AAAC BBCEE 14 7 0 2
  std::vector<std::string> parts = absl::StrSplit(line, ' ');
  SPIEL_CHECK_EQ(parts.size(), 3 + 2 * num_players + 2);

  int _size, _colors, _players;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(parts[0], &_size));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(parts[1], &_colors));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(parts[2], &_players));
  SPIEL_CHECK_EQ(_size, size);
  SPIEL_CHECK_EQ(_colors, num_colors);
  SPIEL_CHECK_EQ(_players, num_players);

  SPIEL_CHECK_EQ(parts[3].size(), size * size);
  for (int i = 0; i < parts[3].size(); ++i) {
    board[i] = CharToColor(parts[3].at(i));
  }

  for (Player p = 0; p < num_players; ++p) {
    num_chips[p] = parts[4 + p].length();
    for (int i = 0; i < parts[4 + p].length(); ++i) {
      int chip_color = CharToColor(parts[4 + p].at(i));
      chips[p][chip_color]++;
    }
  }

  for (int i = 0; i < num_players + 1; ++i) {
    SPIEL_CHECK_TRUE(
        absl::SimpleAtoi(parts[4 + num_players + i], &positions[i]));
  }
}

std::string Trade::ToString() const {
  if (giving.empty() || receiving.empty()) {
    return "Pass trade.";
  }
  return absl::StrCat(ComboToString(giving), " for ", ComboToString(receiving));
}

int Trade::DistanceTo(const Trade& other) const {
  int sum = 0;
  if (other.giving.empty() || other.receiving.empty()) {
    // Pass trade is the furthest possible distance.
    return kDefaultTradeDistanceUpperBound + 1;
  }
  for (int i = 0; i < giving.size(); ++i) {
    sum += std::abs(other.giving[i] - giving[i]);
    sum += std::abs(other.receiving[i] - receiving[i]);
  }
  return sum;
}

bool Trade::reduce() {
  for (int i = 0; i < giving.size(); ++i) {
    int min_val = std::min(giving[i], receiving[i]);
    giving[i] -= min_val;
    receiving[i] -= min_val;
  }
  return (std::accumulate(giving.begin(), giving.end(), 0) > 0 &&
          std::accumulate(receiving.begin(), receiving.end(), 0) > 0);
}

Trade::Trade(const std::vector<int> _giving, const std::vector<int> _receiving)
    : giving(_giving), receiving(_receiving) {}

Trade::Trade(const Trade& other)
    : giving(other.giving), receiving(other.receiving) {}

std::string ColoredTrailsState::ActionToString(Player player,
                                               Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome ", move_id);
  } else if (player < kResponderId) {
    return absl::StrCat("Proposer ", player, ": ",
                        parent_game_->LookupTrade(move_id).ToString());
  } else if (player == kResponderId) {
    if (move_id == num_distinct_actions_ - 3) {
      return "Deal: trade with proposer 0";
    } else if (move_id == num_distinct_actions_ - 2) {
      return "Deal: trade with proposer 1";
    } else if (move_id == num_distinct_actions_ - 1) {
      return "No Deal!";
    } else {
      SpielFatalError(absl::StrCat("move_id unrecognized: ", move_id));
    }
  } else {
    SpielFatalError(absl::StrCat("Player and move case unrecognized: ", player,
                                 " ", move_id));
  }
}

bool ColoredTrailsState::IsTerminal() const {
  return cur_player_ == kTerminalPlayerId;
}

std::vector<double> ColoredTrailsState::Returns() const { return returns_; }

std::string ColoredTrailsState::ObservationString(Player player) const {
  return InformationStateString(player);
}

std::string ColoredTrailsState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string str =
      absl::StrCat(board_.PrettyBoardString(), "\n");
  absl::StrAppend(&str, "Player: ", player, "\nPos: ",
                  absl::StrJoin(board_.positions, " "), "\n");
  if (player < kResponderId) {
    absl::StrAppend(&str, "My chips: ", ComboToString(board_.chips[player]),
                    "\n");
    absl::StrAppend(&str, "Responder chips: ",
                    ComboToString(board_.chips[kResponderId]), "\n");
  } else if (player == kResponderId) {
    absl::StrAppend(&str, "P0 chips: ", ComboToString(board_.chips[0]), "\n");
    absl::StrAppend(&str, "P1 chips: ", ComboToString(board_.chips[1]), "\n");
    if (CurrentPlayer() == kResponderId) {
      SPIEL_CHECK_EQ(proposals_.size(), 2);
      absl::StrAppend(&str, "Proposal 0: ", proposals_[0].ToString(), "\n");
      absl::StrAppend(&str, "Proposal 1: ", proposals_[1].ToString(), "\n");
    }
  } else {
    SpielFatalError(absl::StrCat("Bad player id: ", player));
  }
  return str;
}

void ColoredTrailsState::ObservationTensor(Player player,
                                           absl::Span<float> values) const {
  InformationStateTensor(player, values);
}

std::unique_ptr<State> ColoredTrailsState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::vector<std::unique_ptr<State>> candidates;
  const std::vector<Board>& all_boards = parent_game_->AllBoards();

  for (int o = 0; o < all_boards.size(); ++o) {
    if (board_.ToString() != all_boards[o].ToString()) {
      continue;
    }

    std::unique_ptr<State> candidate_state = parent_game_->NewInitialState();
    candidate_state->ApplyAction(o);

    if (player_id == 0) {
      if (candidate_state->InformationStateString(0) ==
          InformationStateString(0)) {
        candidates.push_back(std::move(candidate_state));
      }
    } else if (player_id == 1) {
      // Enumerate legal moves.
      for (Action action : candidate_state->LegalActions()) {
        std::unique_ptr<State> candidate_child = candidate_state->Child(action);
        if (candidate_child->InformationStateString(1) ==
            InformationStateString(1)) {
          candidates.push_back(std::move(candidate_child));
        } else {
          // Player 0's move is hidden. No need to keep trying actions if P1's
          // infostate doesn't match.
          break;
        }
      }
    } else {
      SPIEL_CHECK_EQ(player_id, 2);
      SPIEL_CHECK_EQ(History().size(), 3);
      Action p0_action = History()[1];
      Action p1_action = History()[2];
      // Receiver sees everything, so replay the moves.
      std::vector<Action> legal_actions = candidate_state->LegalActions();
      if (absl::c_find(legal_actions, p0_action) != legal_actions.end()) {
        candidate_state->ApplyAction(p0_action);
        legal_actions = candidate_state->LegalActions();
        if (absl::c_find(legal_actions, p1_action) != legal_actions.end()) {
          candidate_state->ApplyAction(p1_action);
          candidates.push_back(std::move(candidate_state));
        }
      }
    }
  }

  SPIEL_CHECK_GE(candidates.size(), 1);
  if (candidates.size() == 1) {
    return std::move(candidates[0]);
  } else {
    int idx = static_cast<int>(rng() * candidates.size());
    SPIEL_CHECK_LE(idx, candidates.size());
    return std::move(candidates[idx]);
  }
}

void ColoredTrailsState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorSize());
  std::fill(values.begin(), values.end(), 0);

  if (IsChanceNode()) {
    // No observations at chance nodes.
    return;
  }

  int offset = 0;

  // Player.
  values[player] = 1;
  offset += num_players_;

  // Terminal?
  if (IsTerminal()) {
    values[offset] = 1;
  }
  offset += 1;

  // The board
  for (int i = 0; i < board_.board.size(); ++i) {
    values[offset + board_.board[i]] = 1;
    offset += board_.num_colors;
  }

  // Positions
  for (int i = 0; i < board_.positions.size(); ++i) {
    values[offset + board_.positions[i]] = 1;
    offset += board_.size * board_.size;
  }

  // Chips.
  std::array<const std::vector<int>*, 3> chips_ptrs;
  std::vector<int> zeros(board_.num_colors, 0);
  if (player < kResponderId) {
    chips_ptrs[0] = &board_.chips[player];
    chips_ptrs[1] = &zeros;
    chips_ptrs[2] = &board_.chips[kResponderId];
  } else {
    chips_ptrs[0] = &board_.chips[0];
    chips_ptrs[1] = &board_.chips[1];
    chips_ptrs[2] = &board_.chips[kResponderId];
  }
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < board_.num_colors; ++i) {
      for (int j = 0; j <= chips_ptrs[c]->at(i); ++j) {
        values[offset + j] = 1;
      }
      offset += (kNumChipsUpperBound + 1);
    }
  }

  // Proposals
  if (player == kResponderId && CurrentPlayer() == kResponderId) {
    SPIEL_CHECK_EQ(proposals_.size(), 2);
    for (int p : {0, 1}) {
      if (IsPassTrade(proposals_[p])) {
        chips_ptrs[0] = &zeros;
        chips_ptrs[1] = &zeros;
      } else {
        chips_ptrs[0] = &(proposals_[p].giving);
        chips_ptrs[1] = &(proposals_[p].receiving);
      }

      for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < board_.num_colors; ++i) {
          for (int j = 0; j <= chips_ptrs[c]->at(i); ++j) {
            values[offset + j] = 1;
          }
          offset += (kNumChipsUpperBound + 1);
        }
      }
    }
  } else {
    // Proposers have no observations of the proposals.
    // Responder doesn't observe the chips until its their turn.
    offset += (kNumChipsUpperBound + 1) * board_.num_colors * 2 *
        (num_players_ - 1);
  }
  SPIEL_CHECK_EQ(offset, values.size());
}

ColoredTrailsState::ColoredTrailsState(std::shared_ptr<const Game> game,
                                       int board_size, int num_colors)
    : State(game),
      cur_player_(kChancePlayerId),
      parent_game_(down_cast<const ColoredTrailsGame*>(game.get())),
      board_(board_size, num_colors, game->NumPlayers()),
      returns_(game->NumPlayers(), 0) {}

int ColoredTrailsState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void ColoredTrailsState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    const std::vector<Board>& all_boards = parent_game_->AllBoards();
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, all_boards.size());
    board_ = all_boards[action];
    cur_player_ = 0;
  } else if (cur_player_ < kResponderId) {
    proposals_.push_back(parent_game_->LookupTrade(action));
    cur_player_++;

    // Special case when using SetChipsAndProposals, check the future_trade_.
    // If it's now the second player, and there's a future trade queued, apply
    // it.
    if (cur_player_ == 1 &&
        (!future_trade_.giving.empty() || !future_trade_.receiving.empty())) {
      proposals_.push_back(future_trade_);
      cur_player_++;
    }
  } else {
    // Base scores.
    SPIEL_CHECK_EQ(cur_player_, kResponderId);
    for (Player p = 0; p < board_.num_players; ++p) {
      returns_[p] = Score(p, board_).first;
    }

    if (action == parent_game_->ResponderTradeWithPlayerAction(0)) {
      if (!IsPassTrade(proposals_[0])) {
        board_.ApplyTrade({0, kResponderId}, proposals_[0]);
      }
    } else if (action == parent_game_->ResponderTradeWithPlayerAction(1)) {
      if (!IsPassTrade(proposals_[1])) {
        board_.ApplyTrade({1, kResponderId}, proposals_[1]);
      }
    } else if (action == parent_game_->PassAction()) {
      // No trade.
    } else {
      std::string error = absl::StrCat("Invalid action: ", action,
          parent_game_->ActionToString(kResponderId, action), "\n",
                                       ToString());
      SpielFatalErrorWithStateInfo(error, *parent_game_, *this);
    }

    // Gain is final score minus base score.
    for (Player p = 0; p < board_.num_players; ++p) {
      returns_[p] = Score(p, board_).first - returns_[p];
    }

    cur_player_ = kTerminalPlayerId;
  }
}

bool ColoredTrailsState::IsPassTrade(const Trade& trade) const {
  return (trade.giving.empty() && trade.receiving.empty());
}

bool ColoredTrailsState::IsLegalTrade(Player proposer,
                                      const Trade& trade) const {
  return colored_trails::IsLegalTrade(board_, trade, board_.chips[proposer],
                                      board_.chips[kResponderId]);
}

std::vector<Action> ColoredTrailsState::LegalActionsForChips(
    const std::vector<int>& player_chips,
    const std::vector<int>& responder_chips) const {
  // First, check the cache.
  std::string key = absl::StrCat(ComboToString(player_chips), " ",
                                 ComboToString(responder_chips));
  std::vector<Action> actions = parent_game_->LookupTradesCache(key);
  if (!actions.empty()) {
    return actions;
  }

  actions = GenerateLegalActionsForChips(parent_game_, board_, player_chips,
                                         responder_chips);

  // Add these to the cache.
  parent_game_->AddToTradesCache(key, actions);
  return actions;
}

std::vector<Action> ColoredTrailsState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (cur_player_ < kResponderId) {
    return LegalActionsForChips(board_.chips[cur_player_],
                                board_.chips[kResponderId]);
  } else {
    SPIEL_CHECK_EQ(cur_player_, kResponderId);
    // Last three actions correspond to "trade with 0", "trade with 1", and
    // "no trade".
    return {parent_game_->ResponderTradeWithPlayerAction(0),
            parent_game_->ResponderTradeWithPlayerAction(1),
            parent_game_->PassAction()};
  }
}

std::vector<std::pair<Action, double>> ColoredTrailsState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  const int num_boards = parent_game_->AllBoards().size();
  outcomes.reserve(num_boards);
  double uniform_prob = 1.0 / num_boards;
  for (int i = 0; i < num_boards; ++i) {
    outcomes.push_back({i, uniform_prob});
  }
  return outcomes;
}

std::string ColoredTrailsState::ToString() const {
  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str;
  if (MoveNumber() > 0) {
    absl::StrAppend(&str, "Move Number: ", MoveNumber(), "\n",
                    board_.PrettyBoardString(), "\n");
    for (Player p = 0; p < num_players_; ++p) {
      absl::StrAppend(&str, "P", p, " chips: ", ComboToString(board_.chips[p]),
                      "\n");
    }
  }

  absl::StrAppend(&str, "Pos: ", absl::StrJoin(board_.positions, " "), "\n");
  for (int i = 0; i < proposals_.size(); ++i) {
    absl::StrAppend(&str, "Proposal ", i, ": ", proposals_[i].ToString(), "\n");
  }
  return str;
}

std::unique_ptr<State> ColoredTrailsState::Clone() const {
  return std::unique_ptr<State>(new ColoredTrailsState(*this));
}

void ColoredTrailsState::SetChipsAndTradeProposal(
    Player player, std::vector<int> chips, Trade trade,
    std::vector<double>& rng_rolls) {
  // First, check the chips.
  int rng_idx = 0;
  int num_chips = std::accumulate(chips.begin(), chips.end(), 0);

  while (num_chips < kNumChipsLowerBound) {
    std::vector<int> indices;
    for (int i = 0; i < chips.size(); i++) {
      if (chips[i] == 0) {
        indices.push_back(i);
      }
    }
    SPIEL_CHECK_LT(rng_idx, rng_rolls.size());
    int selected_idx =
        indices[static_cast<int>(rng_rolls[rng_idx] * indices.size())];
    chips[selected_idx]++;
    rng_idx++;
    num_chips = std::accumulate(chips.begin(), chips.end(), 0);
  }

  while (num_chips > kNumChipsUpperBound) {
    std::vector<int> indices;
    for (int i = 0; i < chips.size(); i++) {
      if (chips[i] > 0) {
        indices.push_back(i);
      }
    }
    SPIEL_CHECK_LT(rng_idx, rng_rolls.size());
    int selected_idx =
        indices[static_cast<int>(rng_rolls[rng_idx] * indices.size())];
    chips[selected_idx]--;
    rng_idx++;
    num_chips = std::accumulate(chips.begin(), chips.end(), 0);
  }

  board_.chips[player] = chips;
  trade.reduce();

  // Now check if the Trade is legal. If not, chose one of the closest legal
  // ones in edit distance
  if (!IsLegalTrade(player, trade)) {
    std::vector<Trade> closest_trades;
    int lowest_distance = kDefaultTradeDistanceUpperBound + 100;
    std::vector<Action> legal_actions =
        LegalActionsForChips(chips, board_.chips[kResponderId]);
    for (Action action : legal_actions) {
      const Trade& legal_trade = parent_game_->LookupTrade(action);
      int dist = trade.DistanceTo(legal_trade);
      if (dist == lowest_distance) {
        closest_trades.push_back(legal_trade);
      } else if (dist < lowest_distance) {
        lowest_distance = dist;
        closest_trades = {legal_trade};
      }
    }

    if (closest_trades.empty()) {
      std::cout << ToString() << std::endl;
      std::cout << "Trade: " << trade.ToString() << std::endl;
    }

    SPIEL_CHECK_GT(closest_trades.size(), 0);
    if (closest_trades.size() == 1) {
      trade = closest_trades[0];
    } else {
      trade = closest_trades[static_cast<int>(rng_rolls[rng_idx] *
                                              closest_trades.size())];
      rng_idx++;
    }
  }

  if (player == 0) {
    SPIEL_CHECK_NE(cur_player_, 0);
    proposals_[0] = trade;
  } else if (player == 1) {
    SPIEL_CHECK_NE(cur_player_, 1);
    if (cur_player_ == 0) {
      future_trade_ = trade;
    } else {
      proposals_[1] = trade;
    }
  }
}

ColoredTrailsGame::ColoredTrailsGame(const GameParameters& params)
    : Game(kGameType, params),
      num_colors_(ParameterValue<int>("num_colors", kDefaultNumColors)),
      board_size_(ParameterValue<int>("board_size", kDefaultBoardSize)),
      num_players_(ParameterValue<int>("players", kDefaultNumPlayers)) {
  // Only support the 3-player game.
  SPIEL_CHECK_EQ(num_players_, kDefaultNumPlayers);

  std::string filename = ParameterValue<std::string>("boards_file", "");
  if (!filename.empty()) {
    ParseBoardsFile(&all_boards_, filename, num_colors_, board_size_,
                    num_players_);
  } else {
    ParseBoardsString(&all_boards_, kDefaultBoardsString, num_colors_,
                      board_size_, num_players_);
  }
  InitTradeInfo(&trade_info_, num_colors_);
}

int ColoredTrailsGame::NumDistinctActions() const {
  return trade_info_.possible_trades.size() + 3;
}

std::vector<int> ColoredTrailsGame::ObservationTensorShape() const {
  return InformationStateTensorShape();
}

std::vector<int> ColoredTrailsGame::InformationStateTensorShape() const {
  return {
    num_players_ +   // Who is observing
    1 +              // is it terminal?
    board_size_ * board_size_ * num_colors_ +  // board
    board_size_ * board_size_ * (num_players_ + 1) +  // player + flag positions
    // thermometer of bits representation of the chips (proposers + receiver)
    (kNumChipsUpperBound + 1) * num_colors_ * 3 +
    // thermometer of bits representation of the proposals
    // 0 to upperboard of chip combos for each in X for Y, and max two proposals
    (kNumChipsUpperBound + 1) * num_colors_ * 2 * (num_players_ - 1)
  };
}

std::vector<Action> ColoredTrailsGame::LookupTradesCache(
    const std::string& key) const {
  const auto& iter = trades_cache_.find(key);
  if (iter == trades_cache_.end()) {
    return {};
  }
  return iter->second;
}

void ColoredTrailsGame::AddToTradesCache(const std::string& key,
                                         std::vector<Action>& actions) const {
  trades_cache_[key] = actions;
}

bool CheckBoard(const Board& board) {
  std::vector<int> base_scores(board.num_players);
  int min_score = board.size * 100;
  int max_score = board.size * -100;

  for (Player player = 0; player < board.num_players; ++player) {
    std::pair<int, bool> score_and_solved = Score(player, board);
    if (score_and_solved.second) {
      // Cannot be solvable without negotiation.
      return false;
    }
    base_scores[player] = score_and_solved.first;
    min_score = std::min(min_score, base_scores[player]);
    max_score = std::max(max_score, base_scores[player]);
  }

  if (max_score - min_score > kBaseScoreEpsilon) {
    return false;
  }

  // Now check that there exist two trades:
  // - one between player 0 and 2, such that both can reach the goal
  // - one between player 1 and 2, such that both can reach the goal
  for (int proposer : {0, 1}) {
    bool found_trade = false;
    ChipComboIterator iter1(board.chips[proposer]);
    while (!found_trade && !iter1.IsFinished()) {
      std::vector<int> combo1 = iter1.Next();
      ChipComboIterator iter2(board.chips[2]);
      while (!found_trade && !iter2.IsFinished()) {
        std::vector<int> combo2 = iter2.Next();
        // Do the trade and check if both can reach the goal.
        Board board_copy = board;
        Trade trade(combo1, combo2);
        board_copy.ApplyTrade({proposer, 2}, trade);
        std::pair<int, bool> prop_score_and_goal = Score(proposer, board_copy);
        if (prop_score_and_goal.second) {
          std::pair<int, bool> rec_score_and_goal = Score(2, board_copy);
          if (rec_score_and_goal.second) {
            found_trade = true;
          }
        }
      }
    }
    if (!found_trade) {
      return false;
    }
  }

  return true;
}

bool CheckBoardForProposer(const Board& board, Player proposer) {
  std::vector<int> base_scores(board.num_players);
  int min_score = board.size * 100;
  int max_score = board.size * -100;

  std::pair<int, bool> score_and_solved = Score(proposer, board);
  if (score_and_solved.second) {
    // Cannot be solvable without negotiation.
    return false;
  }
  base_scores[proposer] = score_and_solved.first;
  min_score = std::min(min_score, base_scores[proposer]);
  max_score = std::max(max_score, base_scores[proposer]);

  if (max_score - min_score > kBaseScoreEpsilon) {
    return false;
  }

  // Now check that there exist two trades:
  bool found_trade = false;
  ChipComboIterator iter1(board.chips[proposer]);
  while (!found_trade && !iter1.IsFinished()) {
    std::vector<int> combo1 = iter1.Next();
    ChipComboIterator iter2(board.chips[2]);
    while (!found_trade && !iter2.IsFinished()) {
      std::vector<int> combo2 = iter2.Next();
      // Do the trade and check if both can reach the goal.
      Board board_copy = board;
      Trade trade(combo1, combo2);
      board_copy.ApplyTrade({proposer, 2}, trade);
      std::pair<int, bool> prop_score_and_goal = Score(proposer, board_copy);
      if (prop_score_and_goal.second) {
        std::pair<int, bool> rec_score_and_goal = Score(2, board_copy);
        if (rec_score_and_goal.second) {
          found_trade = true;
        }
      }
    }
  }
  if (!found_trade) {
    return false;
  }

  return true;
}


std::pair<Board, Action> ColoredTrailsGame::SampleRandomBoardCompletion(
    int seed, const Board& board, Player player) const {
  std::mt19937 rng(seed);
  Board new_board = board;
  const int max_tries = 1000;
  int tries = 0;

  do {
    tries += 1;
    for (int i = 0; i < new_board.chips[player].size(); ++i) {
      new_board.chips[player][i] = 0;
    }
    int width = kNumChipsUpperBound - kNumChipsLowerBound + 1;
    new_board.num_chips[player] =
        kNumChipsLowerBound + absl::Uniform<int>(rng, 0, width);
    for (int i = 0; i < new_board.num_chips[player]; ++i) {
      int chip = absl::Uniform<int>(rng, 0, new_board.num_colors);
      new_board.chips[player][chip]++;
    }
  } while (!CheckBoardForProposer(new_board, player) && tries < max_tries);
  SPIEL_CHECK_LT(tries, max_tries);

  std::string key = absl::StrCat(ComboToString(new_board.chips[player]), " ",
                                 ComboToString(new_board.chips[kResponderId]));
  std::vector<Action> actions = LookupTradesCache(key);
  if (actions.empty()) {
    actions = GenerateLegalActionsForChips(this, new_board,
                                           new_board.chips[player],
                                           new_board.chips[kResponderId]);
    AddToTradesCache(key, actions);
  }

  Action action = actions[absl::Uniform<int>(rng, 0, actions.size())];
  return {new_board, action};
}


}  // namespace colored_trails
}  // namespace open_spiel
