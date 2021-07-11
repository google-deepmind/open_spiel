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

#include "open_spiel/games/parcheesi.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace parcheesi {
namespace {


const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 18),
    std::pair<Action, double>(1, 1.0 / 18),
    std::pair<Action, double>(2, 1.0 / 18),
    std::pair<Action, double>(3, 1.0 / 18),
    std::pair<Action, double>(4, 1.0 / 18),
    std::pair<Action, double>(5, 1.0 / 18),
    std::pair<Action, double>(6, 1.0 / 18),
    std::pair<Action, double>(7, 1.0 / 18),
    std::pair<Action, double>(8, 1.0 / 18),
    std::pair<Action, double>(9, 1.0 / 18),
    std::pair<Action, double>(10, 1.0 / 18),
    std::pair<Action, double>(11, 1.0 / 18),
    std::pair<Action, double>(12, 1.0 / 18),
    std::pair<Action, double>(13, 1.0 / 18),
    std::pair<Action, double>(14, 1.0 / 18),
    std::pair<Action, double>(15, 1.0 / 36),
    std::pair<Action, double>(16, 1.0 / 36),
    std::pair<Action, double>(17, 1.0 / 36),
    std::pair<Action, double>(18, 1.0 / 36),
    std::pair<Action, double>(19, 1.0 / 36),
    std::pair<Action, double>(20, 1.0 / 36),
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};


// Facts about the game
const GameType kGameType{
    /*short_name=*/"parcheesi",
    /*long_name=*/"Parcheesi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/4,
    /*max_num_players=*/4,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ParcheesiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace


std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case kXPlayerId:
      return "x";
    case kOPlayerId:
      return "o";
    case kChancePlayerId:
      return "*";
    case kTerminalPlayerId:
      return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}


std::string ParcheesiState::ActionToString(Player player,
                                            Action move_id) const {
  if(player == kChancePlayerId){
    return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
  }
  return absl::StrCat("player ", player, " move: ", move_id);
}

std::string ParcheesiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ParcheesiState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int opponent = Opponent(player);
  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();
  // The format of this vector is described in Section 3.4 of "G. Tesauro,
  // Practical issues in temporal-difference learning, 1994."
  // https://link.springer.com/article/10.1007/BF00992697
  for (int count : board_[player]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  for (int count : board_[opponent]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  *value_it++ = (bar_[player]);
  *value_it++ = (scores_[player]);
  *value_it++ = ((cur_player_ == player) ? 1 : 0);

  *value_it++ = (bar_[opponent]);
  *value_it++ = (scores_[opponent]);
  *value_it++ = ((cur_player_ == opponent) ? 1 : 0);

  SPIEL_CHECK_EQ(value_it, values.end());
}

ParcheesiState::ParcheesiState(std::shared_ptr<const Game> game,
                                 ScoringType scoring_type,
                                 bool hyper_backgammon)
    : State(game),
      scoring_type_(scoring_type),
      hyper_backgammon_(hyper_backgammon),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      x_turns_(0),
      o_turns_(0),
      double_turn_(false),
      dice_({}),
      bar_({0, 0}),
      scores_({0, 0}),
      board_(
          {std::vector<int>(kNumPos, 0), std::vector<int>(kNumPos, 0), std::vector<int>(kNumPos, 0), std::vector<int>(kNumPos, 0)}),
      turn_history_info_({}) {
  SetupInitialBoard();
}

void ParcheesiState::SetupInitialBoard() {
  for(int i = 0; i < 4; i++){
    board_[i][0] = 4;
  }  
}

int ParcheesiState::board(int player, int pos) const {
  if (pos == kBarPos) {
    return bar_[player];
  } else {
    SPIEL_CHECK_GE(pos, 0);
    SPIEL_CHECK_LT(pos, kNumPoints);
    return board_[player][pos];
  }
}

Player ParcheesiState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int ParcheesiState::Opponent(int player) const { return 1 - player; }

void ParcheesiState::RollDice(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome][0]);
  dice_.push_back(kChanceOutcomeValues[outcome][1]);
}

int ParcheesiState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());

  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else if (dice_[i] >= 7 && dice_[i] <= 12) {
    // This die is marked as chosen, so return its proper value.
    // Note: dice are only marked as chosen during the legal moves enumeration.
    return dice_[i] - 6;
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

void ParcheesiState::DoApplyAction(Action move) {

  if (IsChanceNode()) {
    turn_history_info_.push_back(TurnHistoryInfo(kChancePlayerId, prev_player_,
                                                 dice_, move, double_turn_,
                                                 false, false));
    SPIEL_CHECK_TRUE(dice_.empty());
    RollDice(move);
    cur_player_ = NextPlayerRoundRobin(prev_player_, num_players_);
    return;
  }

  board_[cur_player_][0] -= move;
  board_[cur_player_][1] += move;

  prev_player_ = cur_player_;
  cur_player_ = kChancePlayerId;
  dice_.clear();
}

void ParcheesiState::UndoAction(int player, Action action) {
  {
    const TurnHistoryInfo& thi = turn_history_info_.back();
    SPIEL_CHECK_EQ(thi.player, player);
    SPIEL_CHECK_EQ(action, thi.action);
    cur_player_ = thi.player;
    prev_player_ = thi.prev_player;
    dice_ = thi.dice;
    double_turn_ = thi.double_turn;
    if (player != kChancePlayerId) {
      std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(player, action);
      SPIEL_CHECK_EQ(moves.size(), 2);
      moves[0].hit = thi.first_move_hit;
      moves[1].hit = thi.second_move_hit;
      UndoCheckerMove(player, moves[1]);
      UndoCheckerMove(player, moves[0]);
      turns_--;
      if (!double_turn_) {
        if (player == kXPlayerId) {
          x_turns_--;
        } else if (player == kOPlayerId) {
          o_turns_--;
        }
      }
    }
  }
  turn_history_info_.pop_back();
  history_.pop_back();
  --move_number_;
}

bool ParcheesiState::IsHit(Player player, int from_pos, int num) const {
  if (from_pos != kPassPos) {
    int to = PositionFrom(player, from_pos, num);
    return to != kScorePos && board(Opponent(player), to) == 1;
  } else {
    return false;
  }
}

Action ParcheesiState::TranslateAction(int from1, int from2,
                                        bool use_high_die_first) const {
  int player = CurrentPlayer();
  int num1 = use_high_die_first ? dice_.at(1) : dice_.at(0);
  int num2 = use_high_die_first ? dice_.at(0) : dice_.at(1);
  bool hit1 = IsHit(player, from1, num1);
  bool hit2 = IsHit(player, from2, num2);
  std::vector<CheckerMove> moves = {{from1, num1, hit1}, {from2, num2, hit2}};
  return CheckerMovesToSpielMove(moves);
}

Action ParcheesiState::EncodedBarMove() const { return 24; }

Action ParcheesiState::EncodedPassMove() const { return 25; }

Action ParcheesiState::CheckerMovesToSpielMove(
    const std::vector<CheckerMove>& moves) const {
  SPIEL_CHECK_LE(moves.size(), 2);
  int dig0 = EncodedPassMove();
  int dig1 = EncodedPassMove();
  bool high_roll_first = false;
  int high_roll = DiceValue(0) >= DiceValue(1) ? DiceValue(0) : DiceValue(1);

  if (!moves.empty()) {
    int pos1 = moves[0].pos;
    if (pos1 == kBarPos) {
      pos1 = EncodedBarMove();
    }
    if (pos1 != kPassPos) {
      int num1 = moves[0].num;
      dig0 = pos1;
      high_roll_first = num1 == high_roll;
    }
  }

  if (moves.size() > 1) {
    int pos2 = moves[1].pos;
    if (pos2 == kBarPos) {
      pos2 = EncodedBarMove();
    }
    if (pos2 != kPassPos) {
      dig1 = pos2;
    }
  }

  Action move = dig1 * 26 + dig0;
  if (!high_roll_first) {
    move += 676;  // 26**2
  }
  SPIEL_CHECK_GE(move, 0);
  SPIEL_CHECK_LT(move, kNumDistinctActions);
  return move;
}

std::vector<CheckerMove> ParcheesiState::SpielMoveToCheckerMoves(
    int player, Action spiel_move) const {
  SPIEL_CHECK_GE(spiel_move, 0);
  SPIEL_CHECK_LT(spiel_move, kNumDistinctActions);

  bool high_roll_first = spiel_move < 676;
  if (!high_roll_first) {
    spiel_move -= 676;
  }

  std::vector<Action> digits = {spiel_move % 26, spiel_move / 26};
  std::vector<CheckerMove> cmoves;
  int high_roll = DiceValue(0) >= DiceValue(1) ? DiceValue(0) : DiceValue(1);
  int low_roll = DiceValue(0) < DiceValue(1) ? DiceValue(0) : DiceValue(1);

  for (int i = 0; i < 2; ++i) {
    SPIEL_CHECK_GE(digits[i], 0);
    SPIEL_CHECK_LE(digits[i], 25);

    int num = -1;
    if (i == 0) {
      num = high_roll_first ? high_roll : low_roll;
    } else {
      num = high_roll_first ? low_roll : high_roll;
    }
    SPIEL_CHECK_GE(num, 1);
    SPIEL_CHECK_LE(num, 6);

    if (digits[i] == EncodedPassMove()) {
      cmoves.push_back(CheckerMove(kPassPos, -1, false));
    } else {
      cmoves.push_back(CheckerMove(
          digits[i] == EncodedBarMove() ? kBarPos : digits[i], num, false));
    }
  }

  return cmoves;
}

std::vector<CheckerMove> ParcheesiState::AugmentWithHitInfo(
    int player, const std::vector<CheckerMove> &cmoves) const {
  std::vector<CheckerMove> new_cmoves = cmoves;
  for (int i = 0; i < 2; ++i) {
    new_cmoves[i].hit = IsHit(player, cmoves[i].pos, cmoves[i].num);
  }
  return new_cmoves;
}

bool ParcheesiState::IsPosInHome(int player, int pos) const {
  switch (player) {
    case kXPlayerId:
      return (pos >= 18 && pos <= 23);
    case kOPlayerId:
      return (pos >= 0 && pos <= 5);
    default:
      SpielFatalError(absl::StrCat("Unknown player ID: ", player));
  }
}

int ParcheesiState::CheckersInHome(int player) const {
  int c = 0;
  for (int i = 0; i < 6; i++) {
    c += board(player, (player == kXPlayerId ? (23 - i) : i));
  }
  return c;
}

bool ParcheesiState::AllInHome(int player) const {
  if (bar_[player] > 0) {
    return false;
  }

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LE(player, 1);

  // Looking for any checkers outside home.
  // --> XPlayer scans 0-17.
  // --> OPlayer scans 6-23.
  int scan_start = (player == kXPlayerId ? 0 : 6);
  int scan_end = (player == kXPlayerId ? 17 : 23);

  for (int i = scan_start; i <= scan_end; ++i) {
    if (board_[player][i] > 0) {
      return false;
    }
  }

  return true;
}

int ParcheesiState::HighestUsableDiceOutcome() const {
  if (UsableDiceOutcome(dice_[1])) {
    return dice_[1];
  } else if (UsableDiceOutcome(dice_[0])) {
    return dice_[0];
  } else {
    return -1;
  }
}

int ParcheesiState::FurthestCheckerInHome(int player) const {
  // Looking for any checkers in home.
  // --> XPlayer scans 23 -> 18
  // --> OPlayer scans  0 -> 5
  int scan_start = (player == kXPlayerId ? 23 : 0);
  int scan_end = (player == kXPlayerId ? 17 : 6);
  int inc = (player == kXPlayerId ? -1 : 1);

  int furthest = (player == kXPlayerId ? 24 : -1);

  for (int i = scan_start; i != scan_end; i += inc) {
    if (board_[player][i] > 0) {
      furthest = i;
    }
  }

  if (furthest == 24 || furthest == -1) {
    return -1;
  } else {
    return furthest;
  }
}

bool ParcheesiState::UsableDiceOutcome(int outcome) const {
  return (outcome >= 1 && outcome <= 6);
}

int ParcheesiState::PositionFromBar(int player, int spaces) const {
  if (player == kXPlayerId) {
    return -1 + spaces;
  } else if (player == kOPlayerId) {
    return 24 - spaces;
  } else {
    SpielFatalError(absl::StrCat("Invalid player: ", player));
  }
}

int ParcheesiState::PositionFrom(int player, int pos, int spaces) const {
  if (pos == kBarPos) {
    return PositionFromBar(player, spaces);
  }

  if (player == kXPlayerId) {
    int new_pos = pos + spaces;
    return (new_pos > 23 ? kScorePos : new_pos);
  } else if (player == kOPlayerId) {
    int new_pos = pos - spaces;
    return (new_pos < 0 ? kScorePos : new_pos);
  } else {
    SpielFatalError(absl::StrCat("Invalid player: ", player));
  }
}

int ParcheesiState::NumOppCheckers(int player, int pos) const {
  return board_[Opponent(player)][pos];
}

int ParcheesiState::GetDistance(int player, int from, int to) const {
  SPIEL_CHECK_NE(from, kScorePos);
  SPIEL_CHECK_NE(to, kScorePos);
  if (from == kBarPos && player == kXPlayerId) {
    from = -1;
  } else if (from == kBarPos && player == kOPlayerId) {
    from = 24;
  }
  return std::abs(to - from);
}

bool ParcheesiState::IsOff(int player, int pos) const {
  // Returns if an absolute position is off the board.
  return ((player == kXPlayerId && pos > 23) ||
          (player == kOPlayerId && pos < 0));
}

bool ParcheesiState::IsFurther(int player, int pos1, int pos2) const {
  if (pos1 == pos2) {
    return false;
  }

  if (pos1 == kBarPos) {
    return true;
  }

  if (pos2 == kBarPos) {
    return false;
  }

  if (pos1 == kPassPos) {
    return false;
  }

  if (pos2 == kPassPos) {
    return false;
  }

  return ((player == kXPlayerId && pos1 < pos2) ||
          (player == kOPlayerId && pos1 > pos2));
}

int ParcheesiState::GetToPos(int player, int from_pos, int pips) const {
  if (player == kXPlayerId) {
    return (from_pos == kBarPos ? -1 : from_pos) + pips;
  } else if (player == kOPlayerId) {
    return (from_pos == kBarPos ? 24 : from_pos) - pips;
  } else {
    SpielFatalError(absl::StrCat("Player (", player, ") unrecognized."));
  }
}

// Basic from_to check (including bar checkers).
bool ParcheesiState::IsLegalFromTo(int player, int from_pos, int to_pos,
                                    int my_checkers_from,
                                    int opp_checkers_to) const {
  // Must have at least one checker the from position.
  if (my_checkers_from == 0) {
    return false;
  }

  if (opp_checkers_to > 1) {
    return false;
  }

  // Quick validity checks out of the way. This appears to be a valid move.
  // Now, must check: if there are moves on this player's bar, they must move
  // them first, and if there are no legal moves out of the bar, the player
  // loses their turn.
  int my_bar_checkers = board(player, kBarPos);
  if (my_bar_checkers > 0 && from_pos != kBarPos) {
    return false;
  }

  // If this is a scoring move, then check that all this player's checkers are
  // either scored or home.
  if (to_pos < 0 || to_pos > 23) {
    if ((CheckersInHome(player) + scores_[player]) != 15) {
      return false;
    }

    // If it's not *exactly* the right amount, then we have to do a check to see
    // if there exist checkers further from home, as those must be moved first.
    if (player == kXPlayerId && to_pos > 24) {
      for (int pos = from_pos - 1; pos >= 18; pos--) {
        if (board(player, pos) > 0) {
          return false;
        }
      }
    } else if (player == kOPlayerId && to_pos < -1) {
      for (int pos = from_pos + 1; pos <= 5; pos++) {
        if (board(player, pos) > 0) {
          return false;
        }
      }
    }
  }

  return true;
}

std::string ParcheesiState::DiceToString(int outcome) const {
  if (outcome > 6) {
    return std::to_string(outcome - 6) + "u";
  } else {
    return std::to_string(outcome);
  }
}

int ParcheesiState::CountTotalCheckers(int player) const {
  int total = 0;
  for (int i = 0; i < 24; ++i) {
    SPIEL_CHECK_GE(board_[player][i], 0);
    total += board_[player][i];
  }
  SPIEL_CHECK_GE(bar_[player], 0);
  total += bar_[player];
  SPIEL_CHECK_GE(scores_[player], 0);
  total += scores_[player];
  return total;
}

int ParcheesiState::IsGammoned(int player) const {
  if (hyper_backgammon_) {
    // TODO(author5): remove this when the doubling cube is implemented.
    // In Hyper-backgammon, gammons and backgammons only multiply when the cube
    // has been offered and accepted. However, we do not yet support the cube.
    return false;
  }

  // Does the player not have any checkers borne off?
  return scores_[player] == 0;
}

int ParcheesiState::IsBackgammoned(int player) const {
  if (hyper_backgammon_) {
    // TODO(author5): remove this when the doubling cube is implemented.
    // In Hyper-backgammon, gammons and backgammons only multiply when the cube
    // has been offered and accepted. However, we do not yet support the cube.
    return false;
  }

  // Does the player not have any checkers borne off and either has a checker
  // still in the bar or still in the opponent's home?
  if (scores_[player] > 0) {
    return false;
  }

  if (bar_[player] > 0) {
    return true;
  }

  // XPlayer scans 0-5.
  // OPlayer scans 18-23.
  int scan_start = (player == kXPlayerId ? 0 : 18);
  int scan_end = (player == kXPlayerId ? 5 : 23);

  for (int i = scan_start; i <= scan_end; ++i) {
    if (board_[player][i] > 0) {
      return true;
    }
  }

  return false;
}

std::set<CheckerMove> ParcheesiState::LegalCheckerMoves(int player) const {
  std::set<CheckerMove> moves;

  if (bar_[player] > 0) {
    // If there are any checkers are the bar, must move them out first.
    for (int outcome : dice_) {
      if (UsableDiceOutcome(outcome)) {
        int pos = PositionFromBar(player, outcome);
        if (NumOppCheckers(player, pos) <= 1) {
          bool hit = NumOppCheckers(player, pos) == 1;
          moves.insert(CheckerMove(kBarPos, outcome, hit));
        }
      }
    }
    return moves;
  }

  // Regular board moves.
  bool all_in_home = AllInHome(player);
  for (int i = 0; i < kNumPoints; ++i) {
    if (board_[player][i] > 0) {
      for (int outcome : dice_) {
        if (UsableDiceOutcome(outcome)) {
          int pos = PositionFrom(player, i, outcome);
          if (pos == kScorePos && all_in_home) {
            // Check whether a bear off move is legal.

            // It is ok to bear off if all the checkers are at home and the
            // point being used to move from exactly matches the distance from
            // just stepping off the board.
            if ((player == kXPlayerId && i + outcome == 24) ||
                (player == kOPlayerId && i - outcome == -1)) {
              moves.insert(CheckerMove(i, outcome, false));
            } else {
              // Otherwise, a die can only be used to move a checker off if
              // there are no checkers further than it in the player's home.
              if (i == FurthestCheckerInHome(player)) {
                moves.insert(CheckerMove(i, outcome, false));
              }
            }
          } else if (pos != kScorePos && NumOppCheckers(player, pos) <= 1) {
            // Regular move.
            bool hit = NumOppCheckers(player, pos) == 1;
            moves.insert(CheckerMove(i, outcome, hit));
          }
        }
      }
    }
  }
  return moves;
}

bool ParcheesiState::ApplyCheckerMove(int player, const CheckerMove& move) {
  // Pass does nothing.
  if (move.pos < 0) {
    return false;
  }

  // First, remove the checker.
  int next_pos = -1;
  if (move.pos == kBarPos) {
    bar_[player]--;
    next_pos = PositionFromBar(player, move.num);
  } else {
    board_[player][move.pos]--;
    next_pos = PositionFrom(player, move.pos, move.num);
  }

  // Mark the die as used.
  for (int i = 0; i < 2; ++i) {
    if (dice_[i] == move.num) {
      dice_[i] += 6;
      break;
    }
  }

  // Now add the checker (or score).
  if (next_pos == kScorePos) {
    scores_[player]++;
  } else {
    board_[player][next_pos]++;
  }

  bool hit = false;
  // If there was a hit, remove opponent's piece and add to bar.
  // Note: the move.hit will only be properly set during the legal moves search,
  // so we have to also check here if there is a hit candidate.
  if (move.hit ||
      (next_pos != kScorePos && board_[Opponent(player)][next_pos] == 1)) {
    hit = true;
    board_[Opponent(player)][next_pos]--;
    bar_[Opponent(player)]++;
  }

  return hit;
}

// Undoes a checker move. Important note: this checkermove needs to have
// move.hit set from the history to properly undo a move (this information is
// not tracked in the action value).
void ParcheesiState::UndoCheckerMove(int player, const CheckerMove& move) {
  // Undoing a pass does nothing
  if (move.pos < 0) {
    return;
  }

  // First, figure out the next position.
  int next_pos = -1;
  if (move.pos == kBarPos) {
    next_pos = PositionFromBar(player, move.num);
  } else {
    next_pos = PositionFrom(player, move.pos, move.num);
  }

  // If there was a hit, take it out of the opponent's bar and put it back
  // onto the next position.
  if (move.hit) {
    bar_[Opponent(player)]--;
    board_[Opponent(player)][next_pos]++;
  }

  // Remove the moved checker or decrement score.
  if (next_pos == kScorePos) {
    scores_[player]--;
  } else {
    board_[player][next_pos]--;
  }

  // Mark the die as unused.
  for (int i = 0; i < 2; ++i) {
    if (dice_[i] == move.num + 6) {
      dice_[i] -= 6;
      break;
    }
  }

  // Finally, return back the checker to its original position.
  if (move.pos == kBarPos) {
    bar_[player]++;
  } else {
    board_[player][move.pos]++;
  }
}

// Returns the maximum move size (2, 1, or 0)
int ParcheesiState::RecLegalMoves(
    std::vector<CheckerMove> moveseq,
    std::set<std::vector<CheckerMove>>* movelist) {
  if (moveseq.size() == 2) {
    movelist->insert(moveseq);
    return moveseq.size();
  }

  std::set<CheckerMove> moves_here = LegalCheckerMoves(cur_player_);

  if (moves_here.empty()) {
    movelist->insert(moveseq);
    return moveseq.size();
  }

  int max_moves = -1;
  for (const auto& move : moves_here) {
    moveseq.push_back(move);
    ApplyCheckerMove(cur_player_, move);
    int child_max = RecLegalMoves(moveseq, movelist);
    UndoCheckerMove(cur_player_, move);
    max_moves = std::max(child_max, max_moves);
    moveseq.pop_back();
  }

  return max_moves;
}

std::vector<Action> ParcheesiState::ProcessLegalMoves(
    int max_moves, const std::set<std::vector<CheckerMove>>& movelist) const {
  if (max_moves == 0) {
    SPIEL_CHECK_EQ(movelist.size(), 1);
    SPIEL_CHECK_TRUE(movelist.begin()->empty());

    // Passing is always a legal move!
    return {CheckerMovesToSpielMove(
        {{kPassPos, -1, false}, {kPassPos, -1, false}})};
  }

  // Rule 2 in Movement of Checkers:
  // A player must use both numbers of a roll if this is legally possible (or
  // all four numbers of a double). When only one number can be played, the
  // player must play that number. Or if either number can be played but not
  // both, the player must play the larger one. When neither number can be used,
  // the player loses his turn. In the case of doubles, when all four numbers
  // cannot be played, the player must play as many numbers as he can.
  std::vector<Action> legal_actions;
  int max_roll = -1;
  for (const auto& move : movelist) {
    if (max_moves == 2) {
      // Only add moves that are size 2.
      if (move.size() == 2) {
        legal_actions.push_back(CheckerMovesToSpielMove(move));
      }
    } else if (max_moves == 1) {
      // We are just finding the maximum roll.
      max_roll = std::max(max_roll, move[0].num);
    }
  }

  if (max_moves == 1) {
    // Another round to add those that have the max die roll.
    for (const auto& move : movelist) {
      if (move[0].num == max_roll) {
        legal_actions.push_back(CheckerMovesToSpielMove(move));
      }
    }
  }

  SPIEL_CHECK_FALSE(legal_actions.empty());
  return legal_actions;
}

std::vector<Action> ParcheesiState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  if(dice_[0] == 5 && dice_[1] == 5){
    return {2};
  }
  if(dice_[0] == 5 || dice_[1] == 5 || dice_[0] + dice_[1] == 5){
    return {1};
  }
  return{0};
}

std::vector<std::pair<Action, double>> ParcheesiState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return kChanceOutcomes;
}

std::string ParcheesiState::ToString() const {
  std::string board_str = "";
  std::vector<std::string> colors = {"r", "g", "b", "y"};
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < board_[i][0]; j++){
      absl::StrAppend(&board_str, colors[i]);
    }    
  }
  absl::StrAppend(&board_str, " - ");
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < board_[i][1]; j++){
      absl::StrAppend(&board_str, colors[i]);
    }    
  }  
  return board_str;
}

bool ParcheesiState::IsTerminal() const {
  for(int i = 0; i < 4; i++)
    if(board_[i][1] >= 4)
      return true;
  return false;
}

std::vector<double> ParcheesiState::Returns() const {
  int winner = -1;
  int loser = -1;
  if (scores_[kXPlayerId] == 15) {
    winner = kXPlayerId;
    loser = kOPlayerId;
  } else if (scores_[kOPlayerId] == 15) {
    winner = kOPlayerId;
    loser = kXPlayerId;
  } else {
    return {0.0, 0.0};
  }

  // Magnify the util based on the scoring rules for this game.
  int util_mag = 1;
  switch (scoring_type_) {
    case ScoringType::kWinLossScoring:
    default:
      break;

    case ScoringType::kEnableGammons:
      util_mag = (IsGammoned(loser) ? 2 : 1);
      break;

    case ScoringType::kFullScoring:
      util_mag = (IsBackgammoned(loser) ? 3 : IsGammoned(loser) ? 2 : 1);
      break;
  }

  std::vector<double> returns(kNumPlayers);
  returns[winner] = util_mag;
  returns[loser] = -util_mag;
  return returns;
}

std::unique_ptr<State> ParcheesiState::Clone() const {
  return std::unique_ptr<State>(new ParcheesiState(*this));
}

void ParcheesiState::SetState(int cur_player, bool double_turn,
                               const std::vector<int>& dice,
                               const std::vector<int>& bar,
                               const std::vector<int>& scores,
                               const std::vector<std::vector<int>>& board) {
  cur_player_ = cur_player;
  double_turn_ = double_turn;
  dice_ = dice;
  bar_ = bar;
  scores_ = scores;
  board_ = board;

}

ParcheesiGame::ParcheesiGame(const GameParameters& params)
    : Game(kGameType, params) {}

double ParcheesiGame::MaxUtility() const {
  if (hyper_backgammon_) {
    // We do not have the cube implemented, so Hyper-backgammon us currently
    // restricted to a win-loss game regardless of the scoring type.
    return 1;
  }

  switch (scoring_type_) {
    case ScoringType::kWinLossScoring:
      return 1;
    case ScoringType::kEnableGammons:
      return 2;
    case ScoringType::kFullScoring:
      return 3;
    default:
      SpielFatalError("Unknown scoring_type");
  }
}


}  // namespace parcheesi
}  // namespace open_spiel
