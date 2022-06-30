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

#include "open_spiel/games/backgammon.h"

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
namespace backgammon {
namespace {

// A few constants to help with the conversion to human-readable string formats.
// TODO: remove these once we've changed kBarPos and kScorePos (see TODO in
// header).
constexpr int kNumBarPosHumanReadable = 25;
constexpr int kNumOffPosHumanReadable = -2;

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

int NumCheckersPerPlayer(const Game* game) {
  return static_cast<const BackgammonGame*>(game)->NumCheckersPerPlayer();
}

// Facts about the game
const GameType kGameType{
    /*short_name=*/"backgammon",
    /*long_name=*/"Backgammon",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/2,
    /*max_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"hyper_backgammon", GameParameter(kDefaultHyperBackgammon)},
     {"scoring_type",
      GameParameter(static_cast<std::string>(kDefaultScoringType))}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BackgammonGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

ScoringType ParseScoringType(const std::string& st_str) {
  if (st_str == "winloss_scoring") {
    return ScoringType::kWinLossScoring;
  } else if (st_str == "enable_gammons") {
    return ScoringType::kEnableGammons;
  } else if (st_str == "full_scoring") {
    return ScoringType::kFullScoring;
  } else {
    SpielFatalError("Unrecognized scoring_type parameter: " + st_str);
  }
}

std::string PositionToString(int pos) {
  switch (pos) {
    case kBarPos:
      return "Bar";
    case kScorePos:
      return "Score";
    case -1:
      return "Pass";
    default:
      return absl::StrCat(pos);
  }
}

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

std::string PositionToStringHumanReadable(int pos) {
  if (pos == kNumBarPosHumanReadable) {
    return "Bar";
  } else if (pos == kNumOffPosHumanReadable) {
    return "Off";
  } else {
    return PositionToString(pos);
  }
}

int BackgammonState::AugmentCheckerMove(CheckerMove* cmove, int player,
                                        int start) const {
  int end = cmove->num;
  if (end != kPassPos) {
    // Not a pass, so work out where the piece finished
    end = start - cmove->num;
    if (end <= 0) {
      end = kNumOffPosHumanReadable;  // Off
    } else if (board_[Opponent(player)]
                     [player == kOPlayerId ? (end - 1) : (kNumPoints - end)] ==
               1) {
      cmove->hit = true;  // Check to see if move is a hit
    }
  }
  return end;
}

std::string BackgammonState::ActionToString(Player player,
                                            Action move_id) const {
  if (player == kChancePlayerId) {
    if (turns_ >= 0) {
      // Normal chance roll.
      return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
    } else {
      // Initial roll to determine who starts.
      const char* starter = (move_id < 15 ? "X starts" : "O starts");
      if (move_id >= 15) {
        move_id -= 15;
      }
      return absl::StrCat("chance outcome ", move_id, " ", starter, ", ",
                          "(roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
    }
  } else {
    // Assemble a human-readable string representation of the move using
    // standard backgammon notation:
    //
    // - Always show the numbering going from Bar->24->0->Off, irrespective of
    //   which player is moving.
    // - Show the start position followed by end position.
    // - Show hits with an asterisk, e.g. 9/7*.
    // - Order the moves by highest number first, e.g. 22/7 10/8 not 10/8 22/7.
    //   Not an official requirement, but seems to be standard convention.
    // - Show duplicate moves as 10/8(2).
    // - Show moves on a single piece as 10/8/5 not 10/8 8/5
    //
    // Note that there are tests to ensure the ActionToString follows this
    // output format. Any changes would need to be reflected in the tests as
    // well.
    std::vector<CheckerMove> cmoves = SpielMoveToCheckerMoves(player, move_id);

    int cmove0_start;
    int cmove1_start;
    if (player == kOPlayerId) {
      cmove0_start = (cmoves[0].pos == kBarPos ? kNumBarPosHumanReadable
                                               : cmoves[0].pos + 1);
      cmove1_start = (cmoves[1].pos == kBarPos ? kNumBarPosHumanReadable
                                               : cmoves[1].pos + 1);
    } else {
      // swap the board numbering round for Player X so player is moving
      // from 24->0
      cmove0_start = (cmoves[0].pos == kBarPos ? kNumBarPosHumanReadable
                                               : kNumPoints - cmoves[0].pos);
      cmove1_start = (cmoves[1].pos == kBarPos ? kNumBarPosHumanReadable
                                               : kNumPoints - cmoves[1].pos);
    }

    // Add hit information and compute whether the moves go off the board.
    int cmove0_end = AugmentCheckerMove(&cmoves[0], player, cmove0_start);
    int cmove1_end = AugmentCheckerMove(&cmoves[1], player, cmove1_start);

    // check for 2 pieces hitting on the same point.
    bool double_hit =
        (cmoves[1].hit && cmoves[0].hit && cmove1_end == cmove0_end);

    std::string returnVal = "";
    if (cmove0_start == cmove1_start &&
        cmove0_end == cmove1_end) {     // same move, show as (2).
      if (cmoves[1].num == kPassPos) {  // Player can't move at all!
        returnVal = "Pass";
      } else {
        returnVal = absl::StrCat(move_id, " - ",
                                 PositionToStringHumanReadable(cmove0_start),
                                 "/", PositionToStringHumanReadable(cmove0_end),
                                 cmoves[0].hit ? "*" : "", "(2)");
      }
    } else if ((cmove0_start < cmove1_start ||
                (cmove0_start == cmove1_start && cmove0_end < cmove1_end) ||
                cmoves[0].num == kPassPos) &&
               cmoves[1].num != kPassPos) {
      // tradition to start with higher numbers first,
      // so swap moves round if this not the case. If
      // there is a pass move, put it last.
      if (cmove1_end == cmove0_start) {
        // Check to see if the same piece is moving for both
        // moves, as this changes the format of the output.
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
            PositionToStringHumanReadable(cmove1_end), cmoves[1].hit ? "*" : "",
            "/", PositionToStringHumanReadable(cmove0_end),
            cmoves[0].hit ? "*" : "");
      } else {
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove1_start), "/",
            PositionToStringHumanReadable(cmove1_end), cmoves[1].hit ? "*" : "",
            " ",
            (cmoves[0].num != kPassPos)
                ? PositionToStringHumanReadable(cmove0_start)
                : "",
            (cmoves[0].num != kPassPos) ? "/" : "",
            PositionToStringHumanReadable(cmove0_end),
            (cmoves[0].hit && !double_hit) ? "*" : "");
      }
    } else {
      if (cmove0_end == cmove1_start) {
        // Check to see if the same piece is moving for both
        // moves, as this changes the format of the output.
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
            PositionToStringHumanReadable(cmove0_end), cmoves[0].hit ? "*" : "",
            "/", PositionToStringHumanReadable(cmove1_end),
            cmoves[1].hit ? "*" : "");
      } else {
        returnVal = absl::StrCat(
            move_id, " - ", PositionToStringHumanReadable(cmove0_start), "/",
            PositionToStringHumanReadable(cmove0_end), cmoves[0].hit ? "*" : "",
            " ",
            (cmoves[1].num != kPassPos)
                ? PositionToStringHumanReadable(cmove1_start)
                : "",
            (cmoves[1].num != kPassPos) ? "/" : "",
            PositionToStringHumanReadable(cmove1_end),
            (cmoves[1].hit && !double_hit) ? "*" : "");
      }
    }

    return returnVal;
  }
}

std::string BackgammonState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void BackgammonState::ObservationTensor(Player player,
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

BackgammonState::BackgammonState(std::shared_ptr<const Game> game,
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
          {std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}),
      turn_history_info_({}) {
  SetupInitialBoard();
}

void BackgammonState::SetupInitialBoard() {
  if (hyper_backgammon_) {
    // https://bkgm.com/variants/HyperBackgammon.html
    // Each player has one checker on each of the furthest points.
    board_[kXPlayerId][0] = board_[kXPlayerId][1] = board_[kXPlayerId][2] = 1;
    board_[kOPlayerId][23] = board_[kOPlayerId][22] = board_[kOPlayerId][21] =
        1;
  } else {
    // Setup the board. First, XPlayer.
    board_[kXPlayerId][0] = 2;
    board_[kXPlayerId][11] = 5;
    board_[kXPlayerId][16] = 3;
    board_[kXPlayerId][18] = 5;
    // OPlayer.
    board_[kOPlayerId][23] = 2;
    board_[kOPlayerId][12] = 5;
    board_[kOPlayerId][7] = 3;
    board_[kOPlayerId][5] = 5;
  }
}

int BackgammonState::board(int player, int pos) const {
  if (pos == kBarPos) {
    return bar_[player];
  } else {
    SPIEL_CHECK_GE(pos, 0);
    SPIEL_CHECK_LT(pos, kNumPoints);
    return board_[player][pos];
  }
}

Player BackgammonState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int BackgammonState::Opponent(int player) const { return 1 - player; }

void BackgammonState::RollDice(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome][0]);
  dice_.push_back(kChanceOutcomeValues[outcome][1]);
}

int BackgammonState::DiceValue(int i) const {
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

void BackgammonState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    turn_history_info_.push_back(TurnHistoryInfo(kChancePlayerId, prev_player_,
                                                 dice_, move, double_turn_,
                                                 false, false));

    if (turns_ == -1) {
      SPIEL_CHECK_TRUE(dice_.empty());
      if (move < 15) {
        // X starts.
        cur_player_ = prev_player_ = kXPlayerId;
      } else {
        // O Starts
        cur_player_ = prev_player_ = kOPlayerId;
        move -= 15;
      }
      RollDice(move);
      turns_ = 0;
      return;
    } else {
      // Normal chance node.
      SPIEL_CHECK_TRUE(dice_.empty());
      RollDice(move);
      cur_player_ = Opponent(prev_player_);
      return;
    }
  }

  // Normal move action.
  std::vector<CheckerMove> moves = SpielMoveToCheckerMoves(cur_player_, move);
  bool first_move_hit = ApplyCheckerMove(cur_player_, moves[0]);
  bool second_move_hit = ApplyCheckerMove(cur_player_, moves[1]);

  turn_history_info_.push_back(
      TurnHistoryInfo(cur_player_, prev_player_, dice_, move, double_turn_,
                      first_move_hit, second_move_hit));

  if (!double_turn_) {
    turns_++;
    if (cur_player_ == kXPlayerId) {
      x_turns_++;
    } else if (cur_player_ == kOPlayerId) {
      o_turns_++;
    }
  }

  prev_player_ = cur_player_;

  // Check for doubles.
  bool extra_turn = false;
  if (!double_turn_ && dice_[0] == dice_[1]) {
    // Check the dice, and unuse them if they are used.
    int dice_used = 0;
    for (int i = 0; i < 2; i++) {
      if (dice_[i] > 6) {
        dice_[i] -= 6;
        dice_used++;
      }
      SPIEL_CHECK_GE(dice_[i], 1);
      SPIEL_CHECK_LE(dice_[i], 6);
    }

    if (dice_used == 2) {
      extra_turn = true;
    }
  }

  if (extra_turn) {
    // Dice have been unused above.
    double_turn_ = true;
  } else {
    cur_player_ = kChancePlayerId;
    dice_.clear();
    double_turn_ = false;
  }
}

void BackgammonState::UndoAction(int player, Action action) {
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

bool BackgammonState::IsHit(Player player, int from_pos, int num) const {
  if (from_pos != kPassPos) {
    int to = PositionFrom(player, from_pos, num);
    return to != kScorePos && board(Opponent(player), to) == 1;
  } else {
    return false;
  }
}

Action BackgammonState::TranslateAction(int from1, int from2,
                                        bool use_high_die_first) const {
  int player = CurrentPlayer();
  int num1 = use_high_die_first ? dice_.at(1) : dice_.at(0);
  int num2 = use_high_die_first ? dice_.at(0) : dice_.at(1);
  bool hit1 = IsHit(player, from1, num1);
  bool hit2 = IsHit(player, from2, num2);
  std::vector<CheckerMove> moves = {{from1, num1, hit1}, {from2, num2, hit2}};
  return CheckerMovesToSpielMove(moves);
}

Action BackgammonState::EncodedBarMove() const { return 24; }

Action BackgammonState::EncodedPassMove() const { return 25; }

Action BackgammonState::CheckerMovesToSpielMove(
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

std::vector<CheckerMove> BackgammonState::SpielMoveToCheckerMoves(
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

std::vector<CheckerMove> BackgammonState::AugmentWithHitInfo(
    int player, const std::vector<CheckerMove> &cmoves) const {
  std::vector<CheckerMove> new_cmoves = cmoves;
  for (int i = 0; i < 2; ++i) {
    new_cmoves[i].hit = IsHit(player, cmoves[i].pos, cmoves[i].num);
  }
  return new_cmoves;
}

bool BackgammonState::IsPosInHome(int player, int pos) const {
  switch (player) {
    case kXPlayerId:
      return (pos >= 18 && pos <= 23);
    case kOPlayerId:
      return (pos >= 0 && pos <= 5);
    default:
      SpielFatalError(absl::StrCat("Unknown player ID: ", player));
  }
}

int BackgammonState::CheckersInHome(int player) const {
  int c = 0;
  for (int i = 0; i < 6; i++) {
    c += board(player, (player == kXPlayerId ? (23 - i) : i));
  }
  return c;
}

bool BackgammonState::AllInHome(int player) const {
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

int BackgammonState::HighestUsableDiceOutcome() const {
  if (UsableDiceOutcome(dice_[1])) {
    return dice_[1];
  } else if (UsableDiceOutcome(dice_[0])) {
    return dice_[0];
  } else {
    return -1;
  }
}

int BackgammonState::FurthestCheckerInHome(int player) const {
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

bool BackgammonState::UsableDiceOutcome(int outcome) const {
  return (outcome >= 1 && outcome <= 6);
}

int BackgammonState::PositionFromBar(int player, int spaces) const {
  if (player == kXPlayerId) {
    return -1 + spaces;
  } else if (player == kOPlayerId) {
    return 24 - spaces;
  } else {
    SpielFatalError(absl::StrCat("Invalid player: ", player));
  }
}

int BackgammonState::PositionFrom(int player, int pos, int spaces) const {
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

int BackgammonState::NumOppCheckers(int player, int pos) const {
  return board_[Opponent(player)][pos];
}

int BackgammonState::GetDistance(int player, int from, int to) const {
  SPIEL_CHECK_NE(from, kScorePos);
  SPIEL_CHECK_NE(to, kScorePos);
  if (from == kBarPos && player == kXPlayerId) {
    from = -1;
  } else if (from == kBarPos && player == kOPlayerId) {
    from = 24;
  }
  return std::abs(to - from);
}

bool BackgammonState::IsOff(int player, int pos) const {
  // Returns if an absolute position is off the board.
  return ((player == kXPlayerId && pos > 23) ||
          (player == kOPlayerId && pos < 0));
}

bool BackgammonState::IsFurther(int player, int pos1, int pos2) const {
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

int BackgammonState::GetToPos(int player, int from_pos, int pips) const {
  if (player == kXPlayerId) {
    return (from_pos == kBarPos ? -1 : from_pos) + pips;
  } else if (player == kOPlayerId) {
    return (from_pos == kBarPos ? 24 : from_pos) - pips;
  } else {
    SpielFatalError(absl::StrCat("Player (", player, ") unrecognized."));
  }
}

// Basic from_to check (including bar checkers).
bool BackgammonState::IsLegalFromTo(int player, int from_pos, int to_pos,
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

std::string BackgammonState::DiceToString(int outcome) const {
  if (outcome > 6) {
    return std::to_string(outcome - 6) + "u";
  } else {
    return std::to_string(outcome);
  }
}

int BackgammonState::CountTotalCheckers(int player) const {
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

int BackgammonState::IsGammoned(int player) const {
  if (hyper_backgammon_) {
    // TODO(author5): remove this when the doubling cube is implemented.
    // In Hyper-backgammon, gammons and backgammons only multiply when the cube
    // has been offered and accepted. However, we do not yet support the cube.
    return false;
  }

  // Does the player not have any checkers borne off?
  return scores_[player] == 0;
}

int BackgammonState::IsBackgammoned(int player) const {
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

std::set<CheckerMove> BackgammonState::LegalCheckerMoves(int player) const {
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

bool BackgammonState::ApplyCheckerMove(int player, const CheckerMove& move) {
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
void BackgammonState::UndoCheckerMove(int player, const CheckerMove& move) {
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
int BackgammonState::RecLegalMoves(
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

std::vector<Action> BackgammonState::ProcessLegalMoves(
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

std::vector<Action> BackgammonState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  SPIEL_CHECK_EQ(CountTotalCheckers(kXPlayerId),
                 NumCheckersPerPlayer(game_.get()));
  SPIEL_CHECK_EQ(CountTotalCheckers(kOPlayerId),
                 NumCheckersPerPlayer(game_.get()));

  std::unique_ptr<State> cstate = this->Clone();
  BackgammonState* state = dynamic_cast<BackgammonState*>(cstate.get());
  std::set<std::vector<CheckerMove>> movelist;
  int max_moves = state->RecLegalMoves({}, &movelist);
  SPIEL_CHECK_GE(max_moves, 0);
  SPIEL_CHECK_LE(max_moves, 2);
  std::vector<Action> legal_actions = ProcessLegalMoves(max_moves, movelist);
  std::sort(legal_actions.begin(), legal_actions.end());
  return legal_actions;
}

std::vector<std::pair<Action, double>> BackgammonState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (turns_ == -1) {
    // Doubles not allowed for the initial roll to determine who goes first.
    // Range 0-14: X goes first, range 15-29: O goes first.
    std::vector<std::pair<Action, double>> outcomes;
    outcomes.reserve(30);
    const double uniform_prob = 1.0 / 30.0;
    for (Action action = 0; action < 30; ++action) {
      outcomes.push_back({action, uniform_prob});
    }
    return outcomes;
  } else {
    return kChanceOutcomes;
  }
}

std::string BackgammonState::ToString() const {
  std::vector<std::string> board_array = {
      "+------|------+", "|......|......|", "|......|......|",
      "|......|......|", "|......|......|", "|......|......|",
      "|      |      |", "|......|......|", "|......|......|",
      "|......|......|", "|......|......|", "|......|......|",
      "+------|------+"};

  // Fill the board.
  for (int pos = 0; pos < 24; pos++) {
    if (board_[kXPlayerId][pos] > 0 || board_[kOPlayerId][pos] > 0) {
      int start_row = (pos < 12 ? 11 : 1);
      int col = (pos < 12 ? (pos >= 6 ? 12 - pos : 13 - pos)
                          : (pos < 18 ? pos - 11 : pos - 10));

      int row_offset = (pos < 12 ? -1 : 1);

      int owner = board_[kXPlayerId][pos] > 0 ? kXPlayerId : kOPlayerId;
      char piece = (owner == kXPlayerId ? 'x' : 'o');
      int my_checkers = board_[owner][pos];

      for (int i = 0; i < 5 && i < my_checkers; i++) {
        board_array[start_row + i * row_offset][col] = piece;
      }

      // Check for special display of >= 10 and >5 pieces
      if (my_checkers >= 10) {
        char lsd = std::to_string(my_checkers % 10)[0];
        // Make sure it reads downward.
        if (pos < 12) {
          board_array[start_row + row_offset][col] = '1';
          board_array[start_row][col] = lsd;
        } else {
          board_array[start_row][col] = '1';
          board_array[start_row + row_offset][col] = lsd;
        }
      } else if (my_checkers > 5) {
        board_array[start_row][col] = std::to_string(my_checkers)[0];
      }
    }
  }

  std::string board_str = absl::StrJoin(board_array, "\n") + "\n";

  // Extra info like whose turn it is etc.
  absl::StrAppend(&board_str, "Turn: ");
  absl::StrAppend(&board_str, CurPlayerToString(cur_player_));
  absl::StrAppend(&board_str, "\n");
  absl::StrAppend(&board_str, "Dice: ");
  absl::StrAppend(&board_str, !dice_.empty() ? DiceToString(dice_[0]) : "");
  absl::StrAppend(&board_str, dice_.size() > 1 ? DiceToString(dice_[1]) : "");
  absl::StrAppend(&board_str, "\n");
  absl::StrAppend(&board_str, "Bar:");
  absl::StrAppend(&board_str,
                  (bar_[kXPlayerId] > 0 || bar_[kOPlayerId] > 0 ? " " : ""));
  for (int p = 0; p < 2; p++) {
    for (int n = 0; n < bar_[p]; n++) {
      absl::StrAppend(&board_str, (p == kXPlayerId ? "x" : "o"));
    }
  }
  absl::StrAppend(&board_str, "\n");
  absl::StrAppend(&board_str, "Scores, X: ", scores_[kXPlayerId]);
  absl::StrAppend(&board_str, ", O: ", scores_[kOPlayerId], "\n");

  return board_str;
}

bool BackgammonState::IsTerminal() const {
  return (scores_[kXPlayerId] == NumCheckersPerPlayer(game_.get()) ||
          scores_[kOPlayerId] == NumCheckersPerPlayer(game_.get()));
}

std::vector<double> BackgammonState::Returns() const {
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

std::unique_ptr<State> BackgammonState::Clone() const {
  return std::unique_ptr<State>(new BackgammonState(*this));
}

void BackgammonState::SetState(int cur_player, bool double_turn,
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

  SPIEL_CHECK_EQ(CountTotalCheckers(kXPlayerId),
                 NumCheckersPerPlayer(game_.get()));
  SPIEL_CHECK_EQ(CountTotalCheckers(kOPlayerId),
                 NumCheckersPerPlayer(game_.get()));
}

BackgammonGame::BackgammonGame(const GameParameters& params)
    : Game(kGameType, params),
      scoring_type_(
          ParseScoringType(ParameterValue<std::string>("scoring_type"))),
      hyper_backgammon_(ParameterValue<bool>("hyper_backgammon")) {}

double BackgammonGame::MaxUtility() const {
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

int BackgammonGame::NumCheckersPerPlayer() const {
  if (hyper_backgammon_) {
    return 3;
  } else {
    return kNumCheckersPerPlayer;
  }
}

}  // namespace backgammon
}  // namespace open_spiel
