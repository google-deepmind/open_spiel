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

#include <algorithm>
#include <cmath>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/colored_trails.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace colored_trails {
namespace {

constexpr int kNumDirections = 4;
constexpr std::array<int, kNumDirections> kRowOffsets = {-1, 0, 1, 0};
constexpr std::array<int, kNumDirections> kColumnOffsets = {0, -1, 0, 1};

void InitChipCombosRec(TradeInfo* trade_info, int num_colors,
                       std::string cur_combo_str) {
  if (cur_combo_str.length() > 0 &&
      cur_combo_str.length() <= kNumChipsUpperBound) {
    trade_info->chip_combinations.push_back(
        ComboStringToCombo(cur_combo_str, num_colors));
  } else if (cur_combo_str.length() > kNumChipsUpperBound) {
    return;
  }

  int last_color =
      (cur_combo_str.empty() ? 0 : CharToColor(cur_combo_str.back()));
  for (int c = last_color; c < num_colors; ++c) {
    std::string child = cur_combo_str;
    child.push_back(ColorToChar(c));
    InitChipCombosRec(trade_info, num_colors, child);
  }
}

int ManhattanDistance(const Board& board, int pos1, int pos2) {
  int r1 = pos1 / board.size, c1 = pos1 % board.size;
  int r2 = pos2 / board.size, c2 = pos2 % board.size;
  return std::abs(r2 - r1) + std::abs(c2 - c1);
}

int CurrentScore(Player p, const Board& board) {
  int score = std::accumulate(board.chips[p].begin(), board.chips[p].end(), 0) *
              kLeftoverChipScore;
  score += kFlagPenaltyPerCell *
           ManhattanDistance(board, board.positions[p], board.positions.back());
  return score;
}

int ScoreRec(Player player, const Board& board, bool* solved) {
  int score = CurrentScore(player, board);
  int row = board.positions[player] / board.size;
  int col = board.positions[player] % board.size;

  if (board.positions.back() == board.positions[player]) {
    // We found the goal. This has to be the maximal score: terminate recursion.
    *solved = true;
    return score;
  }

  for (int dir = 0; dir < kNumDirections; ++dir) {
    int rp = row + kRowOffsets[dir];
    int cp = col + kColumnOffsets[dir];
    if (board.InBounds(rp, cp)) {  // Check this position is in bounds.
      int pos = rp * board.size + cp;
      int color = board.board[pos];
      if (board.chips[player][color] > 0) {
        // If this player has a chip to travel here, then move them and call
        // score on the child board.
        Board child_board = board;
        child_board.chips[player][color]--;
        child_board.positions[player] = pos;
        int child_score = ScoreRec(player, child_board, solved);
        score = std::max(score, child_score);
      }
    }
  }

  return score;
}

}  // namespace

ChipComboIterator::ChipComboIterator(const std::vector<int>& chips)
    : chips_(chips), cur_combo_(chips.size(), 0) {
  SPIEL_CHECK_GT(std::accumulate(chips_.begin(), chips_.end(), 0), 0);
}

bool ChipComboIterator::IsFinished() const {
  // If every digit is maximized, we are done.
  return cur_combo_ == chips_;
}

std::vector<int> ChipComboIterator::Next() {
  // Try to increase the left-most non-maximized chip with non-zero chips. Then
  // reset every digit to the left of it with nonzero chips.
  for (int inc_idx = 0; inc_idx < chips_.size(); ++inc_idx) {
    if (cur_combo_[inc_idx] < chips_[inc_idx]) {
      cur_combo_[inc_idx]++;
      for (int j = inc_idx - 1; j >= 0; --j) {
        cur_combo_[j] = 0;
      }
      break;
    }
  }
  return cur_combo_;
}

std::vector<int> ComboStringToCombo(const std::string& combo_str,
                                    int num_colors) {
  std::vector<int> combo(num_colors, 0);
  for (int i = 0; i < combo_str.length(); ++i) {
    int color = CharToColor(combo_str[i]);
    combo[color]++;
  }
  return combo;
}

std::string ComboToString(const std::vector<int>& combo) {
  std::string combo_str;
  for (int i = 0; i < combo.size(); ++i) {
    for (int k = 0; k < combo[i]; ++k) {
      combo_str.push_back(ColorToChar(i));
    }
  }
  return combo_str;
}

char ColorToChar(int color) { return static_cast<char>('A' + color); }

int CharToColor(char c) { return static_cast<int>(c - 'A'); }

void InitTradeInfo(TradeInfo* trade_info, int num_colors) {
  InitChipCombosRec(trade_info, num_colors, "");
  for (int i = 0; i < trade_info->chip_combinations.size(); ++i) {
    for (int j = 0; j < trade_info->chip_combinations.size(); ++j) {
      Trade candidate(trade_info->chip_combinations[i],
                      trade_info->chip_combinations[j]);
      bool valid = candidate.reduce();
      if (!valid) {
        continue;
      }

      std::string candidate_str = candidate.ToString();

      if (trade_info->trade_str_to_id.find(candidate_str) ==
          trade_info->trade_str_to_id.end()) {
        // std::cout << "Valid trade: " << candidate_str << std::endl;
        trade_info->possible_trades.push_back(
            std::make_unique<Trade>(candidate));
        trade_info->trade_str_to_id[candidate_str] =
            trade_info->possible_trades.size() - 1;
      }
    }
  }
}

std::pair<int, bool> Score(Player player, const Board& board) {
  bool solved = false;
  int score = ScoreRec(player, board, &solved);
  return std::make_pair(score, solved);
}

void ParseBoardsString(std::vector<Board>* boards,
                       const std::string& boards_string,
                       int num_colors, int board_size, int num_players) {
  std::vector<std::string> lines = absl::StrSplit(boards_string, '\n');
  SPIEL_CHECK_GT(lines.size(), 1);
  for (const std::string& line : lines) {
    if (!line.empty()) {
      Board board(board_size, num_colors, num_players);
      board.ParseFromLine(line);
      boards->push_back(board);
    }
  }
}

void ParseBoardsFile(std::vector<Board>* boards, const std::string& filename,
                     int num_colors, int board_size, int num_players) {
  open_spiel::file::File infile(filename, "r");
  std::string contents = infile.ReadContents();
  ParseBoardsString(boards, contents, num_colors, board_size, num_players);
}

}  // namespace colored_trails
}  // namespace open_spiel
