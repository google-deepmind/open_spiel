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

#include "open_spiel/games/crossword/duel_board.h"

#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace crossword_duel {

void CrosswordDuelBoard::SetSize(int num_rows, int num_cols) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
}

void CrosswordDuelBoard::AddCell(char c) { cells_.push_back(c); }

void CrosswordDuelBoard::AddTileInfo(char letter, int count, int value) {
  tile_distribution_.push_back(TileInfo{letter, count, value});
}

bool CrosswordDuelBoard::InBounds(int row, int col) const {
  return row >= 0 && row < num_rows_ && col >= 0 && col < num_cols_;
}

char CrosswordDuelBoard::RawCellAt(int row, int col) const {
  return cells_[row * num_cols_ + col];
}

BonusType CrosswordDuelBoard::BonusAt(int row, int col) const {
  if (!InBounds(row, col)) {
    return BonusType::kNone;
  }
  switch (RawCellAt(row, col)) {
    case 'd':
      return BonusType::kDoubleLetter;
    case 'D':
      return BonusType::kDoubleWord;
    case 't':
      return BonusType::kTripleLetter;
    case 'T':
      return BonusType::kTripleWord;
    default:
      return BonusType::kNone;
  }
}

bool CrosswordDuelBoard::IsStartCell(int row, int col) const {
  return InBounds(row, col) && RawCellAt(row, col) == 's';
}

int CrosswordDuelBoard::LetterValue(char letter) const {
  char upper = absl::ascii_toupper(letter);
  for (const TileInfo& info : tile_distribution_) {
    if (info.letter == upper) {
      return info.value;
    }
  }
  return 0;
}

std::vector<char> CrosswordDuelBoard::BuildTileBag() const {
  std::vector<char> bag;
  for (const TileInfo& info : tile_distribution_) {
    for (int i = 0; i < info.count; ++i) {
      bag.push_back(info.letter);
    }
  }
  return bag;
}

StatusWithValue<CrosswordDuelBoard> ParseDuelBoardFromFile(
    const std::string& filename) {
  std::string contents = file::ReadContentsFromFile(filename, "r");
  return ParseDuelBoardFromString(contents);
}

StatusWithValue<CrosswordDuelBoard> ParseDuelBoardFromString(
    const std::string& contents) {
  CrosswordDuelBoard board;
  std::vector<std::string> raw_lines = absl::StrSplit(contents, '\n');
  std::vector<std::string> lines;
  lines.reserve(raw_lines.size());
  for (const std::string& line : raw_lines) {
    lines.push_back(std::string(absl::StripAsciiWhitespace(line)));
  }

  // 0: board layout, 1: tile distribution.
  int section = -1;
  bool prev_line_empty = true;
  int num_rows = 0;
  int num_cols = 0;

  for (const std::string& line : lines) {
    if (!line.empty() && prev_line_empty) {
      prev_line_empty = false;
      section++;
    }
    if (line.empty()) {
      prev_line_empty = true;
      continue;
    }

    if (section == 0) {
      if (num_cols == 0) {
        num_cols = line.size();
      } else if (num_cols != line.size()) {
        return ErrorStatusWithValue<CrosswordDuelBoard>(
            absl::StrCat("Wrong number of columns in board row ", num_rows),
            board);
      }
      num_rows++;
      for (char c : line) {
        board.AddCell(c);
      }
    } else if (section == 1) {
      std::vector<std::string> parts =
          absl::StrSplit(line, ' ', absl::SkipEmpty());
      int count = 0;
      int value = 0;
      if (parts.size() != 3 || parts[0].size() != 1 ||
          !absl::SimpleAtoi(parts[1], &count) ||
          !absl::SimpleAtoi(parts[2], &value)) {
        return ErrorStatusWithValue<CrosswordDuelBoard>(
            absl::StrCat("Malformed tile distribution line: ", line), board);
      }
      board.AddTileInfo(absl::ascii_toupper(parts[0][0]), count, value);
    }
  }

  if (num_rows == 0 || num_cols == 0) {
    return ErrorStatusWithValue<CrosswordDuelBoard>(
        "Empty board layout.", board);
  }
  board.SetSize(num_rows, num_cols);
  if (board.tile_distribution().empty()) {
    return ErrorStatusWithValue<CrosswordDuelBoard>(
        "No tile distribution found.", board);
  }

  return OkStatusWithValue<CrosswordDuelBoard>(board);
}

std::string CrosswordDuelBoard::ToString() const {
  std::string result;
  absl::StrAppend(&result, "CrosswordDuelBoard: ", num_rows_, "x", num_cols_,
                  "\n");
  for (int r = 0; r < num_rows_; ++r) {
    for (int c = 0; c < num_cols_; ++c) {
      result.push_back(RawCellAt(r, c));
    }
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "\nTile distribution:\n");
  for (const TileInfo& info : tile_distribution_) {
    absl::StrAppend(&result, std::string(1, info.letter), " ", info.count,
                    " ", info.value, "\n");
  }
  return result;
}

}  // namespace crossword_duel
}  // namespace open_spiel
