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

#include "open_spiel/games/crossword/crossword_board.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/status.h"

namespace open_spiel {
namespace crossword {

std::string ClueId(const Clue& clue) {
  std::string id;
  if (clue.direction == Direction::kAcross) {
    absl::StrAppend(&id, "A");
  } else {
    absl::StrAppend(&id, "D");
  }
  absl::StrAppend(&id, clue.number);
  return id;
}

StatusWithValue<std::pair<Clue, std::string>>
ParseClue(const std::string& line) {
  // Eg. "A1. Puts on, as paint ~ APPLIES"
  std::pair<Clue, std::string> clue_and_answer;

  if (line.empty()) {
    return ErrorStatusWithValue<std::pair<Clue, std::string>>(
        "Empty line parsing clue", clue_and_answer);
  }

  Direction direction;
  if (line.at(0) == 'A') {
    direction = Direction::kAcross;
  } else if (line.at(0) == 'D') {
    direction = Direction::kDown;
  } else {
    return ErrorStatusWithValue<std::pair<Clue, std::string>>(
        absl::StrCat("Unknown clue direction (line should start with A or D): ",
                     line),
        clue_and_answer);
  }
  clue_and_answer.first.direction = direction;

  size_t dot_index = line.find('.');
  if (dot_index == std::string::npos) {
    return ErrorStatusWithValue<std::pair<Clue, std::string>>(
        absl::StrCat("Unknown clue format (expected number followed by dot): ",
                     line),
        clue_and_answer);
  }

  size_t tilde_index = line.find('~');
  if (tilde_index == std::string::npos) {
    return ErrorStatusWithValue<std::pair<Clue, std::string>>(
        absl::StrCat("Unknown clue format (missing tilde separator): ",
                     line),
        clue_and_answer);
  }

  std::string number_str = line.substr(1, dot_index - 1);
  if (!absl::SimpleAtoi(number_str, &clue_and_answer.first.number) ||
      clue_and_answer.first.number <= 0) {
    return ErrorStatusWithValue<std::pair<Clue, std::string>>(
        absl::StrCat("Unknown clue format (failed to parse number): ",
                     line),
        clue_and_answer);
  }

  std::string description = line.substr(dot_index + 1,
                                        tilde_index - dot_index - 1);
  clue_and_answer.first.description = absl::StripAsciiWhitespace(description);
  std::string answer = line.substr(tilde_index + 1);
  clue_and_answer.second = absl::StripAsciiWhitespace(answer);

  return OkStatusWithValue<std::pair<Clue, std::string>>(clue_and_answer);
}

void CrosswordBoard::BuildCrosswordSetSize(int num_rows, int num_cols) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
}

// Add a character to the board.
void CrosswordBoard::BuildCrosswordAddCharacter(char c) {
  board_.push_back(c);
}

void CrosswordBoard::BuildCrosswordAddClue(Clue clue,
                                           const std::string& answer) {
  std::string cid = ClueId(clue);
  clues_.push_back(clue);
  cid_to_clue_[cid] = clue;
  cid_to_answer_[cid] = answer;
}

void CrosswordBoard::BuildCrosswordAddClueLocation(const std::string& cid,
                                                   int row, int col) {
  cid_to_cells_[cid] = {row, col};
}

bool CrosswordBoard::InBounds(int row, int col) const {
  return row >= 0 && row < num_rows_ && col >= 0 && col < num_cols_;
}

char CrosswordBoard::CharacterAt(int row, int col) const {
  if (!InBounds(row, col)) {
    return kBlockedCell;
  }
  return board_[row * num_cols_ + col];
}

std::vector<std::string> CrosswordBoard::Answers() const {
  std::vector<std::string> answers;
  answers.reserve(cid_to_answer_.size());
  for (const auto& [cid, answer] : cid_to_answer_) {
    answers.push_back(answer);
  }
  return answers;
}

bool CrosswordBoard::MatchesCorrectLetters(int row, int col,
                                           Direction direction,
                                           const std::string& word) const {
  for (int i = 0; i < word.size(); ++i) {
    int r = row + (direction == Direction::kAcross ? 0 : i);
    int c = col + (direction == Direction::kAcross ? i : 0);
    if (!InBounds(r, c) || CharacterAt(r, c) == kBlockedCell) {
      return false;
    }
    char board_char = CharacterAt(r, c);
    if (absl::ascii_toupper(board_char) != absl::ascii_toupper(word[i])) {
      return false;
    }
  }
  return true;
}

StatusWithValue<CrosswordBoard>
ParseCrosswordFromFile(const std::string& filename) {
  std::string contents = file::ReadContentsFromFile(filename, "r");
  return ParseCrosswordFromString(contents);
}

StatusWithValue<CrosswordBoard>
ParseCrosswordFromString(const std::string& contents) {
  CrosswordBoard board;
  std::vector<std::string> lines = absl::StrSplit(contents, '\n');

  // First strip all whitespace from each line.
  const int num_lines = lines.size();
  for (int i = 0; i < num_lines; ++i) {
    lines[i] = absl::StripAsciiWhitespace(lines[i]);
  }

  // Parse the lines.
  // 0: preamble, 1: puzzle, 2: across clues, 3: down clues
  int section = -1;
  bool prev_line_empty = true;
  int num_rows = 0;
  int num_cols = 0;

  for (int i = 0; i < num_lines; ++i) {
    if (!lines[i].empty() && prev_line_empty) {
      prev_line_empty = false;
      section++;
    }

    if (lines[i].empty()) {
      prev_line_empty = true;
      continue;
    }

    if (section == 1) {
      // Section 1 is the puzzle board itself.
      if (num_cols == 0) {
        num_cols = lines[i].size();
      } else if (num_cols != lines[i].size()) {
         return ErrorStatusWithValue<CrosswordBoard>(
             absl::StrCat("Wrong number of columns in row ", num_rows),
             board);
      }
      num_rows++;
      for (char c : lines[i]) {
        board.BuildCrosswordAddCharacter(c);
      }
    } else if (section > 1) {
      // Parse clues.
      // Do not differentiate between across and down clue sections here
      // (section 2 and 3) to allow for more flexible parsing.
      auto status_with_value = ParseClue(lines[i]);
      if (!status_with_value.ok()) {
        return ErrorStatusWithValue<CrosswordBoard>(
            status_with_value.message(), board);
      }
      std::pair<Clue, std::string> clue_and_answer =
          status_with_value.value();
      board.BuildCrosswordAddClue(clue_and_answer.first,
                                  clue_and_answer.second);
    }
  }

  board.BuildCrosswordSetSize(num_rows, num_cols);

  // Now connect the clue ids to the board locations and verify the answers.
  int clue_index = 0;
  for (const Direction direction : {Direction::kAcross, Direction::kDown}) {
    int row = 0, col = 0;
    while (row < num_rows) {
      // Scan to the next valid starting cell.
      while (board.InBounds(row, col) &&
             board.CharacterAt(row, col) == kBlockedCell) {
        col++;
        if (col >= num_cols) {
          row++;
          col = 0;
        }
      }

      if (!board.InBounds(row, col)) {
        break;
      }

      // If the word works
      std::string cid = ClueId(board.clue(clue_index));
      std::string answer = board.answer(cid);
      if (board.MatchesCorrectLetters(row, col, direction, answer)) {
        board.BuildCrosswordAddClueLocation(cid, row, col);
        clue_index++;
        if (clue_index >= board.num_clues()) {
          break;
        }

        // Skip to the one before next possible starting cell.
        if (direction == Direction::kAcross) {
          col += (answer.size() - 1);
          if (col >= num_cols) {
            row++;
            col -= num_cols;
          }
        }
      }

      // Now, advance one character.
      col++;
      if (col >= num_cols) {
        row++;
        col = 0;
      }
    }
  }

  if (clue_index != board.num_clues()) {
    return ErrorStatusWithValue<CrosswordBoard>(
        absl::StrCat("Could not verify all clues fit on the board.",
                     "Expected ", board.num_clues(), ", found ",
                     clue_index),
        board);
  }

  // Check that cells for Across and Down with the same numbers match.
  for (clue_index = 0; clue_index < board.num_clues(); ++clue_index) {
    std::string cid = ClueId(board.clue(clue_index));
    auto [row, col] = board.cid_to_cells_map().at(cid);
    std::string other_cid = cid;
    other_cid[0] = (cid.at(0) == 'A' ? 'D' : 'A');
    auto iter = board.cid_to_cells_map().find(other_cid);
    if (iter == board.cid_to_cells_map().end()) {
      // No matching across/down clue. Nothing to check.
      continue;
    } else {
      // Found a matching across/down clue. Verify that cells.
      auto [other_row, other_col] = iter->second;
      if (row != other_row || col != other_col) {
        return ErrorStatusWithValue<CrosswordBoard>(
            absl::StrCat("Across and down clues with the same number do not "
                         "share the same starting cell: ",
                         cid, " at (", row, ", ", col, ") and ", other_cid,
                         " at (", other_row, ", ", other_col, ")"),
            board);
      }
    }
  }

  return OkStatusWithValue<CrosswordBoard>(board);
}

std::string CrosswordBoard::ToString() const {
  std::string result;
  absl::StrAppend(&result, "CrosswordBoard:\n");
  absl::StrAppend(&result, "num_rows: ", num_rows_, "\n");
  absl::StrAppend(&result, "num_cols: ", num_cols_, "\n");
  absl::StrAppend(&result, "num_clues: ", num_clues(), "\n\n");

  for (int r = 0; r < num_rows_; ++r) {
    for (int c = 0; c < num_cols_; ++c) {
      result.push_back(CharacterAt(r, c));
    }
    absl::StrAppend(&result, "\n");
  }

  absl::StrAppend(&result, "\nClues:\n");
  for (const auto& clue : clues_) {
    std::string cid = ClueId(clue);
    absl::StrAppend(&result, cid, ". ", clue.description, " ~ ",
                    cid_to_answer_.at(cid), "\n");
  }

  return result;
}

}  // namespace crossword
}  // namespace open_spiel
