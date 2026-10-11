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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_BOARD_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_BOARD_H_

#include <string>
#include <utility>
#include <vector>
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/utils/status.h"

namespace open_spiel {
namespace crossword {

inline constexpr char kBlockedCell = '#';

using Number = int;

enum class Direction {
  kAcross,
  kDown,
};

struct Clue {
  Number number;
  Direction direction;
  std::string description;
};

std::string ClueId(const Clue& clue);

class CrosswordBoard {
 public:
  CrosswordBoard() = default;
  CrosswordBoard(const CrosswordBoard& other) = default;

  std::string ToString() const;

  bool InBounds(int row, int col) const;
  char CharacterAt(int row, int col) const;
  std::vector<std::string> Answers() const;
  bool MatchesCorrectLetters(int row, int col, Direction direction,
                             const std::string& word) const;

  // Methods for building a crossword board.
  void BuildCrosswordSetSize(int num_rows, int num_cols);
  void BuildCrosswordAddCharacter(char c);
  void BuildCrosswordAddClue(Clue clue, const std::string& answer);
  void BuildCrosswordAddClueLocation(const std::string& cid, int row,
                                     int col);

  const Clue& clue(int index) const { return clues_[index]; }
  const Clue& clue(const std::string& cid) const {
    return cid_to_clue_.at(cid);
  }
  const std::string& answer(const std::string& cid) const {
    return cid_to_answer_.at(cid);
  }
  int clue_index(const Clue& clue) const {
    for (int i = 0; i < clues_.size(); ++i) {
      if (clues_[i].number == clue.number &&
          clues_[i].direction == clue.direction) {
        return i;
      }
    }
    return -1;
  }

  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }
  int num_clues() const { return clues_.size(); }

  const absl::flat_hash_map<std::string, std::pair<int, int>>&
      cid_to_cells_map() const {
    return cid_to_cells_;
  }

 private:
  int num_rows_ = -1;
  int num_cols_ = -1;

  // The id of the crossword, something to help identify it.
  std::string id_;

  std::vector<char> board_;

  // A clue id is a string of the form "A1" or "D3" etc.
  std::vector<Clue> clues_;
  absl::flat_hash_map<std::string, Clue> cid_to_clue_;
  absl::flat_hash_map<std::string, std::string> cid_to_answer_;
  absl::flat_hash_map<std::string, std::pair<int, int>> cid_to_cells_;
};

StatusWithValue<CrosswordBoard>
ParseCrosswordFromFile(const std::string& filename);

StatusWithValue<CrosswordBoard>
ParseCrosswordFromString(const std::string& contents);

}  // namespace crossword
}  // namespace open_spiel


#endif  // OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_BOARD_H_
