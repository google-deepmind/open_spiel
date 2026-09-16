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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_DUEL_BOARD_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_DUEL_BOARD_H_

#include <string>
#include <vector>

#include "open_spiel/utils/status.h"

// Static (parameter-independent) board layout and tile economy for the
// CrosswordDuel game: a 2-player word-placement game played like Scrabble.
// See duel_default_board.h for the file format this parses.

namespace open_spiel {
namespace crossword_duel {

enum class Direction {
  kAcross,
  kDown,
};

enum class BonusType {
  kNone,
  kDoubleLetter,
  kTripleLetter,
  kDoubleWord,
  kTripleWord,
};

// One line of the tile distribution: `count` copies of `letter`, each worth
// `value` points.
struct TileInfo {
  char letter;
  int count;
  int value;
};

class CrosswordDuelBoard {
 public:
  CrosswordDuelBoard() = default;
  CrosswordDuelBoard(const CrosswordDuelBoard& other) = default;

  bool InBounds(int row, int col) const;
  BonusType BonusAt(int row, int col) const;
  bool IsStartCell(int row, int col) const;

  // Returns the point value of `letter` according to the tile distribution,
  // or 0 if the letter does not appear in it.
  int LetterValue(char letter) const;

  // Expands the tile distribution into one entry per physical tile, e.g.
  // 9 copies of 'A', 2 copies of 'B', etc. This is the full bag at the start
  // of a game.
  std::vector<char> BuildTileBag() const;

  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }
  const std::vector<TileInfo>& tile_distribution() const {
    return tile_distribution_;
  }

  // Methods for building a board while parsing.
  void SetSize(int num_rows, int num_cols);
  void AddCell(char c);
  void AddTileInfo(char letter, int count, int value);

  std::string ToString() const;

 private:
  char RawCellAt(int row, int col) const;

  int num_rows_ = -1;
  int num_cols_ = -1;

  // Raw layout characters: '.' (normal), 's' (start), 'd'/'D' (double
  // letter/word), 't'/'T' (triple letter/word).
  std::vector<char> cells_;

  std::vector<TileInfo> tile_distribution_;
};

StatusWithValue<CrosswordDuelBoard> ParseDuelBoardFromFile(
    const std::string& filename);

StatusWithValue<CrosswordDuelBoard> ParseDuelBoardFromString(
    const std::string& contents);

}  // namespace crossword_duel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CROSSWORD_DUEL_BOARD_H_
