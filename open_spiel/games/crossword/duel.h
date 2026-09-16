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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_DUEL_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_DUEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/crossword/duel_board.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/status.h"

// CrosswordDuel: a 2-player clone of Scrabble, played as a standalone
// ActionStruct-only game (see the documentation for these types of games in
// spiel.h, under the `action_structs_only` field of GameType, and the
// Crossword game for another example -- this game is implemented separately
// and does not extend/reuse Crossword's classes).
//
// Players alternate placing words on a shared board (see duel_default_board.h
// for the board layout and tile bag/value distribution), scoring points based
// on letter values and the board's bonus squares (double/triple letter/word).
// Each player privately holds a rack of up to 7 tiles, drawn from a shared,
// hidden bag; the game is therefore one of imperfect information.
//
// Actions refer to a placement by board coordinates: a starting (row, col),
// a direction ("across" or "down"), and the full word being formed (which may
// re-use letters already on the board). A player may also pass their turn.
// As with Crossword, this game uses action structs only: flat integer
// actions are only used for the chance nodes that deal/draw tiles.
//
// Parameters:
//   word_list_file    string   The file containing the legal word list, one
//                              word per line. If empty, a small built-in
//                              default word list is used. Default: "".

namespace open_spiel {
namespace crossword_duel {

inline constexpr char kEmptyCell = ' ';
inline constexpr int kNumPlayers = 2;
inline constexpr int kRackSize = 7;
inline constexpr int kMinWordLength = 2;
inline constexpr int kBingoBonus = 50;
inline constexpr int kMaxGameLength = 400;
inline constexpr const char* kDefaultWordListFile = "";

class CrosswordDuelGame;

struct CrosswordDuelActionStruct : public ActionStruct {
  int row = -1;
  int col = -1;
  std::string direction;  // "across" or "down". Ignored when pass = true.
  std::string word;       // Ignored when pass = true.
  bool pass = false;

  CrosswordDuelActionStruct(int _row, int _col, const std::string& _direction,
                            const std::string& _word)
      : row(_row), col(_col), direction(_direction), word(_word) {}

  std::string ToString() const { return ToJson(); }
  SPIEL_STRUCT_BOILERPLATE(CrosswordDuelActionStruct, row, col, direction,
                           word, pass);
};

class CrosswordDuelActionStructSampler : public ActionStructSampler {
 public:
  explicit CrosswordDuelActionStructSampler(const State* state, int rng_seed)
      : ActionStructSampler(state, rng_seed) {}
  std::unique_ptr<ActionStruct> SampleActionStruct() override;
};

class CrosswordDuelState : public State {
 public:
  explicit CrosswordDuelState(std::shared_ptr<const Game> game);
  CrosswordDuelState(const CrosswordDuelState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  // Explicit-stochastic chance outcomes: drawing one tile from the bag.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  Status ValidateActionStruct(
      const ActionStruct& action_struct) const override;

  Status ApplyActionStruct(const ActionStruct& action_struct) override;

  std::unique_ptr<ActionStructSampler> GetActionStructSampler(
      int seed) const override {
    return std::make_unique<CrosswordDuelActionStructSampler>(this, seed);
  }

  std::string Serialize() const override;

  char CharAt(int row, int col) const {
    return board_state_[row * board_.num_cols() + col];
  }

  bool IsBoardEmpty() const;
  int RackValue(Player player) const;

  const CrosswordDuelBoard& board() const { return board_; }
  const std::vector<char>& rack(Player player) const {
    return racks_[player];
  }
  const std::vector<int>& scores() const { return scores_; }
  int bag_size() const { return bag_.size(); }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  std::string ComputeCrossWord(int row, int col, Direction main_direction,
                               char new_letter,
                               int* new_letter_index = nullptr) const;
  int ComputeScoreAndApply(const CrosswordDuelActionStruct& action,
                           Player mover);
  void FinishDraws();
  void AdvanceOrEnd(Player mover);

  const CrosswordDuelGame& parent_game_;
  CrosswordDuelBoard board_;
  std::vector<char> board_state_;
  std::vector<std::vector<char>> racks_;
  std::vector<char> bag_;
  std::vector<int> scores_;
  std::vector<int> reward_;
  std::vector<std::string> serialized_history_;

  Player current_player_ = kChancePlayerId;
  Player drawing_player_ = 0;
  int tiles_needed_ = 0;
  bool dealing_initial_ = true;
  int consecutive_passes_ = 0;
  bool game_over_ = false;
  int num_actions_ = 0;
};

class CrosswordDuelGame : public Game {
 public:
  explicit CrosswordDuelGame(const GameParameters& params);

  // Constructs a game whose board is the given layout string (in the same
  // format as duel_default_board.h), rather than the default board.
  CrosswordDuelGame(const GameParameters& params,
                    const std::string& board_string);

  int NumDistinctActions() const override { return 0; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -300.0; }
  double MaxUtility() const override { return 300.0; }
  int MaxGameLength() const override { return kMaxGameLength; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CrosswordDuelState(shared_from_this()));
  }

  std::pair<std::string, std::string> ActionStructSpec() const override {
    return std::make_pair(
        "{\"row\": int, \"col\": int, \"direction\": string, "
        "\"word\": string, \"pass\": bool}",
        "{\"row\": 7, \"col\": 7, \"direction\": \"across\", "
        "\"word\": \"CAT\", \"pass\": false}");
  }

  const CrosswordDuelBoard& board() const { return board_; }
  const absl::flat_hash_set<std::string>& word_set() const {
    return word_set_;
  }
  int num_words() const { return word_set_.size(); }

  std::unique_ptr<State> DeserializeState(const std::string& str) const
      override;

 private:
  Status ParseWordList();

  std::string word_list_file_;
  absl::flat_hash_set<std::string> word_set_;
  CrosswordDuelBoard board_;
};

// Creates a crossword_duel game whose board is the given layout string (in
// the same format as duel_default_board.h), with the given legal word list
// file (or the built-in default word list, if empty).
std::shared_ptr<const Game> CreateCrosswordDuelGame(
    const std::string& board, const std::string& word_list_file);

}  // namespace crossword_duel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CROSSWORD_DUEL_H_
