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

#ifndef OPEN_SPIEL_GAMES_COLORED_TRAILS_H_
#define OPEN_SPIEL_GAMES_COLORED_TRAILS_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/spiel.h"

// A simple bargaining game [1].
//
// This code currently implements the three-player imperfect information game
// from these papers [2,3] but we would like this to be a generic implementation
// that can handle the several variants of the classic game.
//
// [1] Ya'akov Gal, Barbara Grosz, Sarit Kraus, Avi Pfeffer, and Stuart Shieber.
//     2010. Agent decision-making in open mixed networks. Artificial
//     Intelligence 174(18): 1460-1480.
// [2] de Jong et al. '11, Metastrategies in the Colored Trails Game
//     https://www.ifaamas.org/Proceedings/aamas2011/papers/C4_R57.pdf
// [3] S. G. Ficici and A. Pfeffer. Modeling how Humans Reason about Others with
//     Partial Information. In Proceedings of the Seventh International
//     Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2008.
//
// Parameters:
//     "boards_file"        string  The file containing the boards (default: "")
//     "board_size"         int     number of rows / columns (default = 4)
//     "num_colors"         int     number of colors (default = 5)
//     "players"            int     number of players (default = 3)

namespace open_spiel {
namespace colored_trails {

constexpr int kResponderId = 2;

constexpr int kDefaultNumPlayers = 3;
constexpr int kDefaultNumColors = 5;
constexpr int kDefaultBoardSize = 4;  // 4x4

// [3] states that each player receive between 4 and 8 chips, but [2] shows
// instances with only 3 chips.
constexpr int kNumChipsLowerBound = 3;
constexpr int kNumChipsUpperBound = 8;

constexpr int kLeftoverChipScore = 10;
constexpr int kFlagPenaltyPerCell = -25;

// Default 10-board database used for tests, etc. See
// colored_trails/boards100.txt and create your own using
// colored_trails/colored_trails_board_generator to make your own.
constexpr const char* kDefaultBoardsString =
    "4 5 3 DEADCACCADBDBECC BCD BDDDD AAABCC 4 5 15 12\n"
    "4 5 3 CCADBEEAEDDDDACD ACCD AABC ABBCDDE 14 7 8 11\n"
    "4 5 3 ECBBDECECEECBDCE ABBEEE BCDE ACCCEE 3 10 13 0\n"
    "4 5 3 EBBEABDCAAAEDABD AAABE AAB BBDDDE 6 14 7 12\n"
    "4 5 3 BEBBAADEBBCABABD AACDE ACCDE BBBDDDE 5 1 15 9\n"
    "4 5 3 BACBBEAADBDCECAE ABCCCDD BCDDEE ACCCEEE 0 7 5 13\n"
    "4 5 3 EBCCDDBAEADEEDDE CCD ABDD ACEE 5 7 0 8\n"
    "4 5 3 BCDACCACBDCBDDDB BBCCCE AAABCCEE AAADD 1 12 8 10\n"
    "4 5 3 EEEAEBDBEDCEDBCE ABCCDE DDD BEEE 8 7 10 2\n"
    "4 5 3 EBBEEBEECBECDADB BBCCDDDD AACCDD BEEE 5 14 15 11\n";

class ColoredTrailsGame;  // Forward definition necessary for parent pointer.

struct Trade {
  std::vector<int> giving;
  std::vector<int> receiving;
  Trade() {}
  Trade(const std::vector<int> _giving, const std::vector<int> _receiving);
  Trade(const Trade& other);
  std::string ToString() const;
  bool operator==(const Trade& other) const {
    return (giving == other.giving && receiving == other.receiving);
  }
  bool reduce();  // remove redundant chip exchanges from both sides
                  // returns whether it's a valid trade (nonempty giving
                  // and receiving)
};

struct TradeInfo {
  std::vector<std::vector<int>> chip_combinations;
  std::vector<std::unique_ptr<Trade>> possible_trades;
  absl::flat_hash_map<std::string, int> trade_str_to_id;
};

struct Board {
  int size = kDefaultBoardSize;
  int num_colors = kDefaultNumColors;
  int num_players = kDefaultNumPlayers;
  std::vector<int> board;
  std::vector<int> num_chips;
  std::vector<std::vector<int>> chips;
  std::vector<int> positions;  // Flag position is at positions[num_players]

  Board();
  Board(int _size, int _num_colors, int _num_players);

  void ParseFromLine(const std::string& line);
  bool InBounds(int row, int col) const;
  void init();
  std::string ToString() const;
  std::string PrettyBoardString() const;
  void ApplyTrade(std::pair<int, int> players, const Trade& trade);
};

class ChipComboIterator {
 public:
  ChipComboIterator(const std::vector<int>& chips);
  bool IsFinished() const;
  std::vector<int> Next();

 private:
  std::vector<int> chips_;
  std::vector<int> cur_combo_;
};

class ColoredTrailsState : public State {
 public:
  ColoredTrailsState(std::shared_ptr<const Game> game, int board_size,
                     int num_colors);
  ColoredTrailsState(const ColoredTrailsState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string InformationStateString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;

  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool IsLegalTrade(Player proposer, const Trade& trade) const;

  Player cur_player_;
  const ColoredTrailsGame* parent_game_;
  Board board_;
  std::vector<double> returns_;
  std::vector<Trade> proposals_;
};

class ColoredTrailsGame : public Game {
 public:
  explicit ColoredTrailsGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new ColoredTrailsState(shared_from_this(), board_size_, num_colors_));
  }
  int MaxChanceOutcomes() const override { return all_boards_.size(); }

  int MaxGameLength() const override { return 3; }
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return num_players_; }
  double MaxUtility() const override {
    return kLeftoverChipScore * kNumChipsUpperBound;
  }
  double MinUtility() const override {
    return board_size_ * board_size_ * kFlagPenaltyPerCell;
  }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;

  const std::vector<Board>& AllBoards() const { return all_boards_; }

  Trade LookupTrade(int trade_id) const {
    return *(trade_info_.possible_trades.at(trade_id));
  }

  int LookupTradeId(const std::string& trade_str) const {
    return trade_info_.trade_str_to_id.at(trade_str);
  }

 private:
  const int num_colors_;
  const int board_size_;
  const int num_players_;
  std::vector<Board> all_boards_;
  TradeInfo trade_info_;
};

// Helper functions used by the board generator and game implementation.
// Implementations contained in colored_trails_utils.cc.
char ColorToChar(int color);
int CharToColor(char c);
std::string ComboToString(const std::vector<int>& combo);
std::vector<int> ComboStringToCombo(const std::string& combo_str,
                                    int num_colors);
void InitTradeInfo(TradeInfo* trade_info, int num_colors);

// This is the G function described in [2]: the score if the player were to
// advance as close to the goal as possible given their current chips.
std::pair<int, bool> Score(Player player, const Board& board);

void ParseBoardsFile(std::vector<Board>* boards, const std::string& filename,
                     int num_colors, int board_size, int num_players);
void ParseBoardsString(std::vector<Board>* boards,
                       const std::string& boards_string,
                       int num_colors, int board_size, int num_players);

}  // namespace colored_trails
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COLORED_TRAILS_H_
