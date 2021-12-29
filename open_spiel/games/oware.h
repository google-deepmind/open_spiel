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

#ifndef OPEN_SPIEL_GAMES_OWARE_H_
#define OPEN_SPIEL_GAMES_OWARE_H_

#include <memory>
#include <unordered_set>

#include "open_spiel/games/oware/oware_board.h"
#include "open_spiel/spiel.h"

// Oware (https://en.wikipedia.org/wiki/Oware) is a strategy game within the
// family of Mancala games. Several variations of the game exist. This
// implementation uses the basic rules as described here:
// https://en.wikipedia.org/wiki/Oware or here:
// http://www.joansala.com/auale/rules/en/
//
// In particular if the opponent has no seeds, the current player must make a
// move to give the opponent seeds. If no such move exists the game ends and the
// current player collects the seeds in his row. If at the end of an action the
// opponent would be left with no seeds because they would all be captured
// (a Grand Slam), no seeds are captured instead.
//
// When the game reaches a state which occurred before, it ends and both players
// collect the remaining seeds in their respective rows.

namespace open_spiel {
namespace oware {

inline constexpr int kMinCapture = 2;
inline constexpr int kMaxCapture = 3;

inline constexpr int kDefaultHousesPerPlayer = 6;
inline constexpr int kDdefaultSeedsPerHouse = 4;

// Informed guess based on
// https://mancala.fandom.com/wiki/Statistics
inline constexpr int kMaxGameLength = 1000;

class OwareState : public State {
 public:
  OwareState(std::shared_ptr<const Game> game, int num_houses_per_player,
             int num_seeds_per_house);

  OwareState(const OwareState&) = default;

  // Custom board setup to support testing.
  explicit OwareState(std::shared_ptr<const Game> game,
                      const OwareBoard& board);

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : board_.current_player;
  }

  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  const OwareBoard& Board() const { return board_; }
  std::string ObservationString(Player player) const override;

  // The game board is provided as a vector, encoding the players' seeds
  // and their score, as a fraction of the number of total number of seeds in
  // the game. This provides an interface that can be used for neural network
  // training, although the given representation is not necessary the best
  // for that purpose.
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  void WritePlayerScore(std::ostringstream& out, Player player) const;

  // Collects the seeds from the given house and distributes them
  // counterclockwise, skipping the starting position in all cases.
  // Returns the index of the last house in which a seed was dropped.
  int DistributeSeeds(int house);

  int OpponentSeeds() const;

  bool InOpponentRow(int house) const;

  // If the opponent would be left with no seeds after capturing starts from
  // the given house, it is a Grand Slam. Such a move is allowed but no pieces
  // will be captured.
  bool IsGrandSlam(int house) const;

  // Collects all seeds of both players and terminates the game.
  void CollectAndTerminate();

  // Captures opponent seeds starting from given house clockwise as long as
  // the number of seeds is between kMinCapture and kMaxCapture.
  // Returns the number of seeds captured.
  int DoCaptureFrom(int house);

  int LowerHouse(int house) const {
    return (house / num_houses_per_player_) * num_houses_per_player_;
  }

  int UpperHouse(int house) const {
    return LowerHouse(house) + num_houses_per_player_ - 1;
  }

  int PlayerLowerHouse(Player player) const {
    return player * num_houses_per_player_;
  }

  int PlayerUpperHouse(Player player) const {
    return player * num_houses_per_player_ + num_houses_per_player_ - 1;
  }

  bool ShouldCapture(int seeds) const {
    return seeds >= kMinCapture && seeds <= kMaxCapture;
  }

  Action HouseToAction(int house) const {
    return house % num_houses_per_player_;
  }

  int ActionToHouse(Player player, Action action) const {
    return player * num_houses_per_player_ + action;
  }

  int NumHouses() const { return kNumPlayers * num_houses_per_player_; }

  class OwareBoardHash {
   public:
    std::size_t operator()(const OwareBoard& board) const {
      return board.HashValue();
    }
  };

  const int num_houses_per_player_;
  const int total_seeds_;

  // We keep the set of visited board states to detect repetition, at which
  // point the game ends and both players collect the seeds on their own row.
  // Because captured seeds never enter the game again, this set is reset
  // on any capture.
  std::unordered_set<OwareBoard, OwareBoardHash> boards_since_last_capture_;
  OwareBoard board_;
};

// Game object.
class OwareGame : public Game {
 public:
  explicit OwareGame(const GameParameters& params);
  int NumDistinctActions() const override { return num_houses_per_player_; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new OwareState(
        shared_from_this(), num_houses_per_player_, num_seeds_per_house_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }

  int MaxGameLength() const override { return kMaxGameLength; }
  std::vector<int> ObservationTensorShape() const override;

 private:
  const int num_houses_per_player_;
  const int num_seeds_per_house_;
};

}  // namespace oware
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_OWARE_H_
