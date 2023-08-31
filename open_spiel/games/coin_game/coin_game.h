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

#ifndef OPEN_SPIEL_GAMES_COIN_GAME_H_
#define OPEN_SPIEL_GAMES_COIN_GAME_H_

#include <memory>
#include <set>

#include "open_spiel/spiel.h"

// An implementation of the 'Coin Game'. Different descriptions of this game
// exist with slightly different rules. In particular:
// a) "Modeling Others using Oneself in Multi-Agent Reinforcement Learning"
//    (https://arxiv.org/abs/1802.09640)
// b) "Maintaining cooperation in complex social dilemmas using deep
//    reinforcement learning" (https://arxiv.org/abs/1707.01068)
// c) "Learning with Opponent-Learning Awareness"
//    (https://arxiv.org/abs/1709.04326)
// The current implementation follows the description given in a).
//
// Players live on a a grid, which also contains coins of different colors.
// Players can collect coins by moving around and walking into the coin's
// square. They can move in all directions or choose not to move at all.
// If a player would move outside of the grid or into the square of another
// player, they stay where they are.
// Each player has a preferred color. They are rewarded for collecting
// coins of their own or other players' preference, but punished for collecting
// coins that are no one's preference. Players initially only know their own
// coin preference. The initial positions of players and coins on the board is
// randomized, as well as the players' color preferences.
// Players move sequentially, in fixed order, starting with player 0.

namespace open_spiel {
namespace coin_game {

class CoinGame;
using Location = std::pair<int, int>;

// Different phases of the game, first setup, then play.
enum struct GamePhase {
  kAssignPreferences = 0,
  kDeployPlayers = 1,
  kDeployCoins = 2,
  kPlay = 3
};

// Part of CoinState related to the setup phase.
struct Setup {
  Setup(int num_rows, int num_columns, int num_coin_colors);
  std::set<int> available_coin_colors_;
  std::set<int> available_positions_;
  int num_players_assigned_preference = 0;
  int num_players_on_field = 0;
  int num_coins_on_field = 0;
};

class CoinState : public State {
 public:
  explicit CoinState(std::shared_ptr<const Game> game);
  CoinState(const CoinState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;
  std::string ObservationString(Player player) const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  GamePhase GetPhase() const;
  Location LocationFromIndex(int index) const;
  char GetField(Location loc) const;
  void SetField(Location loc, char symbol);
  bool InBounds(Location loc) const;
  int GetPlayerCoinCount(Player player, int coin_color) const;
  void IncPlayerCoinCount(Player player, int coin_color);

  void PrintCoinsCollected(std::ostream& out) const;
  void PrintPreferences(std::ostream& out) const;
  void PrintBoardDelimiterRow(std::ostream& out) const;
  void PrintBoard(std::ostream& out) const;

  void ApplyDeployPlayersAction(Action index);
  void ApplyDeployCoinsAction(Action index);
  void ApplyAssignPreferenceAction(Action coin_color);
  void ApplyPlayAction(Action move);

  const CoinGame& parent_game_;

  Setup setup_;
  Player cur_player_ =
      kChancePlayerId;  // Chance player for setting up the game.
  int total_moves_ = 0;
  std::vector<int> player_preferences_;
  std::vector<Location> player_location_;
  // num_rows x num_columns representation of playing field.
  std::vector<char> field_;
  // num_players x num_coin_colors representation of how many coins each player
  // collected.
  std::vector<int> player_coins_;
};

class CoinGame : public Game {
 public:
  explicit CoinGame(const GameParameters& params);

  int NumDistinctActions() const override;
  int MaxChanceOutcomes() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return num_players_; }
  double MaxUtility() const override;
  double MinUtility() const override;
  int MaxGameLength() const override;
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumRows() const { return num_rows_; }
  int NumColumns() const { return num_columns_; }
  int EpisodeLength() const { return episode_length_; }
  int NumCoinColors() const { return num_coin_colors_; }
  int NumCoinsPerColor() const { return num_coins_per_color_; }
  int TotalCoins() const { return num_coin_colors_ * num_coins_per_color_; }

 private:
  int num_players_;
  int num_rows_;
  int num_columns_;
  int episode_length_;
  int num_coin_colors_;
  int num_coins_per_color_;
};

}  // namespace coin_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COIN_GAME_H_
