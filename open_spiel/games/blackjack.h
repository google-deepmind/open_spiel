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

#ifndef OPEN_SPIEL_GAMES_BLACKJACK_H_
#define OPEN_SPIEL_GAMES_BLACKJACK_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple game that includes chance and imperfect information
// http://en.wikipedia.org/wiki/Blackjack
// Currently, it supports only a single player against the dealer.

namespace open_spiel {
namespace blackjack {

constexpr int kNumSuits = 4;
constexpr int kCardsPerSuit = 13;
constexpr int kDeckSize = kCardsPerSuit * kNumSuits;

class BlackjackGame;

class BlackjackState : public State {
 public:
  BlackjackState(const BlackjackState&) = default;
  BlackjackState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  ActionsAndProbs ChanceOutcomes() const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<Action> LegalActions() const override;

  int GetBestPlayerTotal(int player) const;
  int DealerId() const;
  int NextTurnPlayer() const;
  bool InitialCardsDealt(int player) const;
  int CardValue(int card) const;
  void EndPlayerTurn(int player);
  void DealCardToPlayer(int player, int card);

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void MaybeApplyDealerAction();

  // Initialize to bad/invalid values. Use open_spiel::NewInitialState()

  int total_moves_ = -1;    // Total num moves taken during the game.
  Player cur_player_ = -1;  // Player to play.
  int turn_player_ = -1;    // Whose actual turn is it. At chance nodes, we need
                            // to remember whose is playing for next turns.
  int live_players_ = 0;    // Number of players who haven't yet bust.
  std::vector<int>
      non_ace_total_;  // Total value of cards for each player, excluding aces.
  std::vector<int> num_aces_;            // Number of aces owned by each player.
  std::vector<int> turn_over_;           // Whether each player's turn is over.
  std::vector<int> deck_;                // Remaining cards in the deck.
  std::vector<std::vector<int>> cards_;  // Cards dealt to each player.
};

class BlackjackGame : public Game {
 public:
  explicit BlackjackGame(const GameParameters& params);

  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BlackjackState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return kDeckSize; }
  int MaxGameLength() const { return 12; }

  int NumPlayers() const override { return 1; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return +1; }
};

}  // namespace blackjack
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BLACKJACK_H_
