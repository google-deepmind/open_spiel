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

#ifndef OPEN_SPIEL_GAMES_CRIBBAGE_H_
#define OPEN_SPIEL_GAMES_CRIBBAGE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of Cribbage:
// https://en.wikipedia.org/wiki/Cribbage

namespace open_spiel {
namespace cribbage {

constexpr int kNumSuits = 4;
constexpr int kCardsPerSuit = 13;
constexpr int kDeckSize = kCardsPerSuit * kNumSuits;

// First 52 represents single-card actions.
// Next 52*52 represents two-card actions.
// 1 for the pass action.
constexpr int kNumDistinctActions = 2757;
constexpr int kPassAction = 2756;

enum Suit {
  kClubs = 0,
	kDiamonds = 1,
	kHearts = 2,
	kSpades = 3
};

enum Rank {
  kAce = 0,
	kTwo = 1,
	kThree = 2,
	kFour = 3,
	kFive = 4,
	kSix = 5,
	kSeven = 6,
	kEight = 7,
	kNine = 8,
	kTen = 9,
	kJack = 10,
	kQueen = 11,
	kKing = 12,
};

const char kSuitNames[kNumSuits + 1] = "CDHS";
const char kRanks[kCardsPerSuit + 1] = "A23456789TJQK";

struct Card {
  int id;
	int rank;
	int suit;
	int value() const;
  std::string to_string() const;
};

bool operator==(const Card& lhs, const Card& rhs);
bool operator<(const Card& lhs, const Card& rhs);

enum Phase {
  kCardPhase = 0,
  kPlayPhase = 1
};

class CribbageGame;

class CribbageState : public State {
 public:
  CribbageState(const CribbageState&) = default;
  CribbageState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  ActionsAndProbs ChanceOutcomes() const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<Action> LegalActions() const override;

	int round() const { return round_; }

 protected:
  void DoApplyAction(Action move_id) override;

 private:
	std::vector<Action> LegalOneCardCribActions() const;
	std::vector<Action> LegalTwoCardCribActions() const;
	void SortHands();
	void SortCrib();
	void MoveCardToCrib(Player player, const Card& card);
	void Score(Player player, int points);
	bool AllHandsAreEmpty() const;
	bool AllPlayersHavePassed() const;

  const CribbageGame& parent_game_;
  int round_ = -1;
	int dealer_ = -1;          // Who is the dealer?
  int start_player_ = -1;    // Who is starting this round.
	Phase phase_;						   // Choosing cards or play phase?
  Player cur_player_ = -1;   // Player to play.
	std::vector<int> rewards_; // Intermediate rewards
	std::vector<int> scores_;  // Current points for each player.

	std::optional<Card> starter_;
	std::vector<Card> deck_;
	std::vector<std::vector<Card>> hands_;
	std::vector<std::vector<Card>> discards_;
	std::vector<Card> crib_;
	std::vector<Card> played_cards_;
	std::vector<bool> passed_;
	Player last_played_player_;   // Last player to have played a card.
	int current_sum_ = -1;

	void NextRound();
};

class CribbageGame : public Game {
 public:
  explicit CribbageGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CribbageState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return kDeckSize; }
  int MaxGameLength() const { return 0; }

  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return +1; }
  std::vector<int> ObservationTensorShape() const override { return {}; }

	int cards_per_player() const { return cards_per_player_; }
	int cards_to_crib() const { return cards_to_crib_; }

 private:
  const int num_players_;
	const int cards_per_player_;
	const int cards_to_crib_;
};

Card GetCard(int id);

}  // namespace cribbage 
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CRIBBAGE_H_
