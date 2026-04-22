// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_GO_FISH_H_
#define OPEN_SPIEL_GAMES_GO_FISH_H_

// Implementation of the card game go fish.
//
// https://en.wikipedia.org/wiki/Go_Fish
//
// There are multiple house vraiations. We support either ending the game 
// when a player goes out while making a book, or playing the game until all 
// cards are in books. The winner is either the player who goes out or 
// the one with the most books. We support any number of ranks or suits, 
// although "suits" is really just the number of copies of each card.
//
#include <vector> 

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace go_fish {
enum Phase {
	kDeal = 0,
	kAsk = 1,
	kFish = 2,
	kTerminal = 3
};

constexpr int kDefaultPlayers = 2;

std::string RankString(int rank);
std::string PhaseString(Phase phase);
int RankFromChar(char c);

struct Event {
	int player_id;
	int target;
  int rank;
	int received;
	int booked; // need a book rank for fishing
	bool caught;
	Event(int player_id, int target, int rank, int received, int booked)
    : player_id(player_id), target(target), rank(rank),
		  received(received), booked(booked), caught(false) {}
  Event(int player_id, int rank, int booked, bool caught)
	 :  player_id(player_id), target(-1), rank(rank),
		  received(-1), booked(booked), caught(caught) {}
	std::string ToString() const {
		std::string result;
		if (target == -1) {  // fishing
			absl::StrAppend(&result, player_id, " drew");
			if (caught) absl::StrAppend(&result, " caught ", rank);
			if (booked) absl::StrAppend(&result, " booked ", booked);
		} else {
			 absl::StrAppend(&result, player_id, " asked ", target, " for ", rank);
			 if (received > 0) absl::StrAppend(&result, " recieved ", received);
			 if (booked >= 0)  {
				 SPIEL_CHECK_EQ(booked, rank);
				 absl::StrAppend(&result, " booked ", booked);
			 }
		}
	  return result;
	}
};


class GoFishState: public State {
 public:
  explicit GoFishState(std::shared_ptr<const Game> game,
                       const std::string& state_str = "");

  GoFishState(const GoFishState&) = default;
  GoFishState& operator=(const GoFishState&) = default;
	Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == kTerminal; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
	int PoolSize() const;
	int IndexToRank(int index) const;
	// count of cards held by each player
	int PlayerCounts(int player_id) const;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
 protected:
  void DoApplyAction(Action move) override;
 private:
	Phase phase_;
  Player current_player_;  // Player zero goes first by default
	int initial_cards_;
	// secret info
  std::vector<std::vector<int>> player_cards_;
	std::vector<int> pool_;
	// public info:
	// books cashed by this player
	std::vector<int> player_books_;
  // how many times has this player ask for this card
	std::vector<std::vector<int>> player_did_ask_;
	// was player asked for this rank?
	std::vector<std::vector<bool>> player_was_asked_;
	// cards drawn by this player after being asked (zero if never asked)
  std::vector<std::vector<int>> drawn_since_was_asked_;
	// minimum a player can have of this rank
  std::vector<std::vector<int>> player_min_;
	// has this rank been booked already?
	std::vector<int> booked_;
	bool CheckBook(int player_id, int rank);
	void CheckPhase();
	void AdvancePlayer();
	std::vector<Action> GenerateAsks(int player_id) const;
	std::vector<Action> GenerateDraws() const; // draw from deck
	std::vector<Event> events_;
  
	// retrieve the card (rank) at index from pool_ 
	int Draw(int index);
	int num_players_;
	int ranks_;
	int suits_;
	int last_ask_;
	int first_out_;
	bool most_books_wins_;
	bool end_on_first_out_;
};

class GoFishGame : public Game {
 public:
  explicit GoFishGame(const GameParameters& params);
  int NumDistinctActions() const override {
		return std::max(num_players_ * ranks_, suits_ * ranks_);
	}
	int MaxChanceOutcomes() const override {
		return suits_ * ranks_;
	}
	int MaxChanceNodesInHistory() const override {
		return suits_ * ranks_;
	}
	// in principle the game coul go on forever since we don't
	// outlaw asking a player for a card you should know he doesn't have.
	int MaxGameLength() const override {
		return 10 * num_players_ * ranks_ * suits_;
	}
	double MaxUtility() const override { return +1; }
	double MinUtility() const override { return -1; }
	absl::optional<double> UtilitySum() const override { return 0; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new GoFishState(shared_from_this()));
  }
 std::vector<int> ObservationTensorShape() const;
 int NumPlayers() const override { return num_players_; }
 int Ranks() const { return ranks_; }
 int Suits() const { return suits_; }
 int InitialCards() const { return initial_cards_; }
 bool MostBooksWins() const { return most_books_wins_; }
 bool EndOnFirstOut() const { return end_on_first_out_; }

 private:
	int num_players_;
	int ranks_;
	int suits_;
	int initial_cards_;
	bool most_books_wins_;
	bool end_on_first_out_;
};

}  // namespace go_fish
}  // namespace open_spiel


#endif  // OPEN_SPIEL_GAMES_GO_FISH_H_

