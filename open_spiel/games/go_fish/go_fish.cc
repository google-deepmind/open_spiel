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

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "open_spiel/games/go_fish/go_fish.h"

#include <vector>


namespace open_spiel {
namespace go_fish {
// Facts about the game.
const GameType kGameType{/*short_name=*/"go_fish",
                         /*long_name=*/"Go Fish",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/10,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"ranks", GameParameter(13)},
                          {"suits", GameParameter(4)},
                          {"initial_cards", GameParameter(-1)},
                          {"most_books_wins", GameParameter(true)},
												  {"end_on_first_out", GameParameter(false)},
												  {"ask_after_empty_draw", GameParameter(true)}}
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GoFishGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::string RankString(int rank) {
	return std::string(1, 'a' + rank);
}

int RankFromChar(char c) {
	return c - 'a';
}

std::string PhaseString(Phase phase) {
	switch(phase) {
		case kDeal:
			return "Deal";
		case kFish:
			return "Fish";
		case kAsk:
			return "Ask";
		case kEmptyDraw:
			return "EmptyDraw";
		case kTerminal:
			return "GameOver";
	}
	// this never happens
	SpielFatalError("Illegal phase");
	return "WTF";
}

Phase PhaseFromString(std::string s) {
	if (s == "Deal") return kDeal;
  if (s == "Fish") return kFish;
	if (s == "Ask") return kAsk;
	if (s =="EmptyDraw") return kEmptyDraw;
	if (s == "GameOver") return kTerminal;
	SpielFatalError("Illegal phase");
	// this never happens
	return kTerminal;

}

void GoFishState::ParseRankCounts(
    absl::string_view s, std::vector<int>* counts) {
  SPIEL_CHECK_EQ(s.size() % 2, 0);

  for (int i = 0; i < s.size(); i += 2) {
    int rank = RankFromChar(s[i]);
    int count = s[i + 1] - '0';

    SPIEL_CHECK_GE(rank, 0);
    SPIEL_CHECK_LT(rank, ranks_);
    SPIEL_CHECK_GE(count, 0);
    SPIEL_CHECK_LE(count, suits_);

    (*counts)[rank] = count;
  }
}

GoFishState::GoFishState(std::shared_ptr<const Game> game,
		const std::string& state_str)
  : State(game),
    ranks_(static_cast<const GoFishGame*>(game.get())->Ranks()),
    suits_(static_cast<const GoFishGame*>(game.get())->Suits()),
    most_books_wins_(static_cast<const GoFishGame*>(game.get())->MostBooksWins()),
    end_on_first_out_(static_cast<const GoFishGame*>(game.get())->EndOnFirstOut()),
		ask_after_empty_draw_(static_cast<const GoFishGame*>(game.get())->AskAfterEmptyDraw()),
    num_players_(static_cast<const GoFishGame*>(game.get())->NumPlayers()) {
	// initialize all the vectors
	player_cards_.assign(num_players_, std::vector<int>(ranks_, 0));
	pool_.assign(ranks_, suits_);
	booked_.assign(ranks_, false);
	player_books_.assign(num_players_, 0);
	player_did_ask_.assign(num_players_, std::vector<int>(ranks_, 0));
	player_was_asked_.assign(num_players_, std::vector<bool>(ranks_, false));
	drawn_since_was_asked_.assign(num_players_, std::vector<int>(ranks_, 0));
	player_min_.assign(num_players_, std::vector<int>(ranks_, 0));
	current_player_ = 0;
	first_out_ = -1;
	int parent_initial = static_cast<const GoFishGame*>(game.get())->InitialCards();
	// TODO add some sanity checks
	if (parent_initial == -1) {
		if (num_players_ == 2){
			initial_cards_ = 7;
		} else {
			initial_cards_ = 5;
		}
	} else {
		initial_cards_ = parent_initial;
	}
	phase_ = kDeal;
	if (!state_str.empty()) {
		std::fill(pool_.begin(), pool_.end(), 0);
    std::vector<std::string> lines = absl::StrSplit(state_str, '\n');
    SPIEL_CHECK_EQ(lines.size(), num_players_ + 3);

    phase_ = PhaseFromString(lines[0]);
    current_player_ = std::stoi(lines[1]);

    for (int pid = 0; pid < num_players_; ++pid) {
       std::vector<std::string> parts = absl::StrSplit(lines[pid + 2], ':');
       SPIEL_CHECK_EQ(parts.size(), 2);

       ParseRankCounts(parts[0], &player_cards_[pid]);
       player_books_[pid] = std::stoi(parts[1]);
   }

   ParseRankCounts(lines[num_players_ + 2], &pool_);
	 int booked = 0;
	 int unbooked = 0;
	 for (int rank = 0; rank < ranks_; ++rank) {
      int total = pool_[rank];
      for (int pid = 0; pid < num_players_; ++pid) {
        total += player_cards_[pid][rank];
      }
			if (total > 0) {
        SPIEL_CHECK_EQ(total, suits_);
				++unbooked;
			} else {
				booked_[rank] = true;
				++booked;
			}
	 }
	 SPIEL_CHECK_EQ(booked + unbooked, ranks_);
  }
}

std::unique_ptr<State> GoFishState::Clone() const {
  return std::unique_ptr<State>(new GoFishState(*this));
}

int GoFishState::PoolSize() const {
	int cards = 0;
	for (int ii = 0; ii < pool_.size(); ++ii){
		cards += pool_[ii];
	}
	return cards;
}

int GoFishState::PlayerCounts(int player_id) const {
	int cards = 0;
	for (int ii = 0; ii < player_cards_[player_id].size(); ++ii){
		cards += player_cards_[player_id][ii];
	}
	return cards;
}

void GoFishState::DoApplyAction(Action move_id) {
	if (phase_ == kDeal) {
		int player_id = 0;
		int pc = PlayerCounts(0);
		while (pc == initial_cards_) {
			++player_id;
			pc = PlayerCounts(player_id);
		}
		int rank = move_id;
		--pool_[rank];
		player_cards_[player_id][rank] += 1;
		if (pc + 1 == initial_cards_ && player_id == num_players_ - 1){
			current_player_ = Player(0);
			phase_ = kAsk;
		}
		// Check for rare case where a player scored book during deal.
		for (int pid = 0; pid < num_players_; ++pid) {
			for (int rank0 = 0; rank0 < ranks_; ++rank0){
				CheckBook(pid, rank0);
			}
		}
		return;
	}
	bool hit = false;
	bool advance = true;
	if (phase_ == kAsk) {
		int event_player = current_player_;
		int target = move_id / ranks_;
		int rank = move_id % ranks_;
		if (player_min_[current_player_][rank] == 0) {
			player_min_[current_player_][rank] = 1;
		}
		player_did_ask_[current_player_][rank] += 1;

		player_was_asked_[target][rank] = true;
		drawn_since_was_asked_[target][rank] = 0;
		player_min_[target][rank] = 0;
    bool made_book;
		int received = 0;
		if (player_cards_[target][rank] > 0) {
			hit = true;
			received = player_cards_[target][rank]; 
			player_cards_[current_player_][rank] += received;
			player_min_[current_player_][rank] += received;
			player_cards_[target][rank] = 0;
			made_book = CheckBook(current_player_, rank);
			if (phase_ != kTerminal) {
			  CheckEmptyAsk();
			}
		} else { // go fish
			if (PoolSize() > 0) {
				phase_ = kFish;
				last_ask_ = rank;
			} else { // pool empty, play pases to next player with cards.
				  AdvancePlayer();
			}
		}
		if (current_player_ != event_player) last_ask_ = -1;
		int booked = made_book ? rank : -1; 
		Event event(event_player, target, rank, received, booked);
		events_.push_back(event);
		return;
	}
	if (phase_ == kFish || phase_ == kEmptyDraw) {
		int rank = move_id;
		if (phase_ == kFish && rank == last_ask_) hit = true;
		--pool_[rank];
		player_cards_[current_player_][rank] += 1;
		for (int rank0 = 0; rank0 < ranks_; ++rank0) {
       if (player_was_asked_[current_player_][rank0]) {
				  drawn_since_was_asked_[current_player_][rank0] += 1;
			 }
		}
		CheckBook(current_player_, rank);
		if (phase_ != kTerminal) {
			bool advance = false;
			if (phase_ == kEmptyDraw && ask_after_empty_draw_) {
				advance = false;
			}
			if (phase_ == kFish && rank == last_ask_) {
					advance = false;
			}
			if (advance) AdvancePlayer();
	  }
	}
	last_ask_ = -1;
	phase_ = kAsk;
	CheckEmptyAsk();
}

// special case: If a player would be asking but he has no
// cards left (and there are cards in the pool) he fishes.
// If ask_after_empty_draw is true he may then
// ask for what he drew.
void GoFishState::CheckEmptyAsk() {
	if (PlayerCounts(current_player_) == 0) {
		  if (PoolSize() > 0) {
			   phase_ = kEmptyDraw;
			   last_ask_ = -1;
			   return;
			} else {
				AdvancePlayer();
			}
	}
}

std::string GoFishState::ActionToString(Player player, Action action_id) const {
	if (player == kChancePlayerId) {
    return RankString(action_id);
  }
	int target = action_id / ranks_;
	int rank = action_id % ranks_;
	return  absl::StrCat('0' + target, 'a' + rank);
}

std::vector<std::pair<Action, double>> GoFishState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
	int num_outcomes = PoolSize();
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_outcomes);

  for (int rank = 0; rank < ranks_; rank++) {
	  if (pool_[rank] > 0) {
		  double prob = 1.0 * pool_[rank] / num_outcomes;
      outcomes.emplace_back(std::make_pair(rank, prob));
		}
  }
  return outcomes;
}

Player GoFishState::CurrentPlayer() const {
  if (phase_ == kDeal || phase_ == kFish || phase_ == kEmptyDraw) {
    return kChancePlayerId;
  }
	if (phase_ == kTerminal) return kTerminalPlayerId;
  return current_player_;
}

bool GoFishState::CheckBook(int player_id, int rank) {
	if (player_cards_[player_id][rank] < suits_) return false;
	player_books_[player_id] += 1;
	player_cards_[player_id][rank] = 0;
	player_min_[player_id][rank] = 0;
	booked_[rank] = true;
	// Check if player is now out of cards.
	if (PlayerCounts(player_id) == 0) {
		if (end_on_first_out_) {
			first_out_ = player_id;
			phase_ = kTerminal;
			return true;
		}
	}
	if ( PoolSize() > 0) return true;
	// if any player still has cards we continue
	for (int check_player = 0; check_player < num_players_; ++check_player) {
		if (PlayerCounts(check_player) > 0 ) {
			return true;
		}
	}
	// If we got here there are no cards left
	phase_ =  kTerminal;
  return true;
}

void GoFishState::AdvancePlayer() {
	int old = current_player_;
	current_player_ += 1;
	if (current_player_ == num_players_) {
		current_player_ = 0;
	}
	if (PlayerCounts(current_player_) > 0) return;
	if (PoolSize() > 0) {
		phase_ = kEmptyDraw;
		return;
	}
	while (PlayerCounts(current_player_) == 0) {
		current_player_ += 1;
		if (current_player_ == num_players_) {
			current_player_ = 0;	
		}
		SPIEL_CHECK_NE(current_player_, old);
	}
}

std::vector<double> GoFishState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }
	// TODO cache
	std::vector<double> result(num_players_);
	if (!most_books_wins_) {
		for (int ii = 0; ii < num_players_; ++ii) {
			if (ii == first_out_) { 
				result[ii] = (1.0);
			} else {
				result[ii] = (-1.0 / (num_players_ - 1));
			}	
		}
		return result;
	} else {
		int max_score = *std::max_element(player_books_.begin(), player_books_.end());
		int num_winners = std::count(player_books_.begin(), player_books_.end(), max_score);
		if (num_winners == num_players_) { // all way tie
		  return std::vector<double>(num_players_, 0.0);
		}
		for (int p = 0; p < num_players_; ++p) {
			if (player_books_[p] == max_score) {
				result[p] = 1.0/ num_winners;
			} else {
				result[p] = -1.0 / (num_players_ - num_winners);
			}
		}
	}
	return result;
}

std::vector<Action> GoFishState::GenerateAsks(int player_id) const {
	std::vector<Action> result;
	for (int target = 0; target < num_players_; ++target) {
		if (target == player_id) continue; // no self ask
		if (PlayerCounts(target) == 0) continue;
		for (int rank = 0; rank < ranks_; ++rank) {
			if (player_cards_[player_id][rank] > 0){
				result.push_back(Action(target * ranks_ + rank));
		  }
	  }
	}
	return result;
}

// Draw action should represent ranks which can be drawn
std::vector<Action> GoFishState::GenerateDraws() const {
	std::vector<Action> result;
	for (int rank = 0; rank < ranks_; ++rank) {
		if (pool_[rank] > 0) {
			result.push_back(rank);
		}
	}
	return result;
}

std::vector<Action> GoFishState::LegalActions() const {
	// TODO cache
	if (phase_ == kFish || phase_ == kDeal || phase_ == kEmptyDraw) {
		return GenerateDraws();
	}
	if (phase_ == kAsk) {
		return GenerateAsks(current_player_);
	}
	SPIEL_CHECK_EQ(phase_, kTerminal); // never happens?
	return std::vector<Action>();
}

std::string  GoFishState::ToString() const {
	std::string result;
	absl::StrAppend(&result, PhaseString(phase_), "\n");
	absl::StrAppend(&result, current_player_, "\n");
	for (int pid = 0; pid < num_players_; ++pid) {
		for (int rank = 0; rank < ranks_; ++rank) {
			if (player_cards_[pid][rank] > 0) {
	       absl::StrAppend(&result, RankString(rank), player_cards_[pid][rank]);
			}
		}
		absl::StrAppend(&result, ":");
		absl::StrAppend(&result, player_books_[pid], "\n");
	}
	for (int rank = 0; rank < ranks_; ++rank) {
     if (pool_[rank] > 0) {
			 absl::StrAppend(&result, RankString(rank),pool_[rank]);
		 }
	}
	// no terminal new line :-)
	return result;
}

void GoFishState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.0f);
  int offset = 0;
	// player knows hism own cards for sure.
	// encode counts as fraction of total
	for (int rank = 0; rank < ranks_; ++rank) {
		values[offset++] = 1.0f * player_cards_[player][rank] / suits_;
	}
	// everything else is common info
	// one hots for phase
	if (phase_ == kDeal) values[offset++] = 1;
	if (phase_ == kAsk) values[offset++] = 1;
	if (phase_ == kFish) values[offset++] = 1;
	if (phase_ == kTerminal) values[offset++] = 1;
	// pool size.
	values[offset++] = 1.0f * PoolSize() / (ranks_ * suits_);
	for (int pid = 0; pid < num_players_; ++pid) {
		// Add common info about each player
	  // outer fields count = players
    values[offset++] = pid == current_player_ ? 1.0f : 0.0f;
		values[offset++] = 1.0f * player_books_[pid] / suits_;
		values[offset++] = PlayerCounts(pid);
		for (int rank = 0; rank < ranks_; ++rank) {
      // inner fields count = players * ranks
      values[offset++] = player_did_ask_[pid][rank] ? 1.0f : 0.0f;
      values[offset++] = player_was_asked_[pid][rank] ? 1.0f : 0.0f;
      values[offset++] = 1.0f * drawn_since_was_asked_[pid][rank]/ (ranks_ * suits_);
      values[offset++] = 1.0f * player_min_[pid][rank]/ suits_;
		}
	}
	// count = ranks
	for (int rank = 0; rank < ranks_; ++rank) {
		values[offset++] = booked_[rank] ? 1.0f : 0.0f;
	}
}

std::string GoFishState::ObservationString(Player player) const {
	std::string result;
	absl::StrAppend(&result, "Phase ", PhaseString(phase_), "\n");
	absl::StrAppend(&result, "Current Player ",  current_player_, "\n");
  absl::StrAppend(&result, "Your cards: ");
	for (int rank = 0; rank < ranks_; ++rank) {
		if (player_cards_[player][rank] > 0) {
			absl::StrAppend(&result, RankString(rank), player_cards_[player][rank], " ");
		}
	}
	absl::StrAppend(&result, "\n");
	for (int pid = 0; pid < num_players_; ++pid) {
		absl::StrAppend(&result, "player ", pid, " cards ", PlayerCounts(pid),
				" books ", player_books_[pid], "\n");
	}
	// history going back in time to last move by player
	if (events_.size() == 0) return result;
	int index = events_.size() - 1;
	while (index >=0 && events_[index].player_id != player) {
		 absl::StrAppend(&result, events_[index].ToString(), "\n");
		 --index;
	}
  return result;
}


std::string GoFishState::InformationStateString(Player player) const {
  return ObservationString(player);
}

std::vector<int> GoFishGame::ObservationTensorShape() const {
	int size = 4 + // phase one hots
					   2 * ranks_ + // secret info, booked
						 3 * num_players_ +  // current, player_booked, player counts
						 4 * num_players_ * ranks_;
  return {size};
}

std::unique_ptr<State> GoFishGame::NewInitialState() const {
  return std::unique_ptr<State>(new GoFishState(shared_from_this()));
}

std::unique_ptr<State> GoFishGame::NewInitialState(
    const std::string& state) const {
  return std::unique_ptr<State>(new GoFishState(shared_from_this(), state));
}

GoFishGame::GoFishGame(const GameParameters& params)
		: Game(kGameType, params),
		num_players_(ParameterValue<int>("players")),
		ranks_(ParameterValue<int>("ranks")),
		suits_(ParameterValue<int>("suits")),
		initial_cards_(ParameterValue<int>("initial_cards")),
		most_books_wins_(ParameterValue<bool>("most_books_wins")),
		end_on_first_out_(ParameterValue<bool>("end_on_first_out")),
		ask_after_empty_draw_(ParameterValue<bool>("ask_after_empty_draw")) {
}

}  // go_fish
}  // open_spiel
