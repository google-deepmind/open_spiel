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

#include "open_spiel/games/cribbage/cribbage.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <memory>

#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

constexpr int kSeed = 2871611;

namespace open_spiel {
namespace cribbage {
namespace {

namespace testing = open_spiel::testing;

void CardToStringTest() {
	std::cout << "CardToStringTest" << std::endl;
	std::vector<std::string> card_strings;
	card_strings.reserve(52);
	std::string suit_names(kSuitNames);
	std::string ranks(kRanks);

  for (int i = 0; i < 52; ++i) {
		Card card = GetCard(i);
		std::string card_string = card.to_string();
		size_t rank_pos = ranks.find(card_string[0]);
		SPIEL_CHECK_TRUE(rank_pos != std::string::npos);
		size_t suit_pos = suit_names.find(card_string[1]);
		SPIEL_CHECK_TRUE(suit_pos != std::string::npos);
		auto iter = std::find(card_strings.begin(), card_strings.end(),
												  card_string);
		SPIEL_CHECK_TRUE(iter == card_strings.end());
		card_strings.push_back(card_string);
	}
}

void BasicLoadTest() {
	std::cout << "BasicLoadTest" << std::endl;
	std::shared_ptr<const Game> game = LoadGame("cribbage");
	std::unique_ptr<State> state = game->NewInitialState();
	std::cout << state->ToString() << std::endl;
	SPIEL_CHECK_EQ(game->NumPlayers(), 2);
	
	game = LoadGame("cribbage(players=3)");
	state = game->NewInitialState();
	std::cout << state->ToString() << std::endl;
	SPIEL_CHECK_EQ(game->NumPlayers(), 3);
	
	game = LoadGame("cribbage(players=4)");
	state = game->NewInitialState();
	std::cout << state->ToString() << std::endl;
	SPIEL_CHECK_EQ(game->NumPlayers(), 4);
}

void BasicOneTurnPlaythrough() {
	std::cout << "BasicOneTurnPlaythroughTest" << std::endl;
  std::mt19937 rng(kSeed);
	std::shared_ptr<const Game> game = LoadGame("cribbage");
	std::unique_ptr<State> state = game->NewInitialState();
	CribbageState* crib_state = static_cast<CribbageState*>(state.get());

	// Deal.
	while (state->IsChanceNode()) {
		std::cout << state->ToString() << std::endl;
		double z = absl::Uniform(rng, 0.0, 1.0);
		Action outcome = SampleAction(state->ChanceOutcomes(), z).first;
		std::cout << "Sampled outcome: "
		          << state->ActionToString(kChancePlayerId, outcome) << std::endl;
		state->ApplyAction(outcome);
	}

	// Card choices.
	for (int p = 0; p < game->NumPlayers(); ++p) {
		std::cout << state->ToString() << std::endl;
		std::vector<Action> legal_actions = state->LegalActions();
		int idx = absl::Uniform<int>(rng, 0, legal_actions.size());
		Action action = legal_actions[idx];
		std::cout << "Sampled action: "
		          << state->ActionToString(state->CurrentPlayer(), action)
							<< std::endl;
		state->ApplyAction(action);
	}

	// Starter.
	std::cout << state->ToString() << std::endl;
	double z = absl::Uniform(rng, 0.0, 1.0);
	Action outcome = SampleAction(state->ChanceOutcomes(), z).first;
	std::cout << "Sampled outcome: "
						<< state->ActionToString(kChancePlayerId, outcome) << std::endl;
	state->ApplyAction(outcome);
	SPIEL_CHECK_FALSE(state->IsChanceNode());

	// Play phase.
	while (crib_state->round() < 1) {
		std::cout << state->ToString() << std::endl;
		std::vector<Action> legal_actions = state->LegalActions();
		int idx = absl::Uniform<int>(rng, 0, legal_actions.size());
		Action action = legal_actions[idx];
		std::cout << "Sampled action: "
		          << state->ActionToString(state->CurrentPlayer(), action)
							<< std::endl;
		state->ApplyAction(action);
	}

	std::cout << state->ToString() << std::endl;
}

void HandScoringTests() {
  // Suit order: CDHS
  std::vector<Card> hand;
  hand = GetHandFromStrings({"QC", "TD", "7H", "9H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 4);
  hand = GetHandFromStrings({"QC", "QD", "7H", "9H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 6);
  hand = GetHandFromStrings({"QC", "QD", "QH", "9H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 12);
  hand = GetHandFromStrings({"QC", "QD", "QH", "5S", "QS"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 20);
  hand = GetHandFromStrings({"5C", "QC", "5D", "5H", "5S"});
  SPIEL_CHECK_EQ(ScoreHand(hand), 28);  // 8 for 15s w/ Q, 12 4-of-K, 8 more 15s
  hand = GetHandFromStrings({"QC", "JD", "7H", "9H"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("5D")), 5);  // 4 15s + jack
  hand = GetHandFromStrings({"QC", "JD", "7H", "9H"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("5S")), 4);  // 4 15s
  // Flushes. 5-card flush, then a 4-card flush, then no flush.
  hand = GetHandFromStrings({"QC", "TC", "8C", "4C"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("2C")), 5);
  hand = GetHandFromStrings({"QC", "TC", "8C", "4C"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("2D")), 4);
  hand = GetHandFromStrings({"QD", "TC", "8C", "4C"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("2C")), 0);
  // 5-card flush and run of 5 + nobs = 11.
  hand = GetHandFromStrings({"9C", "TC", "JC", "QC"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("KC")), 11);
  // Examples of runs from the rule book.
  hand = GetHandFromStrings({"5C", "6C", "7C", "8D"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("8S")), 14);
  // 3 runs of 3 (9) + 3-of-a-kind (6) + three 15s (8) = 21.
  hand = GetHandFromStrings({"4C", "4D", "4S", "5D"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("6S")), 21);
  // 4 runs of 3 (12) + 2 pairs (4) + 2 15s (4) = 20.
  hand = GetHandFromStrings({"6C", "6D", "7S", "7D"});
  SPIEL_CHECK_EQ(ScoreHand(hand, GetCardByString("8S")), 20);
}

void BasicCribbageTests() {
  testing::RandomSimTestNoSerialize(*LoadGame("cribbage"), 10);
}

}  // namespace
}  // namespace cribbage
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::cribbage::CardToStringTest();
  open_spiel::cribbage::BasicLoadTest();
  open_spiel::cribbage::BasicCribbageTests();
	open_spiel::cribbage::BasicOneTurnPlaythrough();
  open_spiel::cribbage::HandScoringTests();
  open_spiel::cribbage::BasicCribbageTests();
}

