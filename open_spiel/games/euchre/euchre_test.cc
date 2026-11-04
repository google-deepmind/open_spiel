// Copyright 2022 DeepMind Technologies Limited
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


#include "open_spiel/games/euchre/euchre.h"

#include <array>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace euchre {
namespace {

constexpr const char* kLoneDefenderNoStick =
    "euchre(allow_lone_defender=true,stick_the_dealer=false)";
constexpr const char* kGameStrings[] = {"euchre", kLoneDefenderNoStick};
constexpr int kNumRandomSims = 10;
constexpr int kNumResamples = 200;
constexpr int kKittySeed = 777;
constexpr int kNumPrivacySims = 20;
constexpr int kPrivacySeed = 2468;

// Offsets of the information state tensor's blocks.
constexpr int kBiddingOffset = kNumPlayers + kNumCards;
constexpr int kGoAloneOffset = kBiddingOffset + (kNumSuits + 1) * kMaxBids;
constexpr int kDiscardOffset = kGoAloneOffset + 3;

// The deal used by the scenario tests: each player p holds the 9 to K of
// suit p (North clubs, East diamonds, South hearts, West spades), the up-card
// is the ace of clubs and the kitty is the other three aces.
std::array<std::vector<int>, kNumPlayers> SuitHands() {
  std::array<std::vector<int>, kNumPlayers> hands;
  for (int player = 0; player < kNumPlayers; ++player) {
    for (int rank = 0; rank < kNumTricks; ++rank) {
      hands[player].push_back(Card(static_cast<Suit>(player), rank));
    }
  }
  return hands;
}
constexpr int kAceRank = kNumCardsPerSuit - 1;
const int kUpcard = Card(Suit::kClubs, kAceRank);

// Selects `dealer` and deals `hands` and `upcard`.
std::unique_ptr<State> DealtState(
    const std::string& game_string, Player dealer,
    const std::array<std::vector<int>, kNumPlayers>& hands, int upcard) {
  std::unique_ptr<State> state = LoadGame(game_string)->NewInitialState();
  state->ApplyAction(dealer);
  std::array<int, kNumPlayers> next{};
  for (int dealt = 0; dealt < kNumPlayers * kNumTricks; ++dealt) {
    const Player player = (dealer + dealt) % kNumPlayers;
    state->ApplyAction(hands[player][next[player]++]);
  }
  state->ApplyAction(upcard);
  return state;
}

std::unique_ptr<State> SuitDeal(const std::string& game_string,
                                Player dealer) {
  return DealtState(game_string, dealer, SuitHands(), kUpcard);
}

const EuchreState& AsEuchre(const State& state) {
  return static_cast<const EuchreState&>(state);
}

void BasicGameTests() {
  testing::LoadGameTest("euchre");
  testing::ChanceOutcomesTest(*LoadGame("euchre"));
  testing::RandomSimTest(*LoadGame("euchre"), 10);
  testing::RandomSimTest(*LoadGame(kLoneDefenderNoStick), kNumRandomSims);

  auto observer = LoadGame("euchre")
                      ->MakeObserver(kInfoStateObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("euchre"), observer);
}

void ResampleFromInfostateTest(const std::string& game_string) {
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::mt19937 rng(12345);
  int num_sims = 100;
  for (int sim = 0; sim < num_sims; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        for (int p = 0; p < state->NumPlayers(); ++p) {
          UniformProbabilitySampler sampler(static_cast<int>(rng()));
          std::unique_ptr<State> resampled =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         resampled->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         resampled->InformationStateString(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());
        }
      }
      std::vector<Action> actions;
      if (state->IsChanceNode()) {
        for (const auto& outcome : state->ChanceOutcomes()) {
          actions.push_back(outcome.first);
        }
      } else {
        actions = state->LegalActions();
      }
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
  }
}

// The kitty is never seen, so resampling must sometimes move a kitty card into
// a hand, for every viewer.
void ResampleMovesKittyTest(int seed) {
  std::shared_ptr<const Game> game = LoadGame("euchre");
  std::mt19937 rng(seed);
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[rng() % outcomes.size()].first);
  }
  const EuchreState& euchre_state = AsEuchre(*state);
  std::vector<int> kitty;
  for (int card = 0; card < kNumCards; ++card) {
    if (!euchre_state.CardHolder()[card].has_value() &&
        card != euchre_state.Upcard()) {
      kitty.push_back(card);
    }
  }
  SPIEL_CHECK_EQ(kitty.size(), 3);
  for (Player viewer = 0; viewer < kNumPlayers; ++viewer) {
    bool moved = false;
    for (int sample_seed = 0; sample_seed < kNumResamples && !moved;
         ++sample_seed) {
      UniformProbabilitySampler sampler(sample_seed);
      std::unique_ptr<State> sample =
          state->ResampleFromInfostate(viewer, sampler);
      for (int card : kitty) {
        moved |= AsEuchre(*sample).CardHolder()[card].has_value();
      }
    }
    SPIEL_CHECK_TRUE(moved);
  }
}

// The dealer's discard is face down: resampling for anyone but the dealer
// must sometimes change it, and resampling for the dealer never does.
void ResampleMovesDiscardTest() {
  std::unique_ptr<State> state = SuitDeal("euchre", /*dealer=*/0);
  for (int bid = 0; bid < kNumPlayers - 1; ++bid) {
    state->ApplyAction(kPassAction);
  }
  state->ApplyAction(kClubsTrumpAction);  // The dealer picks up.
  state->ApplyAction(kPlayWithPartnerAction);
  const int discard = Card(Suit::kClubs, 0);
  state->ApplyAction(discard);
  SPIEL_CHECK_EQ(AsEuchre(*state).Discard(), discard);
  for (Player viewer = 0; viewer < kNumPlayers; ++viewer) {
    bool moved = false;
    for (int sample_seed = 0; sample_seed < kNumResamples; ++sample_seed) {
      UniformProbabilitySampler sampler(sample_seed);
      std::unique_ptr<State> sample =
          state->ResampleFromInfostate(viewer, sampler);
      moved |= AsEuchre(*sample).Discard() != discard;
    }
    SPIEL_CHECK_EQ(moved, viewer != 0);
  }
}

// Every call is public and the bidding shows it: the dealer's first-round
// acceptance reads "Pick up!" and a second-round call names its suit.
void BiddingStringTest() {
  // The dealer picks up in the first round.
  std::unique_ptr<State> state = SuitDeal("euchre", /*dealer=*/0);
  for (int bid = 0; bid < kNumPlayers - 1; ++bid) {
    state->ApplyAction(kPassAction);
  }
  state->ApplyAction(kClubsTrumpAction);
  for (Player player = 0; player < kNumPlayers; ++player) {
    const std::string info = state->InformationStateString(player);
    SPIEL_CHECK_TRUE(absl::StrContains(info, "Pick up!"));
    SPIEL_CHECK_FALSE(absl::StrContains(info, "Clubs"));
  }

  // East names a suit in the second round: hearts and spades must differ.
  std::array<std::string, kNumPlayers> hearts_strings;
  for (Action call : {kHeartsTrumpAction, kSpadesTrumpAction}) {
    std::unique_ptr<State> second = SuitDeal("euchre", /*dealer=*/0);
    for (int bid = 0; bid < kNumPlayers; ++bid) {
      second->ApplyAction(kPassAction);
    }
    second->ApplyAction(call);
    const std::string suit = call == kHeartsTrumpAction ? "Hearts" : "Spades";
    for (Player player = 0; player < kNumPlayers; ++player) {
      const std::string info = second->InformationStateString(player);
      SPIEL_CHECK_TRUE(absl::StrContains(info, suit));
      SPIEL_CHECK_FALSE(absl::StrContains(info, "Pick up!"));
      if (call == kHeartsTrumpAction) {
        hearts_strings[player] = info;
      } else {
        SPIEL_CHECK_NE(info, hearts_strings[player]);
      }
    }
  }
}

// A pass sets the Pass slot of its own bid in the tensor, and the call that
// follows sets the suit slot of the next bid.
void BiddingTensorTest() {
  std::unique_ptr<State> state = SuitDeal("euchre", /*dealer=*/0);
  state->ApplyAction(kPassAction);
  for (Player player = 0; player < kNumPlayers; ++player) {
    const std::vector<float> tensor = state->InformationStateTensor(player);
    for (int slot = 0; slot < (kNumSuits + 1) * kMaxBids; ++slot) {
      SPIEL_CHECK_EQ(tensor[kBiddingOffset + slot], slot == kNumSuits ? 1 : 0);
    }
  }
  state->ApplyAction(kClubsTrumpAction);  // South orders up.
  for (Player player = 0; player < kNumPlayers; ++player) {
    const std::vector<float> tensor = state->InformationStateTensor(player);
    for (int slot = 0; slot < (kNumSuits + 1) * kMaxBids; ++slot) {
      const bool set = slot == kNumSuits ||
                       slot == (kNumSuits + 1) + static_cast<int>(Suit::kClubs);
      SPIEL_CHECK_EQ(tensor[kBiddingOffset + slot], set ? 1 : 0);
    }
  }
}

// The go-alone bits are [declarer alone, first defender alone, second
// defender alone].
void CheckGoAloneBits(const State& state, std::array<int, 3> expected) {
  for (Player player = 0; player < kNumPlayers; ++player) {
    const std::vector<float> tensor = state.InformationStateTensor(player);
    for (int bit = 0; bit < 3; ++bit) {
      SPIEL_CHECK_EQ(tensor[kGoAloneOffset + bit], expected[bit]);
    }
  }
}

void GoAloneTensorTest() {
  // Bidding: nobody has gone alone.
  std::unique_ptr<State> state = SuitDeal(kLoneDefenderNoStick, 0);
  CheckGoAloneBits(*state, {0, 0, 0});
  state->ApplyAction(kPassAction);
  CheckGoAloneBits(*state, {0, 0, 0});
  // East passed; South orders up and plays with a partner.
  state->ApplyAction(kClubsTrumpAction);
  CheckGoAloneBits(*state, {0, 0, 0});
  state->ApplyAction(kPlayWithPartnerAction);
  CheckGoAloneBits(*state, {0, 0, 0});
  // The defenders are West (first) and East (second); East is asked first.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(kGoAloneAction);
  CheckGoAloneBits(*state, {0, 0, 1});

  // The declarer goes alone.
  std::unique_ptr<State> alone = SuitDeal("euchre", 0);
  alone->ApplyAction(kClubsTrumpAction);  // East orders up.
  alone->ApplyAction(kGoAloneAction);
  CheckGoAloneBits(*alone, {1, 0, 0});
}

// The declarer says whether they play alone before the dealer takes up the
// up-card, and a dealer whose partner orders up and plays alone sits out
// without taking the up-card.
void GoAloneBeforeDiscardTest() {
  // South, the dealer's partner, orders up and plays alone.
  std::unique_ptr<State> state = SuitDeal("euchre", /*dealer=*/0);
  state->ApplyAction(kPassAction);
  state->ApplyAction(kClubsTrumpAction);
  SPIEL_CHECK_TRUE(AsEuchre(*state).CurrentPhase() == Phase::kGoAlone);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  SPIEL_CHECK_FALSE(AsEuchre(*state).CardHolder()[kUpcard].has_value());
  state->ApplyAction(kGoAloneAction);
  const EuchreState& lone = AsEuchre(*state);
  SPIEL_CHECK_TRUE(lone.CurrentPhase() == Phase::kPlay);
  SPIEL_CHECK_FALSE(lone.CardHolder()[kUpcard].has_value());
  SPIEL_CHECK_EQ(lone.Discard(), kInvalidAction);
  SPIEL_CHECK_FALSE(lone.ActivePlayers()[0]);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  for (int rank = 0; rank < kNumTricks; ++rank) {
    SPIEL_CHECK_EQ(lone.CardHolder()[Card(Suit::kClubs, rank)], 0);
  }

  // South orders up and plays with the dealer: the dealer takes the up-card
  // and discards after the go-alone decision.
  std::unique_ptr<State> with_partner = SuitDeal("euchre", /*dealer=*/0);
  with_partner->ApplyAction(kPassAction);
  with_partner->ApplyAction(kClubsTrumpAction);
  with_partner->ApplyAction(kPlayWithPartnerAction);
  SPIEL_CHECK_TRUE(AsEuchre(*with_partner).CurrentPhase() == Phase::kDiscard);
  SPIEL_CHECK_EQ(with_partner->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(AsEuchre(*with_partner).CardHolder()[kUpcard], 0);

  // East orders up and plays alone: the dealer is a defender and still takes
  // the up-card.
  std::unique_ptr<State> defender_dealer = SuitDeal("euchre", /*dealer=*/0);
  defender_dealer->ApplyAction(kClubsTrumpAction);
  defender_dealer->ApplyAction(kGoAloneAction);
  SPIEL_CHECK_TRUE(AsEuchre(*defender_dealer).CurrentPhase() ==
                   Phase::kDiscard);
  SPIEL_CHECK_EQ(AsEuchre(*defender_dealer).CardHolder()[kUpcard], 0);
  defender_dealer->ApplyAction(Card(Suit::kClubs, 0));
  SPIEL_CHECK_TRUE(AsEuchre(*defender_dealer).CurrentPhase() == Phase::kPlay);
  SPIEL_CHECK_EQ(defender_dealer->CurrentPlayer(), 1);
}

// The dealer's information state records the dealer's own discard (perfect
// recall), and no other player's does. In the two games compared the dealer
// keeps the same five cards and discards a different one (the nine of clubs,
// or the ace of diamonds, which is in the kitty in the first game).
void DiscardInformationTest() {
  const std::array<int, 2> discards = {Card(Suit::kClubs, 0),
                                       Card(Suit::kDiamonds, kAceRank)};
  std::array<std::unique_ptr<State>, 2> states;
  for (int game = 0; game < 2; ++game) {
    std::array<std::vector<int>, kNumPlayers> hands = SuitHands();
    hands[0][0] = discards[game];
    states[game] = DealtState("euchre", /*dealer=*/0, hands, kUpcard);
    for (int bid = 0; bid < kNumPlayers - 1; ++bid) {
      states[game]->ApplyAction(kPassAction);
    }
    states[game]->ApplyAction(kClubsTrumpAction);  // The dealer picks up.
    states[game]->ApplyAction(kPlayWithPartnerAction);
    states[game]->ApplyAction(discards[game]);
    const std::vector<float> tensor = states[game]->InformationStateTensor(0);
    for (int card = 0; card < kNumCards; ++card) {
      SPIEL_CHECK_EQ(tensor[kDiscardOffset + card],
                     card == discards[game] ? 1 : 0);
    }
    SPIEL_CHECK_TRUE(absl::StrContains(
        states[game]->InformationStateString(0),
        absl::StrCat("Discard: ", CardString(discards[game]))));
  }
  int num_compared = 0;
  while (!states[0]->IsTerminal()) {
    SPIEL_CHECK_EQ(states[0]->LegalActions(), states[1]->LegalActions());
    SPIEL_CHECK_NE(states[0]->InformationStateString(0),
                   states[1]->InformationStateString(0));
    SPIEL_CHECK_NE(states[0]->InformationStateTensor(0),
                   states[1]->InformationStateTensor(0));
    for (Player player = 1; player < kNumPlayers; ++player) {
      SPIEL_CHECK_EQ(states[0]->InformationStateString(player),
                     states[1]->InformationStateString(player));
      SPIEL_CHECK_EQ(states[0]->InformationStateTensor(player),
                     states[1]->InformationStateTensor(player));
      const std::vector<float> tensor =
          states[0]->InformationStateTensor(player);
      for (int card = 0; card < kNumCards; ++card) {
        SPIEL_CHECK_EQ(tensor[kDiscardOffset + card], 0);
      }
    }
    ++num_compared;
    const Action action = states[0]->LegalActions()[0];
    states[0]->ApplyAction(action);
    states[1]->ApplyAction(action);
  }
  SPIEL_CHECK_EQ(num_compared, kNumPlayers * kNumTricks);
}

// The cards `viewer` cannot see: neither in the viewer's original hand, nor
// the up-card, nor played, nor (for the dealer) the dealer's own discard.
std::vector<int> HiddenCards(const State& state, Player viewer) {
  const EuchreState& euchre_state = AsEuchre(state);
  const std::vector<Action> history = state.History();
  std::vector<bool> visible(kNumCards, false);
  const Player dealer = euchre_state.Dealer();
  for (int dealt = 0; dealt < kNumPlayers * kNumTricks; ++dealt) {
    if ((dealer + dealt) % kNumPlayers == viewer) {
      visible[history[1 + dealt]] = true;
    }
  }
  visible[euchre_state.Upcard()] = true;
  const int num_history = static_cast<int>(history.size());
  for (int index = num_history - euchre_state.NumCardsPlayed();
       index < num_history; ++index) {
    visible[history[index]] = true;
  }
  if (viewer == dealer && euchre_state.Discard() != kInvalidAction) {
    visible[euchre_state.Discard()] = true;
  }
  std::vector<int> hidden;
  for (int card = 0; card < kNumCards; ++card) {
    if (!visible[card]) hidden.push_back(card);
  }
  return hidden;
}

// The state reached by the same history with cards `a` and `b` exchanged
// wherever a card is dealt or chosen.
std::unique_ptr<State> SwapCards(const State& state, int a, int b) {
  std::unique_ptr<State> swapped = state.GetGame()->NewInitialState();
  const std::vector<Action> history = state.History();
  for (int index = 0; index < static_cast<int>(history.size()); ++index) {
    Action action = history[index];
    if (index > 0 && action == a) {
      action = b;
    } else if (index > 0 && action == b) {
      action = a;
    }
    swapped->ApplyAction(action);
  }
  return swapped;
}

// No player's information state or observation reveals a card that player
// cannot see. For every viewer, exchanging two cards hidden from the viewer
// (in other hands, in the kitty or, for anyone but the dealer, the dealer's
// discard) leaves the viewer's strings and tensor unchanged. The two cards
// have the same suit and are not jacks, so every action stays legal.
void HiddenCardsPrivacyTest(const std::string& game_string) {
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::mt19937 rng(kPrivacySeed);
  int num_pairs = 0;
  int num_discard_pairs = 0;
  for (int sim = 0; sim < kNumPrivacySims; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        const EuchreState& euchre_state = AsEuchre(*state);
        for (Player viewer = 0; viewer < kNumPlayers; ++viewer) {
          const std::vector<int> hidden = HiddenCards(*state, viewer);
          for (int a : hidden) {
            for (int b : hidden) {
              if (a >= b || CardSuit(a) != CardSuit(b) ||
                  CardRank(a) == kJackRank || CardRank(b) == kJackRank) {
                continue;
              }
              std::unique_ptr<State> swapped = SwapCards(*state, a, b);
              SPIEL_CHECK_EQ(state->InformationStateString(viewer),
                             swapped->InformationStateString(viewer));
              SPIEL_CHECK_EQ(state->ObservationString(viewer),
                             swapped->ObservationString(viewer));
              SPIEL_CHECK_EQ(state->InformationStateTensor(viewer),
                             swapped->InformationStateTensor(viewer));
              ++num_pairs;
              if (a == euchre_state.Discard() || b == euchre_state.Discard()) {
                ++num_discard_pairs;
              }
            }
          }
        }
      }
      std::vector<Action> actions;
      if (state->IsChanceNode()) {
        for (const auto& outcome : state->ChanceOutcomes()) {
          actions.push_back(outcome.first);
        }
      } else {
        actions = state->LegalActions();
      }
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
  }
  SPIEL_CHECK_GT(num_pairs, 0);
  SPIEL_CHECK_GT(num_discard_pairs, 0);
}

}  // namespace
}  // namespace euchre
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::euchre::BasicGameTests();
  for (const char* game_string : open_spiel::euchre::kGameStrings) {
    open_spiel::euchre::ResampleFromInfostateTest(game_string);
    open_spiel::euchre::HiddenCardsPrivacyTest(game_string);
  }
  open_spiel::euchre::ResampleMovesKittyTest(open_spiel::euchre::kKittySeed);
  open_spiel::euchre::ResampleMovesDiscardTest();
  open_spiel::euchre::BiddingStringTest();
  open_spiel::euchre::BiddingTensorTest();
  open_spiel::euchre::GoAloneTensorTest();
  open_spiel::euchre::GoAloneBeforeDiscardTest();
  open_spiel::euchre::DiscardInformationTest();
}
