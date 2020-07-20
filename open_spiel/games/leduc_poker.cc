// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/leduc_poker.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace leduc_poker {
namespace {

constexpr double kAnte = 1;

const GameType kGameType{/*short_name=*/"leduc_poker",
                         /*long_name=*/"Leduc Poker",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/10,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"action_mapping", GameParameter(false)},
                          {"suit_isomorphism", GameParameter(false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LeducGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace
LeducState::LeducState(std::shared_ptr<const Game> game,
                       bool action_mapping, bool suit_isomorphism)
    : State(game),
      cur_player_(kChancePlayerId),
      num_calls_(0),
      num_raises_(0),
      round_(1),   // Round number (1 or 2).
      stakes_(1),  // The current 'level' of the bet.
      num_winners_(-1),
      pot_(kAnte * game->NumPlayers()),  // Number of chips in the pot.
      public_card_(kInvalidCard),
      // Number of cards remaining; not equal deck_.size()!
      deck_size_((game->NumPlayers() + 1) * kNumSuits),
      private_cards_dealt_(0),
      remaining_players_(game->NumPlayers()),
      // Is this player a winner? Indexed by pid.
      winner_(game->NumPlayers(), false),
      // Each player's single private card. Indexed by pid.
      private_cards_(game->NumPlayers(), kInvalidCard),
      // How much money each player has, indexed by pid.
      money_(game->NumPlayers(), kStartingMoney - kAnte),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(game->NumPlayers(), kAnte),
      // Flag for whether the player has folded, indexed by pid.
      folded_(game->NumPlayers(), false),
      // Sequence of actions for each round. Needed to report information state.
      round1_sequence_(),
      round2_sequence_(),
      // Always regard all actions as legal, and internally map otherwise
      // illegal actions to check/call.
      action_mapping_(action_mapping),
      // Players cannot distinguish between cards of different suits with the
      // same rank.
      suit_isomorphism_(suit_isomorphism) {
  // Cards by value (0-6 for standard 2-player game, kInvalidCard if no longer
  // in the deck.)
  deck_.resize(deck_size_);
  std::iota(deck_.begin(), deck_.end(), 0);
}

int LeducState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

// In a chance node, `move` should be the card to deal to the current
// underlying player.
// On a player node, it should be ActionType::{kFold, kCall, kRaise}
void LeducState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, deck_.size());
    if (suit_isomorphism_) {
      // One of the two identical cards must be left in the deck.
      SPIEL_CHECK_TRUE(deck_[move * 2] != kInvalidCard ||
                       deck_[move * 2 + 1] != kInvalidCard);
    } else {
      SPIEL_CHECK_NE(deck_[move], kInvalidCard);
    }

    if (private_cards_dealt_ < num_players_) {
      SetPrivate(private_cards_dealt_, move);
    } else {
      // Round 2: A single public card.
      if (suit_isomorphism_) {
        public_card_ = move;
        if (deck_[move * 2] != kInvalidCard) {
          deck_[move * 2] = kInvalidCard;
        } else if (deck_[move * 2 + 1] != kInvalidCard) {
          deck_[move * 2 + 1] = kInvalidCard;
        } else {
          SpielFatalError("Suit isomorphism error.");
        }
        deck_size_--;
      } else {
        public_card_ = deck_[move];
        deck_[move] = kInvalidCard;
        deck_size_--;
      }

      // We have finished the public card, let's bet!
      cur_player_ = NextPlayer();
    }
  } else {
    // Player node.
    if (action_mapping_) {
      // Map otherwise illegal actions to kCall.
      if (move == ActionType::kFold) {
        if (stakes_ <= ante_[cur_player_]) {
          move = ActionType::kCall;
        }
      } else if (move == ActionType::kRaise) {
        if (num_raises_ >= 2) {
          move = ActionType::kCall;
        }
      }
    }

    if (move == ActionType::kFold) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);
      SequenceAppendMove(ActionType::kFold);

      // Player is now out.
      folded_[cur_player_] = true;
      remaining_players_--;

      if (IsTerminal()) {
        ResolveWinner();
      } else if (ReadyForNextRound()) {
        NewRound();
      } else {
        cur_player_ = NextPlayer();
      }
    } else if (move == ActionType::kCall) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);

      // Current player puts in an amount of money equal to the current level
      // (stakes) minus what they have contributed to level their contribution
      // off. Note: this action also acts as a 'check' where the stakes are
      // equal to each player's ante.
      SPIEL_CHECK_GE(stakes_, ante_[cur_player_]);
      int amount = stakes_ - ante_[cur_player_];
      Ante(cur_player_, amount);
      num_calls_++;
      SequenceAppendMove(ActionType::kCall);

      if (IsTerminal()) {
        ResolveWinner();
      } else if (ReadyForNextRound()) {
        NewRound();
      } else {
        cur_player_ = NextPlayer();
      }
    } else if (move == ActionType::kRaise) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);

      // This player matches the current stakes and then brings the stakes up.
      SPIEL_CHECK_LT(num_raises_, kMaxRaises);
      int call_amount = stakes_ - ante_[cur_player_];

      // First, match the current stakes if necessary
      SPIEL_CHECK_GE(call_amount, 0);
      if (call_amount > 0) {
        Ante(cur_player_, call_amount);
      }

      // Now, raise the stakes.
      int raise_amount = (round_ == 1 ? kFirstRaiseAmount : kSecondRaiseAmount);
      stakes_ += raise_amount;
      Ante(cur_player_, raise_amount);
      num_raises_++;
      num_calls_ = 0;
      SequenceAppendMove(ActionType::kRaise);

      if (IsTerminal()) {
        ResolveWinner();
      } else {
        cur_player_ = NextPlayer();
      }
    } else {
      SpielFatalError(absl::StrCat("Move ", move, " is invalid. ChanceNode?",
                                   IsChanceNode()));
    }
  }
}

std::vector<Action> LeducState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> movelist;
  if (IsChanceNode()) {
    if (suit_isomorphism_) {
      // Consecutive cards are identical under suit isomorphism.
      for (int card = 0; card < deck_.size() / 2; card++) {
        if (deck_[card * 2] != kInvalidCard ||
            deck_[card * 2 + 1] != kInvalidCard) {
          movelist.push_back(card);
        }
      }
    } else {
      for (int card = 0; card < deck_.size(); card++) {
        if (deck_[card] != kInvalidCard) movelist.push_back(card);
      }
    }
    return movelist;
  }

  if (action_mapping_) {
    // All actions are regarded as legal
    movelist.push_back(ActionType::kFold);
    movelist.push_back(ActionType::kCall);
    movelist.push_back(ActionType::kRaise);
    return movelist;
  }

  // Can't just randomly fold; only allow fold when under pressure.
  if (stakes_ > ante_[cur_player_]) {
    movelist.push_back(ActionType::kFold);
  }

  // Can always call/check
  movelist.push_back(ActionType::kCall);

  if (num_raises_ < 2) {
    movelist.push_back(ActionType::kRaise);
  }

  return movelist;
}

std::string LeducState::ActionToString(Player player, Action move) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Chance outcome:", move);
  else if (move == ActionType::kFold)
    return "Fold";
  else if (move == ActionType::kCall)
    return "Call";
  else if (move == ActionType::kRaise)
    return "Raise";
  else
    SpielFatalError(
        absl::StrCat("Error in LeducState::ActionToString(). Available actions "
                     "are 0, 1, 2, not ",
                     move));
}

std::string LeducState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "Round: ", round_, "\nPlayer: ", cur_player_,
                  "\nPot: ", pot_, "\nMoney (p1 p2 ...):");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", money_[p]);
  }
  absl::StrAppend(&result, "\nCards (public p1 p2 ...): ", public_card_, " ");
  for (Player player_index = 0; player_index < num_players_; player_index++) {
    absl::StrAppend(&result, private_cards_[player_index], " ");
  }

  absl::StrAppend(&result, "\nRound 1 sequence: ");
  absl::StrAppend(&result, absl::StrJoin(round1_sequence_, " "));

  absl::StrAppend(&result, "\nRound 2 sequence: ");
  absl::StrAppend(&result, absl::StrJoin(round2_sequence_, " "));

  absl::StrAppend(&result, "\n");

  return result;
}

bool LeducState::IsTerminal() const {
  return remaining_players_ == 1 || (round_ == 2 && ReadyForNextRound());
}

std::vector<double> LeducState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  for (auto player = Player{0}; player < num_players_; ++player) {
    // Money vs money at start.
    returns[player] = money_[player] - kStartingMoney;
  }

  return returns;
}

// Information state is card then bets.
std::string LeducState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  // TODO(author1): Fix typos in InformationState string.
  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s[Private: %i]][Round1]: "
      "%s[Public: %i]\nRound 2 sequence: %s",
      round_, cur_player_, pot_, absl::StrJoin(money_, " "),
      private_cards_[player], absl::StrJoin(round1_sequence_, " "),
      public_card_, absl::StrJoin(round2_sequence_, " "));
}

// Observation is card then contribution of each players to the pot.
std::string LeducState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string result;

  absl::StrAppend(&result, "[Round ", round_, "][Player: ", cur_player_,
                  "][Pot: ", pot_, "][Money:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", money_[p]);
  }
  // Add the player's private cards
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "[Private: ", private_cards_[player], "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", ante_[p]);
  }
  absl::StrAppend(&result, "]");

  // Add the public card
  if (public_card_ != kInvalidCard) {
    absl::StrAppend(&result, "[Public: ", public_card_, "]");
  }

  return result;
}

void LeducState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my card: deck_.size() bits
  //   public card: deck_.size() bits
  //   first round sequence: (max round seq length)*2 bits
  //   second round sequence: (max round seq length)*2 bits

  int offset = 0;

  // Mark who I am.
  values[player] = 1;
  offset += num_players_;

  if (private_cards_[player] >= 0) {
    values[offset + private_cards_[player]] = 1;
  }
  if (suit_isomorphism_) {
    offset += deck_.size() / 2;
  } else {
    offset += deck_.size();
  }

  if (public_card_ >= 0) {
    values[offset + public_card_] = 1;
  }
  if (suit_isomorphism_) {
    offset += deck_.size() / 2;
  } else {
    offset += deck_.size();
  }

  for (int r = 1; r <= 2; r++) {
    const std::vector<int>& round_sequence =
        (r == 1 ? round1_sequence_ : round2_sequence_);

    for (int i = 0; i < round_sequence.size(); ++i) {
      SPIEL_CHECK_LT(offset + i + 1, values.size());
      if (round_sequence[i] == ActionType::kCall) {
        // Encode call as 10.
        values[offset + (2 * i)] = 1;
        values[offset + (2 * i) + 1] = 0;
      } else if (round_sequence[i] == ActionType::kRaise) {
        // Encode raise as 01.
        values[offset + (2 * i)] = 0;
        values[offset + (2 * i) + 1] = 1;
      } else {
        // Encode fold as 00.
        values[offset + (2 * i)] = 0;
        values[offset + (2 * i) + 1] = 0;
      }
    }

    // Move offset up to the next round: 2 bits per move.
    offset += game_->MaxGameLength();
  }
}

void LeducState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my card: deck_.size() bits
  //   public card: deck_.size() bits
  //   the contribution of each player to the pot. num_players integers.

  int offset = 0;

  // Mark who I am.
  values[player] = 1;
  offset += num_players_;

  if (private_cards_[player] >= 0) {
    values[offset + private_cards_[player]] = 1;
  }
  if (suit_isomorphism_) {
    offset += deck_.size() / 2;
  } else {
    offset += deck_.size();
  }

  if (public_card_ >= 0) {
    values[offset + public_card_] = 1;
  }
  if (suit_isomorphism_) {
    offset += deck_.size() / 2;
  } else {
    offset += deck_.size();
  }

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < num_players_; p++) {
    values[offset + p] = ante_[p];
  }
}

std::unique_ptr<State> LeducState::Clone() const {
  return std::unique_ptr<State>(new LeducState(*this));
}

std::vector<std::pair<Action, double>> LeducState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  if (suit_isomorphism_) {
    const double p = 1.0 / deck_size_;
    // Consecutive cards in deck are viewed identically.
    for (int card = 0; card < deck_.size() / 2; card++) {
      if (deck_[card * 2] != kInvalidCard &&
          deck_[card * 2 + 1] != kInvalidCard) {
        outcomes.push_back({card, p*2});
      } else if (deck_[card * 2] != kInvalidCard ||
                 deck_[card * 2 + 1] != kInvalidCard) {
        outcomes.push_back({card, p});
      }
    }
    return outcomes;
  }

  const double p = 1.0 / deck_size_;
  for (int card = 0; card < deck_.size(); card++) {
    // This card is still in the deck, prob is 1/decksize.
    if (deck_[card] != kInvalidCard) outcomes.push_back({card, p});
  }
  return outcomes;
}

int LeducState::NextPlayer() const {
  // If we are on a chance node, it is the first player to play
  int current_real_player;
  if (cur_player_ == kChancePlayerId) {
    current_real_player = -1;
  } else {
    current_real_player = cur_player_;
  }
  // Go to the next player who's still in.
  for (int i = 1; i < num_players_; ++i) {
    Player player = (current_real_player + i) % num_players_;

    SPIEL_CHECK_TRUE(player >= 0);
    SPIEL_CHECK_TRUE(player < num_players_);
    if (!folded_[player]) {
      return player;
    }
  }

  SpielFatalError("Error in LeducState::NextPlayer(), should not get here.");
}

int LeducState::RankHand(Player player) const {
  int hand[] = {public_card_, private_cards_[player]};
  // Put the lower card in slot 0, the higher in slot 1.
  if (hand[0] > hand[1]) {
    std::swap(hand[0], hand[1]);
  }

  if (suit_isomorphism_) {
    int num_cards = deck_.size() / 2;
    if (hand[0] == hand[1]) {
      // Pair! Offset by deck_size_^2 to put higher than every singles combo.
      return (num_cards * num_cards + hand[0]);
    } else {
      // Otherwise card value dominates. Suit isomorphism has already removed
      // the distinction between suits, so we can compare the ranks directly.
      // This could lead to ties/draws and/or multiple winners.
      return hand[1] * num_cards + hand[0];
    }
  }

  // E.g. rank for two players:
  // 0 J1, 1 J2, 2 Q1, 3 Q2, 4 K1, 5 K2.
  int num_cards = deck_.size();

  if (hand[0] % 2 == 0 && hand[1] == hand[0] + 1) {
    // Pair! Offset by deck_size_^2 to put higher than every singles combo.
    return (num_cards * num_cards + hand[0]);
  } else {
    // Otherwise card value dominates. No high/low suit: only two suits, and
    // given ordering above, dividing by gets the value (integer division
    // intended.) This could lead to ties/draws and/or multiple winners.
    return (hand[1] / 2) * num_cards + (hand[0] / 2);
  }
}

void LeducState::ResolveWinner() {
  num_winners_ = kInvalidPlayer;

  if (remaining_players_ == 1) {
    // Only one left in? They get the pot!
    for (Player player_index = 0; player_index < num_players_; player_index++) {
      if (!folded_[player_index]) {
        num_winners_ = 1;
        winner_[player_index] = true;
        money_[player_index] += pot_;
        pot_ = 0;
        return;
      }
    }

  } else {
    // Otherwise, showdown!
    // Find the best hand among those still in.
    SPIEL_CHECK_NE(public_card_, kInvalidCard);
    int best_hand_rank = -1;
    num_winners_ = 0;
    std::fill(winner_.begin(), winner_.end(), false);

    for (Player player_index = 0; player_index < num_players_; player_index++) {
      if (!folded_[player_index]) {
        int rank = RankHand(player_index);
        if (rank > best_hand_rank) {
          // Beat the current best hand! Clear the winners list, then add.
          best_hand_rank = rank;
          std::fill(winner_.begin(), winner_.end(), false);
          winner_[player_index] = true;
          num_winners_ = 1;
        } else if (rank == best_hand_rank) {
          // Tied with best hand rank, so this player is a winner as well.
          winner_[player_index] = true;
          num_winners_++;
        }
      }
    }

    // Split the pot among the winners (possibly only one).
    SPIEL_CHECK_TRUE(1 <= num_winners_ && num_winners_ <= num_players_);
    for (Player player_index = 0; player_index < num_players_; player_index++) {
      if (winner_[player_index]) {
        // Give this player their share.
        money_[player_index] += static_cast<double>(pot_) / num_winners_;
      }
    }
    pot_ = 0;
  }
}

bool LeducState::ReadyForNextRound() const {
  return ((num_raises_ == 0 && num_calls_ == remaining_players_) ||
          (num_raises_ > 0 && num_calls_ == (remaining_players_ - 1)));
}

void LeducState::NewRound() {
  SPIEL_CHECK_EQ(round_, 1);
  round_++;
  num_raises_ = 0;
  num_calls_ = 0;
  cur_player_ = kChancePlayerId;  // Public card.
}

void LeducState::SequenceAppendMove(int move) {
  if (round_ == 1) {
    round1_sequence_.push_back(move);
  } else {
    SPIEL_CHECK_EQ(round_, 2);
    round2_sequence_.push_back(move);
  }
}

void LeducState::Ante(Player player, int amount) {
  pot_ += amount;
  ante_[player] += amount;
  money_[player] -= amount;
}

std::vector<int> LeducState::padded_betting_sequence() const {
  std::vector<int> history = round1_sequence_;

  // We pad the history to the end of the first round with kPaddingAction.
  history.resize(game_->MaxGameLength() / 2, kInvalidAction);

  // We insert the actions that happened in the second round, and fill to
  // MaxGameLength.
  history.insert(history.end(), round2_sequence_.begin(),
                 round2_sequence_.end());
  history.resize(game_->MaxGameLength(), kInvalidAction);
  return history;
}

void LeducState::SetPrivate(Player player, Action move) {
  // Round 1. `move` refers to the card value to deal to the current
  // underlying player (given by `private_cards_dealt_`).
  if (suit_isomorphism_) {
    // Consecutive cards are identical under suit isomorphism.
    private_cards_[player] = move;
    if (deck_[move * 2] != kInvalidCard) {
      deck_[move * 2] = kInvalidCard;
    } else if (deck_[move * 2 + 1] != kInvalidCard) {
      deck_[move * 2 + 1] = kInvalidCard;
    } else {
      SpielFatalError("Suit isomorphism error.");
    }
  } else {
    private_cards_[player] = deck_[move];
    deck_[move] = kInvalidCard;
  }
  --deck_size_;
  ++private_cards_dealt_;

  // When all private cards are dealt, move to player 0.
  if (private_cards_dealt_ == num_players_) cur_player_ = 0;
}

std::unique_ptr<State> LeducState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();

  // First, deal out cards:
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < GetGame()->NumPlayers(); ++p) {
    if (p == player_id) {
      clone->ApplyAction(history_.at(p).action);
    } else {
      Action chosen_action = player_chance;
      while (chosen_action == player_chance || chosen_action == public_card_) {
        chosen_action = SampleAction(clone->ChanceOutcomes(), rng()).first;
      }
      clone->ApplyAction(chosen_action);
    }
  }
  for (int action : round1_sequence_) clone->ApplyAction(action);
  if (public_card_ != kInvalidCard) {
    clone->ApplyAction(public_card_);
    for (int action : round2_sequence_) clone->ApplyAction(action);
  }
  return clone;
}

LeducGame::LeducGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      total_cards_((num_players_ + 1) * kNumSuits),
      action_mapping_(ParameterValue<bool>("action_mapping")),
      suit_isomorphism_(ParameterValue<bool>("suit_isomorphism")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
}

std::unique_ptr<State> LeducGame::NewInitialState() const {
  return absl::make_unique<LeducState>(shared_from_this(),
                                       /*action_mapping=*/action_mapping_,
                                       /*suit_isomorphism=*/suit_isomorphism_);
}

int LeducGame::MaxChanceOutcomes() const {
  if (suit_isomorphism_) {
    return total_cards_ / 2;
  } else {
    return total_cards_;
  }
}

std::vector<int> LeducGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  if (suit_isomorphism_) {
    return {(num_players_) + (total_cards_) + (MaxGameLength() * 2)};
  } else {
    return {(num_players_) + (total_cards_ * 2) + (MaxGameLength() * 2)};
  }
}

std::vector<int> LeducGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot
  if (suit_isomorphism_) {
    return {(num_players_) + (total_cards_) + (num_players_)};
  } else {
    return {(num_players_) + (total_cards_ * 2) + (num_players_)};
  }
}

double LeducGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.
  return (num_players_ - 1) * (kTotalRaisesPerRound * kFirstRaiseAmount +
                               kTotalRaisesPerRound * kSecondRaiseAmount + 1);
}

double LeducGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  return -1 * (kTotalRaisesPerRound * kFirstRaiseAmount +
               kTotalRaisesPerRound * kSecondRaiseAmount + 1);
}

}  // namespace leduc_poker
}  // namespace open_spiel
