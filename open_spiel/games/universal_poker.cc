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

#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace {

std::string BettingAbstractionToString(const BettingAbstraction &betting) {
  switch (betting) {
    case BettingAbstraction::kFC: {
      return "BettingAbstration: FC";
      break;
    }
    case BettingAbstraction::kFCHPA: {
      return "BettingAbstration: FCPHA";
      break;
    }
    case BettingAbstraction::kFCPA: {
      return "BettingAbstration: FCPA";
      break;
    }
    case BettingAbstraction::kFULLGAME: {
      return "BettingAbstraction: FULLGAME";
      break;
    }
    default:
      SpielFatalError("Unknown betting abstraction.");
      break;
  }
}

// Does not support chance actions.
// TODO(author1): Remove all of the many varieties of action types and
// switch to use a single enum, preferably project_acpc_server::ActionType.
acpc_cpp::ACPCState::ACPCActionType UniversalPokerActionTypeToACPCActionType(
    StateActionType type) {
  if (type == StateActionType::ACTION_DEAL) {
    SpielFatalError("ACPC does not support deal action types.");
  }
  if (type == StateActionType::ACTION_FOLD) {
    return acpc_cpp::ACPCState::ACPC_FOLD;
  }
  if (type == StateActionType::ACTION_CHECK_CALL) {
    return acpc_cpp::ACPCState::ACPC_CALL;
  }
  if (type == StateActionType::ACTION_BET ||
      type == StateActionType::ACTION_ALL_IN) {
    return acpc_cpp::ACPCState::ACPC_RAISE;
  }
  SpielFatalError(absl::StrCat("Action not found: ", type));

  // Should never be called.
  return acpc_cpp::ACPCState::ACPC_INVALID;
}

}  // namespace

const GameType kGameType{
    /*short_name=*/"universal_poker",
    /*long_name=*/"Universal Poker",
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

    {// The ACPC code uses a specific configuration file to describe the game.
     // The following has been copied from ACPC documentation:
     //
     // Empty lines or lines with '#' as the very first character will be
     // ignored
     //
     // The Game definitions should start with "gamedef" and end with
     // "end gamedef" and can have the fields documented bellow (case is
     // ignored)
     //
     // If you are creating your own game definitions, please note that game.h
     // defines some constants for maximums in games (e.g., number of rounds).
     // These may need to be changed for games outside of the what is being run
     // for the Annual Computer Poker Competition.

     // The ACPC gamedef string.  When present, it will take precedence over
     // everything and no other argument should be provided.
     {"gamedef", GameParameter(std::string(""))},
     // Instead of a single gamedef, specifying each line is also possible.
     // The documentation is adapted from project_acpc_server/game.cc.
     //
     // Number of Players (up to 10)
     {"numPlayers", GameParameter(2)},
     // Betting Type "limit" "nolimit"
     {"betting", GameParameter(std::string("nolimit"))},
     // The stack size for each player at the start of each hand (for
     // no-limit). It will be ignored on "limit".
     // TODO(author2): It's unclear what happens on limit. It defaults to
     // INT32_MAX for all players when not provided.
     {"stack", GameParameter(std::string("1200 1200"))},
     // The size of the blinds for each player (relative to the dealer)
     {"blind", GameParameter(std::string("100 100"))},
     // The size of raises on each round (for limit games only) as numrounds
     // integers. It will be ignored for nolimite games.
     {"raiseSize", GameParameter(std::string("100 100"))},
     // Number of betting rounds per hand of the game
     {"numRounds", GameParameter(2)},
     // The player that acts first (relative to the dealer) on each round
     {"firstPlayer", GameParameter(std::string("1 1"))},
     // maxraises - the maximum number of raises on each round. If not
     // specified, it will default to UINT8_MAX.
     {"maxRaises", GameParameter(std::string(""))},
     // The number of different suits in the deck
     {"numSuits", GameParameter(4)},
     // The number of different ranks in the deck
     {"numRanks", GameParameter(6)},
     // The number of private cards to  deal to each player
     {"numHoleCards", GameParameter(1)},
     // The number of cards revealed on each round
     {"numBoardCards", GameParameter(std::string("0 1"))},
     // Specify which actions are available to the player, in both limit and
     // nolimit games. Available options are: "fc" for fold and check/call.
     // "fcpa" for fold, check/call, bet pot and all in (default).
     // Use "fullgame" for the unabstracted game.
     {"bettingAbstraction", GameParameter(std::string("fcpa"))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return absl::make_unique<UniversalPokerGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Returns how many actions are available at a choice node (3 when limit
// and 4 for no limit).
// TODO(author2): Is that a bug? There are 5 actions? Is no limit means
// "bet bot" is added? or should "all in" be also added?
inline uint32_t GetMaxBettingActions(const acpc_cpp::ACPCGame &acpc_game) {
  return acpc_game.IsLimitGame() ? 3 : 4;
}

// namespace universal_poker
UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game)
    : State(game),
      acpc_game_(
          static_cast<const UniversalPokerGame *>(game.get())->GetACPCGame()),
      acpc_state_(acpc_game_),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
            /*num_ranks=*/acpc_game_->NumRanksDeck()),
      cur_player_(kChancePlayerId),
      possibleActions_(ACTION_DEAL),
      betting_abstraction_(static_cast<const UniversalPokerGame *>(game.get())
                               ->betting_abstraction()) {}

std::string UniversalPokerState::ToString() const {
  std::string str =
      absl::StrCat(BettingAbstractionToString(betting_abstraction_), "\n");
  for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
    absl::StrAppend(&str, "P", p, " Cards: ", HoleCards(p).ToString(), "\n");
  }
  absl::StrAppend(&str, "BoardCards ", BoardCards().ToString(), "\n");

  if (IsChanceNode()) {
    absl::StrAppend(&str, "PossibleCardsToDeal ", deck_.ToString(), "\n");
  }
  if (IsTerminal()) {
    for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
      absl::StrAppend(&str, "P", p, " Reward: ", GetTotalReward(p), "\n");
    }
  }
  absl::StrAppend(&str, "Node type?: ");
  if (IsChanceNode()) {
    absl::StrAppend(&str, "Chance node\n");
  } else if (IsTerminal()) {
    absl::StrAppend(&str, "Terminal Node!\n");
  } else {
    absl::StrAppend(&str, "Player node for player ", cur_player_, "\n");
  }

  if (betting_abstraction_ == BettingAbstraction::kFC ||
      betting_abstraction_ == BettingAbstraction::kFCPA) {
    absl::StrAppend(&str, "PossibleActions (", GetPossibleActionCount(),
                    "): [");
    for (StateActionType action : ALL_ACTIONS) {
      if (action & possibleActions_) {
        if (action == ACTION_ALL_IN) absl::StrAppend(&str, " ACTION_ALL_IN ");
        if (action == ACTION_BET) absl::StrAppend(&str, " ACTION_BET ");
        if (action == ACTION_CHECK_CALL) {
          absl::StrAppend(&str, " ACTION_CHECK_CALL ");
        }
        if (action == ACTION_FOLD) absl::StrAppend(&str, " ACTION_FOLD ");
        if (action == ACTION_DEAL) absl::StrAppend(&str, " ACTION_DEAL ");
      }
    }
  }
  absl::StrAppend(&str, "]", "\nRound: ", acpc_state_.GetRound(),
                  "\nACPC State: ", acpc_state_.ToString(),
                  "\nAction Sequence: ", actionSequence_);
  return str;
}

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  std::string move_str;
  if (IsChanceNode()) {
    move_str = absl::StrCat("Deal(", move, ")");
  } else if (static_cast<ActionType>(move) == ActionType::kFold) {
    move_str = "Fold";
  } else if (static_cast<ActionType>(move) == ActionType::kCall) {
    move_str = "Call";
  } else if (static_cast<ActionType>(move) == ActionType::kHalfPot) {
    move_str = "HalfPot";
  } else if (betting_abstraction_ == BettingAbstraction::kFULLGAME) {
    SPIEL_CHECK_GE(move, 2);
    move_str = absl::StrCat("Bet", move);
  } else if (static_cast<ActionType>(move) == ActionType::kBet) {
    move_str = "Bet";
  } else if (static_cast<ActionType>(move) == ActionType::kAllIn) {
    SPIEL_CHECK_EQ(betting_abstraction_, BettingAbstraction::kFCPA);
    move_str = "AllIn";
  } else {
    SpielFatalError(absl::StrCat("Unknown action: ", move));
  }
  return absl::StrCat("player=", player, " move=", move_str);
}

bool UniversalPokerState::IsTerminal() const {
  bool finished = cur_player_ == kTerminalPlayerId;
  assert(acpc_state_.IsFinished() || !finished);
  return finished;
}

bool UniversalPokerState::IsChanceNode() const {
  return cur_player_ == kChancePlayerId;
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (IsChanceNode()) {
    return kChancePlayerId;
  }

  return Player(acpc_state_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (Player player = 0; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = GetTotalReward(player);
  }

  return returns;
}

void UniversalPokerState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my cards: Initial deck size bits (1 means you have the card), i.e.
  //             MaxChanceOutcomes() = NumSuits * NumRanks
  //   public cards: Same as above, but for the public cards.
  //   NumRounds() round sequence: (max round seq length)*2 bits
  int offset = 0;

  // Mark who I am.
  values[player] = 1;
  offset += NumPlayers();

  const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                 acpc_game_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = full_deck.ToCardArray();
  logic::CardSet holeCards = HoleCards(player);
  logic::CardSet boardCards = BoardCards();

  // TODO(author2): it should be way more efficient to iterate over the cards
  // of the player, rather than iterating over all the cards.
  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    values[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  // Public cards
  for (int i = 0; i < full_deck.NumCards(); ++i) {
    values[i + offset] = boardCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  const std::string actionSeq = GetActionSequence();
  const int length = actionSeq.length();
  SPIEL_CHECK_LT(length, game_->MaxGameLength());

  for (int i = 0; i < length; ++i) {
    SPIEL_CHECK_LT(offset + i + 1, values.size());
    if (actionSeq[i] == 'c') {
      // Encode call as 10.
      values[offset + (2 * i)] = 1;
      values[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'p') {
      // Encode raise as 01.
      values[offset + (2 * i)] = 0;
      values[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'a') {
      // Encode raise as 01.
      values[offset + (2 * i)] = 1;
      values[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'f') {
      // Encode fold as 00.
      // TODO(author2): Should this be 11?
      values[offset + (2 * i)] = 0;
      values[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'd') {
      values[offset + (2 * i)] = 0;
      values[offset + (2 * i) + 1] = 0;
    } else {
      SPIEL_CHECK_EQ(actionSeq[i], 'd');
    }
  }

  // Move offset up to the next round: 2 bits per move.
  offset += game_->MaxGameLength() * 2;
  SPIEL_CHECK_EQ(offset, game_->InformationStateTensorShape()[0]);
}

void UniversalPokerState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my cards: Initial deck size bits (1 means you have the card), i.e.
  //             MaxChanceOutcomes() = NumSuits * NumRanks
  //   public cards: Same as above, but for the public cards.
  //   the contribution of each player to the pot. num_players integers.
  int offset = 0;

  // Mark who I am.
  values[player] = 1;
  offset += NumPlayers();

  const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                 acpc_game_->NumRanksDeck());
  const std::vector<uint8_t> all_cards = full_deck.ToCardArray();
  logic::CardSet holeCards = HoleCards(player);
  logic::CardSet boardCards = BoardCards();

  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    values[i + offset] = holeCards.ContainsCards(all_cards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    values[i + offset] = boardCards.ContainsCards(all_cards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < NumPlayers(); p++) {
    values[offset + p] = acpc_state_.Ante(p);
  }
  offset += NumPlayers();
  SPIEL_CHECK_EQ(offset, game_->ObservationTensorShape()[0]);
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  const uint32_t pot = acpc_state_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - acpc_state_.NumFolded());
  std::string money;
  money.reserve(acpc_game_->GetNbPlayers() * 2);
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    if (p != Player{0}) absl::StrAppend(&money, " ");
    absl::StrAppend(&money, acpc_state_.Money(p));
  }
  std::string sequences;
  sequences.reserve(acpc_state_.GetRound() * 2);
  for (auto r = 0; r <= acpc_state_.GetRound(); r++) {
    if (r != 0) absl::StrAppend(&sequences, "|");
    absl::StrAppend(&sequences, acpc_state_.BettingSequence(r));
  }

  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s][Private: %s][Public: "
      "%s][Sequences: %s]",
      acpc_state_.GetRound(), CurrentPlayer(), pot, money,
      HoleCards(player).ToString(), BoardCards().ToString(), sequences);
}

std::string UniversalPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  std::string result;

  const uint32_t pot = acpc_state_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - acpc_state_.NumFolded());
  absl::StrAppend(&result, "[Round ", acpc_state_.GetRound(),
                  "][Player: ", CurrentPlayer(), "][Pot: ", pot, "][Money:");
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    absl::StrAppend(&result, " ", acpc_state_.Money(p));
  }
  // Add the player's private cards
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "[Private: ", HoleCards(player).ToString(), "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", acpc_state_.Ante(p));
  }
  absl::StrAppend(&result, "]");

  return result;
}

std::unique_ptr<State> UniversalPokerState::Clone() const {
  return absl::make_unique<UniversalPokerState>(*this);
}

std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  auto available_cards = LegalActions();
  const int num_cards = available_cards.size();
  const double p = 1.0 / num_cards;

  // We need to convert std::vector<uint8_t> into std::vector<Action>.
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_cards);
  for (const auto &card : available_cards) {
    outcomes.push_back({Action{card}, p});
  }
  return outcomes;
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  if (IsChanceNode()) {
    const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                   acpc_game_->NumRanksDeck());
    const std::vector<uint8_t> all_cards = full_deck.ToCardArray();
    std::vector<Action> actions;
    actions.reserve(deck_.NumCards());
    for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
      if (deck_.ContainsCards(all_cards[i])) actions.push_back(i);
    }
    return actions;
  }

  std::vector<Action> legal_actions;

  if (betting_abstraction_ != BettingAbstraction::kFULLGAME) {
    if (ACTION_FOLD & possibleActions_) legal_actions.push_back(kFold);
    if (ACTION_CHECK_CALL & possibleActions_) legal_actions.push_back(kCall);
    if (ACTION_BET & possibleActions_) legal_actions.push_back(kBet);
    if (ACTION_ALL_IN & possibleActions_) legal_actions.push_back(kAllIn);

    // For legacy reasons, kHalfPot is the biggest action (in terms of the
    // action representation).
    if (betting_abstraction_ == kFCHPA) {
      legal_actions.push_back(kHalfPot);
    }
    return legal_actions;
  } else {
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
      legal_actions.push_back(kFold);
    }
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
      legal_actions.push_back(kCall);
    }
    int32_t min_bet_size = 0;
    int32_t max_bet_size = 0;
    if (acpc_state_.RaiseIsValid(&min_bet_size, &max_bet_size)) {
      const int original_size = legal_actions.size();
      legal_actions.resize(original_size + max_bet_size - min_bet_size + 1);
      std::iota(legal_actions.begin() + original_size, legal_actions.end(),
                min_bet_size);
    }
  }
  return legal_actions;
}

int UniversalPokerState::PotSize(double multiple) const {
  const project_acpc_server::State &state = acpc_state_.raw_state();
  const project_acpc_server::Game &game = acpc_state_.game()->Game();
  const int pot_size = absl::c_accumulate(
      absl::Span<const int32_t>(state.spent, game.numPlayers), 0);
  const int amount_to_call =
      state.maxSpent -
      state.spent[project_acpc_server::currentPlayer(&game, &state)];
  const int pot_after_call = amount_to_call + pot_size;
  return std::round(state.maxSpent + multiple * pot_after_call);
}

int UniversalPokerState::AllInSize() const {
  int32_t unused_min_bet_size;
  int32_t all_in_size;
  acpc_state_.RaiseIsValid(&unused_min_bet_size, &all_in_size);
  return all_in_size;
}

// We first deal the cards to each player, dealing all the cards to the first
// player first, then the second player, until all players have their private
// cards.
void UniversalPokerState::DoApplyAction(Action action_id) {
  if (IsChanceNode()) {
    // In chance nodes, the action_id is an index into the full deck.
    uint8_t card =
        logic::CardSet(acpc_game_->NumSuitsDeck(), acpc_game_->NumRanksDeck())
            .ToCardArray()[action_id];
    deck_.RemoveCard(card);
    actionSequence_ += 'd';

    // Check where to add this card
    if (hole_cards_dealt_ <
        acpc_game_->GetNbPlayers() * acpc_game_->GetNbHoleCardsRequired()) {
      AddHoleCard(card);
      _CalculateActionsAndNodeType();
      return;
    }

    if (board_cards_dealt_ <
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
      AddBoardCard(card);
      _CalculateActionsAndNodeType();
      return;
    }
  } else {
    int action_int = static_cast<int>(action_id);
    if (action_int == kFold) {
      ApplyChoiceAction(ACTION_FOLD, 0);
      return;
    }
    if (action_int == kCall) {
      ApplyChoiceAction(ACTION_CHECK_CALL, 0);
      return;
    }
    if (betting_abstraction_ == BettingAbstraction::kFC) {
      SpielFatalError(absl::StrCat(
          "Tried to apply action that was not fold or call. Action: ",
          State::ActionToString(action_id)));
    }
    if (betting_abstraction_ != BettingAbstraction::kFULLGAME) {
      if (action_int == kHalfPot) {
        ApplyChoiceAction(ACTION_BET, PotSize(0.5));
        return;
      }
      if (action_int == kBet && acpc_game_->IsLimitGame()) {
        // In a limit game, the bet size is fixed, so the ACPC code expects size
        // to be 0.
        ApplyChoiceAction(ACTION_BET, /*size=*/0);
        return;
      }
      if (action_int == kBet && !acpc_game_->IsLimitGame()) {
        ApplyChoiceAction(ACTION_BET, PotSize());
        return;
      }
      if (action_int == kAllIn) {
        ApplyChoiceAction(ACTION_ALL_IN, AllInSize());
        return;
      }
    }
    if (betting_abstraction_ != BettingAbstraction::kFULLGAME) {
      SpielFatalError(absl::StrCat(
          "Tried to apply action that was not allowed by the betting "
          "abstraction. Action: ",
          State::ActionToString(action_id)));
    }
    if (action_int >= static_cast<int>(kBet) &&
        action_int <= NumDistinctActions()) {
      ApplyChoiceAction(ACTION_BET, action_int);
      return;
    }
    SpielFatalError(absl::StrFormat("Action not recognized: %i", action_id));
  }
}

double UniversalPokerState::GetTotalReward(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  return acpc_state_.ValueOfState(player);
}

std::unique_ptr<HistoryDistribution>
UniversalPokerState::GetHistoriesConsistentWithInfostate(int player_id) const {
  // This is only implemented for 2 players.
  if (acpc_game_->GetNbPlayers() != 2) return {};

  logic::CardSet is_cards;
  logic::CardSet our_cards = HoleCards(player_id);
  for (uint8_t card : our_cards.ToCardArray()) is_cards.AddCard(card);
  for (uint8_t card : BoardCards().ToCardArray()) is_cards.AddCard(card);
  logic::CardSet fresh_deck(/*num_suits=*/acpc_game_->NumSuitsDeck(),
                            /*num_ranks=*/acpc_game_->NumRanksDeck());
  for (uint8_t card : is_cards.ToCardArray()) fresh_deck.RemoveCard(card);
  auto dist = absl::make_unique<HistoryDistribution>();

  // We only consider half the possible hands as we only look at each pair of
  // hands once, i.e. order does not matter.
  const int num_hands =
      0.5 * fresh_deck.NumCards() * (fresh_deck.NumCards() - 1);
  dist->first.reserve(num_hands);
  for (uint8_t hole_card1 : fresh_deck.ToCardArray()) {
    logic::CardSet subset_deck = fresh_deck;
    subset_deck.RemoveCard(hole_card1);
    for (uint8_t hole_card2 : subset_deck.ToCardArray()) {
      if (hole_card1 < hole_card2) continue;
      std::unique_ptr<State> root = game_->NewInitialState();
      if (player_id == 0) {
        for (uint8_t card : our_cards.ToCardArray()) root->ApplyAction(card);
        root->ApplyAction(hole_card1);
        root->ApplyAction(hole_card2);
      } else if (player_id == 1) {
        root->ApplyAction(hole_card1);
        root->ApplyAction(hole_card2);
        for (uint8_t card : our_cards.ToCardArray()) root->ApplyAction(card);
      }
      SPIEL_CHECK_FALSE(root->IsChanceNode());
      dist->first.push_back(std::move(root));
    }
  }
  SPIEL_DCHECK_EQ(dist->first.size(), num_hands);
  const double divisor = 1. / static_cast<double>(dist->first.size());
  dist->second.assign(dist->first.size(), divisor);
  return dist;
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      acpc_game_(gameDesc_) {
  max_game_length_ = MaxGameLength();
  SPIEL_CHECK_TRUE(max_game_length_.has_value());
  std::string betting_abstraction =
      ParameterValue<std::string>("bettingAbstraction");
  if (betting_abstraction == "fc") {
    betting_abstraction_ = BettingAbstraction::kFC;
  } else if (betting_abstraction == "fcpa") {
    betting_abstraction_ = BettingAbstraction::kFCPA;
  } else if (betting_abstraction == "fchpa") {
    betting_abstraction_ = BettingAbstraction::kFCHPA;
  } else if (betting_abstraction == "fullgame") {
    betting_abstraction_ = BettingAbstraction::kFULLGAME;
  } else {
    SpielFatalError(absl::StrFormat("bettingAbstraction: %s not supported.",
                                    betting_abstraction));
  }
}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return absl::make_unique<UniversalPokerState>(shared_from_this());
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_num_cards bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  const int num_players = acpc_game_.GetNbPlayers();
  const int gameLength = MaxGameLength();
  const int total_num_cards = MaxChanceOutcomes();

  return {num_players + 2 * total_num_cards + 2 * gameLength};
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot
  const int num_players = acpc_game_.GetNbPlayers();
  const int total_num_cards = MaxChanceOutcomes();
  return {2 * (num_players + total_num_cards)};
}

double UniversalPokerGame::MaxCommitment() const {
  int max_commit = 0;
  if (acpc_game_.IsLimitGame()) {
    // The most a player can put into the pot is the raise amounts on each round
    // times the maximum number of raises, plus the original chips they put in
    // to play, which has the big blind as an upper bound.
    const auto &acpc_game = acpc_game_.Game();
    max_commit = big_blind();
    for (int i = 0; i < acpc_game_.NumRounds(); ++i) {
      max_commit += acpc_game.maxRaises[i] * acpc_game.raiseSize[i];
    }
  } else {
    // In No-Limit games, this isn't true, as there is no maximum raise value,
    // so the limit is the number of chips that the player has.
    max_commit = acpc_game_.StackSize(0);
  }
  return static_cast<double>(max_commit);
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot,
  // The maximum amount of money a player can win is the maximum bet any player
  // can make, times the number of players (excluding the original player).
  return MaxCommitment() * (acpc_game_.GetNbPlayers() - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game. As such,
  // the most a player can lose is the maximum amount they can bet.
  return -1 * MaxCommitment();
}

int UniversalPokerGame::MaxChanceOutcomes() const {
  return acpc_game_.NumSuitsDeck() * acpc_game_.NumRanksDeck();
}

int UniversalPokerGame::NumPlayers() const { return acpc_game_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  if (betting_abstraction_ == BettingAbstraction::kFULLGAME) {
    // 0 -> fold, 1 -> check/call, N -> bet size
    return max_stack_size_ + 1;
  } else {
    return GetMaxBettingActions(acpc_game_);
  }
}

int UniversalPokerGame::MaxGameLength() const {
  // We cache this as this is very slow to calculate.
  if (max_game_length_) return *max_game_length_;

  // Make a good guess here because bruteforcing the tree is far too slow
  // One Terminal Action
  int length = 1;

  // Deal Actions
  length += acpc_game_.GetTotalNbBoardCards() +
            acpc_game_.GetNbHoleCardsRequired() * acpc_game_.GetNbPlayers();

  // Check Actions
  length += (NumPlayers() * acpc_game_.NumRounds());

  // Bet Actions
  double maxStack = 0;
  double maxBlind = 0;
  for (uint32_t p = 0; p < NumPlayers(); p++) {
    maxStack =
        acpc_game_.StackSize(p) > maxStack ? acpc_game_.StackSize(p) : maxStack;
    maxBlind =
        acpc_game_.BlindSize(p) > maxStack ? acpc_game_.BlindSize(p) : maxBlind;
  }

  while (maxStack > maxBlind) {
    maxStack /= 2.0;         // You have always to bet the pot size
    length += NumPlayers();  // Each player has to react
  }
  return length;
}

/**
 * Parses the Game Paramters and makes a gameDesc out of it
 * @param map
 * @return
 */
std::string UniversalPokerGame::parseParameters(const GameParameters &map) {
  if (map.find("gamedef") != map.end()) {
    // We check for sanity that all parameters are empty
    if (map.size() != 1) {
      std::vector<std::string> game_parameter_keys;
      game_parameter_keys.reserve(map.size());
      for (auto const &imap : map) {
        game_parameter_keys.push_back(imap.first);
      }
      SpielFatalError(
          absl::StrCat("When loading a 'universal_poker' game, the 'gamedef' "
                       "field was present, but other fields were present too: ",
                       absl::StrJoin(game_parameter_keys, ", "),
                       "gamedef is exclusive with other paraemters."));
    }
    return ParameterValue<std::string>("gamedef");
  }

  std::string generated_gamedef = "GAMEDEF\n";

  absl::StrAppend(
      &generated_gamedef, ParameterValue<std::string>("betting"), "\n",
      "numPlayers = ", ParameterValue<int>("numPlayers"), "\n",
      "numRounds = ", ParameterValue<int>("numRounds"), "\n",
      "numsuits = ", ParameterValue<int>("numSuits"), "\n",
      "firstPlayer = ", ParameterValue<std::string>("firstPlayer"), "\n",
      "numRanks = ", ParameterValue<int>("numRanks"), "\n",
      "numHoleCards = ", ParameterValue<int>("numHoleCards"), "\n",
      "numBoardCards = ", ParameterValue<std::string>("numBoardCards"), "\n");

  std::string max_raises = ParameterValue<std::string>("maxRaises");
  if (!max_raises.empty()) {
    absl::StrAppend(&generated_gamedef, "maxRaises = ", max_raises, "\n");
  }

  if (ParameterValue<std::string>("betting") == "limit") {
    std::string raise_size = ParameterValue<std::string>("raiseSize");
    if (!raise_size.empty()) {
      absl::StrAppend(&generated_gamedef, "raiseSize = ", raise_size, "\n");
    }
  } else if (ParameterValue<std::string>("betting") == "nolimit") {
    std::string stack = ParameterValue<std::string>("stack");
    if (!stack.empty()) {
      absl::StrAppend(&generated_gamedef, "stack = ", stack, "\n");
    }
  } else {
    SpielFatalError(absl::StrCat("betting should be limit or nolimit, not ",
                                 ParameterValue<std::string>("betting")));
  }

  absl::StrAppend(&generated_gamedef,
                  "blind = ", ParameterValue<std::string>("blind"), "\n");
  absl::StrAppend(&generated_gamedef, "END GAMEDEF\n");

  std::vector<std::string> blinds =
      absl::StrSplit(ParameterValue<std::string>("blind"), ' ');
  big_blind_ = 0;
  for (const std::string &blind : blinds) {
    big_blind_ = std::max(big_blind_, std::stoi(blind));
  }
  // By requiring a blind/ante of at least a single chip, we're able to
  // structure the action space more intuitively in the kFULLGAME setting.
  // Specifically, action 0 -> fold, 1 -> call, and N -> raise to N chips.
  // While the ACPC server does not require it, in practice poker is always
  // played with a blind or ante, so this is a minor restriction.
  if (big_blind_ <= 0) {
    SpielFatalError("Must have a blind of at least one chip.");
  }
  std::vector<std::string> stacks =
      absl::StrSplit(ParameterValue<std::string>("stack"), ' ');
  max_stack_size_ = 0;
  for (const std::string &stack : stacks) {
    max_stack_size_ = std::max(max_stack_size_, std::stoi(stack));
  }
  return generated_gamedef;
}

const char *actions = "0df0c000p0000000a";

void UniversalPokerState::ApplyChoiceAction(StateActionType action_type,
                                            int size) {
  SPIEL_CHECK_GE(cur_player_, 0);  // No chance not terminal.

  actionSequence_ += (char)actions[action_type];
  if (action_type == ACTION_DEAL) SpielFatalError("Cannot apply deal action.");
  acpc_state_.DoAction(UniversalPokerActionTypeToACPCActionType(action_type),
                       size);
  _CalculateActionsAndNodeType();
}

void UniversalPokerState::_CalculateActionsAndNodeType() {
  possibleActions_ = 0;

  if (acpc_state_.IsFinished()) {
    if (acpc_state_.NumFolded() >= acpc_game_->GetNbPlayers() - 1) {
      // All players except one has fold.
      cur_player_ = kTerminalPlayerId;
    } else {
      if (board_cards_dealt_ <
          acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
        cur_player_ = kChancePlayerId;
        possibleActions_ = ACTION_DEAL;
        return;
      }
      // Showdown!
      cur_player_ = kTerminalPlayerId;
    }

  } else {
    // Check if we need to deal cards. We assume all cards are dealt at the
    // start of the game.
    if (hole_cards_dealt_ <
        acpc_game_->GetNbHoleCardsRequired() * acpc_game_->GetNbPlayers()) {
      cur_player_ = kChancePlayerId;
      possibleActions_ = ACTION_DEAL;
      return;
    }
    // 2. We need to deal a public card.
    if (board_cards_dealt_ <
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
      cur_player_ = kChancePlayerId;
      possibleActions_ = ACTION_DEAL;
      return;
    }

    // Check for CHOICE Actions
    cur_player_ = acpc_state_.CurrentPlayer();
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
      possibleActions_ |= ACTION_FOLD;
    }
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
      possibleActions_ |= ACTION_CHECK_CALL;
    }

    int potSize = 0;
    int allInSize = 0;
    // We have to call this as this sets potSize and allInSize_.
    bool valid_to_raise = acpc_state_.RaiseIsValid(&potSize, &allInSize);
    if (betting_abstraction_ == BettingAbstraction::kFC) return;
    if (valid_to_raise) {
      if (acpc_game_->IsLimitGame()) {
        potSize = 0;
        // There's only one "bet" allowed in Limit, which is "all-in or fixed
        // bet".
        possibleActions_ |= ACTION_BET;
      } else {
        int cur_spent = acpc_state_.CurrentSpent(acpc_state_.CurrentPlayer());
        int pot_raise_to =
            acpc_state_.TotalSpent() + 2 * acpc_state_.MaxSpend() - cur_spent;

        if (pot_raise_to >= potSize && pot_raise_to <= allInSize) {
          potSize = pot_raise_to;
          possibleActions_ |= ACTION_BET;
        }

        if (pot_raise_to != allInSize) {
          // If the raise to amount happens to match the number of chips I have,
          // then this action was already added as a pot-bet.
          possibleActions_ |= ACTION_ALL_IN;
        }
      }
    }
  }
}

const int UniversalPokerState::GetPossibleActionCount() const {
  // _builtin_popcount(int) function is used to count the number of one's
  return __builtin_popcount(possibleActions_);
}

open_spiel::Action ACPCActionToOpenSpielAction(
    const project_acpc_server::Action &action,
    const UniversalPokerState &state) {
  switch (action.type) {
    case project_acpc_server::ActionType::a_fold:
      return ActionType::kFold;
    case project_acpc_server::ActionType::a_call:
      return ActionType::kCall;
    case project_acpc_server::ActionType::a_raise:
      SPIEL_CHECK_NE(state.betting(), BettingAbstraction::kFC);
      if (action.size == state.PotSize(0.5)) {
        return ActionType::kHalfPot;
      }
      if (action.size == state.PotSize()) return ActionType::kBet;
      if (action.size == state.AllInSize()) return ActionType::kAllIn;
      if (state.betting() != BettingAbstraction::kFULLGAME) {
        SpielFatalError(absl::StrCat(
            "Unsupported bet size: ", action.size, ", pot: ", state.PotSize(),
            ", all_in: ", state.AllInSize(),
            ", max_commitment: ", state.acpc_state().raw_state().maxSpent));
      }
      SPIEL_CHECK_EQ(state.betting(), BettingAbstraction::kFULLGAME);
      return ActionType::kBet + action.size;
    case project_acpc_server::ActionType::a_invalid:
      SpielFatalError("Invalid action type.");
    default:
      SpielFatalError(absl::StrCat("Type not found. Type: ", action.type));
  }
  // Will never get called.
  return kInvalidAction;
}

std::ostream &operator<<(std::ostream &os, const BettingAbstraction &betting) {
  os << BettingAbstractionToString(betting);
  return os;
}

}  // namespace universal_poker
}  // namespace open_spiel
