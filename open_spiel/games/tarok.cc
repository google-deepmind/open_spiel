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

#include "open_spiel/games/tarok.h"

#include <algorithm>
#include <cmath>
#include <ctime>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tarok {

const GameType kGameType{"tarok",            // short_name
                         "Slovenian Tarok",  // long_name
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kSampledStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         4,      // max_num_players
                         3,      // min_num_players
                         true,   // provides_information_state_string
                         false,  // provides_information_state_tensor
                         false,  // provides_observation_string
                         false,  // provides_observation_tensor
                         // parameter_specification
                         {{"players", GameParameter(kDefaultNumPLayers)},
                          {"rng_seed", GameParameter(kDefaultSeed)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TarokGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// game implementation
TarokGame::TarokGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      rng_(std::mt19937(ParameterValue<int>("rng_seed") == -1
                            ? std::time(0)
                            : ParameterValue<int>("rng_seed"))) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
}

int TarokGame::NumDistinctActions() const { return 54; }

std::unique_ptr<State> TarokGame::NewInitialState() const {
  return NewInitialTarokState();
}

std::unique_ptr<TarokState> TarokGame::NewInitialTarokState() const {
  return std::make_unique<TarokState>(shared_from_this());
}

int TarokGame::MaxChanceOutcomes() const {
  // game is implicitly stochastic
  return 1;
}

int TarokGame::NumPlayers() const { return num_players_; }

double TarokGame::MinUtility() const { return -500.0; }

double TarokGame::MaxUtility() const { return 500.0; }

int TarokGame::MaxGameLength() const {
  if (num_players_ == 3) {
    // 17 actions + 16 cards each
    return 65;
  } else {
    // 24 actions + 12 cards each
    return 72;
  }
}

std::unique_ptr<State> TarokGame::DeserializeState(
    const std::string& str) const {
  std::unique_ptr<TarokState> state = NewInitialTarokState();
  if (str.empty()) return state;

  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  for (int i = 0; i < lines.size(); i++) {
    if (i == 0) {
      // chance node where we presisted the card dealing seed, see
      // TarokState::DoApplyActionInCardDealing for more info
      std::tie(state->talon_, state->players_cards_) =
          DealCards(num_players_, std::stoi(lines.at(i)));
      state->current_game_phase_ = GamePhase::kBidding;
      state->current_player_ = 1;
      state->AddPrivateCardsToInfoStates();
    } else {
      state->ApplyAction(std::stol(lines.at(i)));
    }
  }
  return state;
}

std::string TarokGame::GetRNGState() const {
  std::ostringstream rng_stream;
  rng_stream << rng_;
  return rng_stream.str();
}

void TarokGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) return;
  std::istringstream rng_stream(rng_state);
  rng_stream >> rng_;
}

int TarokGame::RNG() const { return rng_(); }

// state implementation
TarokState::TarokState(std::shared_ptr<const Game> game)
    : State(game),
      tarok_parent_game_(std::static_pointer_cast<const TarokGame>(game)) {
  players_bids_.reserve(num_players_);
  players_bids_.insert(players_bids_.end(), num_players_, kInvalidBidAction);
  players_collected_cards_.reserve(num_players_);
  players_collected_cards_.insert(players_collected_cards_.end(), num_players_,
                                  std::vector<Action>());
  players_info_states_.reserve(num_players_);
  players_info_states_.insert(players_info_states_.end(), num_players_, "");
}

Player TarokState::CurrentPlayer() const {
  switch (current_game_phase_) {
    case GamePhase::kCardDealing:
      return kChancePlayerId;
    case GamePhase::kFinished:
      return kTerminalPlayerId;
    default:
      return current_player_;
  }
}

bool TarokState::IsTerminal() const {
  return current_game_phase_ == GamePhase::kFinished;
}

GamePhase TarokState::CurrentGamePhase() const { return current_game_phase_; }

std::vector<Action> TarokState::PlayerCards(Player player) const {
  if (current_game_phase_ == GamePhase::kCardDealing) return {};
  return players_cards_.at(player);
}

ContractName TarokState::SelectedContractName() const {
  if (current_game_phase_ == GamePhase::kCardDealing ||
      current_game_phase_ == GamePhase::kBidding) {
    return ContractName::kNotSelected;
  }
  return selected_contract_->name;
}

std::vector<Action> TarokState::Talon() const { return talon_; }

std::vector<std::vector<Action>> TarokState::TalonSets() const {
  if (current_game_phase_ != GamePhase::kTalonExchange) return {};

  int num_talon_sets = talon_.size() / selected_contract_->num_talon_exchanges;
  std::vector<std::vector<Action>> talon_sets;
  talon_sets.reserve(num_talon_sets);

  auto begin = talon_.begin();
  for (int i = 0; i < num_talon_sets; i++) {
    talon_sets.push_back(std::vector<Action>(
        begin, begin + selected_contract_->num_talon_exchanges));
    std::advance(begin, selected_contract_->num_talon_exchanges);
  }
  return talon_sets;
}

std::vector<Action> TarokState::TrickCards() const { return trick_cards_; }

std::vector<Action> TarokState::LegalActions() const {
  // all card actions are encoded as 0, 1, ..., 52, 53 and correspond to card
  // indices wrt. tarok_parent_game_->card_deck_, card actions are returned:
  //   - in the king calling phase
  //   - by LegalActionsInTalonExchange() after the talon set is selected (i.e.
  //     when discarding the cards)
  //   - by LegalActionsInTricksPlaying()
  switch (current_game_phase_) {
    case GamePhase::kCardDealing:
      // return a dummy action due to implicit stochasticity
      return {0};
    case GamePhase::kBidding:
      return LegalActionsInBidding();
    case GamePhase::kKingCalling:
      return {kKingOfHeartsAction, kKingOfDiamondsAction, kKingOfSpadesAction,
              kKingOfClubsAction};
    case GamePhase::kTalonExchange:
      return LegalActionsInTalonExchange();
    case GamePhase::kTricksPlaying:
      return LegalActionsInTricksPlaying();
    case GamePhase::kFinished:
      return {};
  }
}

std::vector<Action> TarokState::LegalActionsInBidding() const {
  // actions 1 - 12 correspond to contracts in tarok_parent_game_->contracts_
  // respectively, action 0 means pass
  auto it = std::max_element(players_bids_.begin(), players_bids_.end());
  int max_bid = *it;
  int max_bid_player = it - players_bids_.begin();

  std::vector<Action> actions;
  if (current_player_ == 0 &&
      players_bids_.at(current_player_) == kInvalidBidAction &&
      AllButCurrentPlayerPassedBidding()) {
    // no bidding has happened before so forehand can
    // bid any contract but can't pass
    actions.insert(actions.end(), {kBidKlopAction, kBidThreeAction});
  } else if (!AllButCurrentPlayerPassedBidding()) {
    // other players still playing
    actions.push_back(kBidPassAction);
  }

  for (int action = 3; action <= 12; action++) {
    if (num_players_ == 3 && action >= kBidSoloThreeAction &&
        action <= kBidSoloOneAction) {
      // skip solo contracts for three players
      continue;
    }
    if (action < max_bid) {
      continue;
    }
    if ((action > max_bid) ||
        (action == max_bid && current_player_ <= max_bid_player)) {
      actions.push_back(action);
    }
  }
  return actions;
}

std::vector<Action> TarokState::LegalActionsInTalonExchange() const {
  if (talon_.size() == 6) {
    // choosing one of the talon card sets where actions are encoded as
    // 0, 1, 2, etc. from left to right, i.e. 0 is the leftmost talon set
    // as returned by TalonSets()
    std::vector<Action> actions(6 / selected_contract_->num_talon_exchanges);
    std::iota(actions.begin(), actions.end(), 0);
    return actions;
  }
  // prevent discarding of taroks and kings
  std::vector<Action> actions;
  for (auto const& action : players_cards_.at(current_player_)) {
    const Card& card = ActionToCard(action);
    if (card.suit != CardSuit::kTaroks && card.points != 5)
      actions.push_back(action);
  }
  // allow discarding of taroks (except of trula) if player has no other choice
  if (actions.empty()) {
    for (auto const& action : players_cards_.at(current_player_)) {
      if (ActionToCard(action).points != 5) actions.push_back(action);
    }
  }
  return actions;
}

std::vector<Action> TarokState::LegalActionsInTricksPlaying() const {
  if (trick_cards_.empty()) {
    // trick opening, i.e. the current player is choosing
    // the first card for this trick
    if (selected_contract_->is_negative)
      return RemovePagatIfNeeded(players_cards_.at(current_player_));
    return players_cards_.at(current_player_);
  } else {
    // trick following
    return LegalActionsInTricksPlayingFollowing();
  }
}

std::vector<Action> TarokState::LegalActionsInTricksPlayingFollowing() const {
  auto [can_follow_suit, cant_follow_suit_but_has_tarok] =
      CanFollowSuitOrCantButHasTarok();

  CardSuit take_suit;
  if (can_follow_suit) {
    take_suit = ActionToCard(trick_cards_.front()).suit;
  } else if (cant_follow_suit_but_has_tarok) {
    take_suit = CardSuit::kTaroks;
  } else {
    // can't follow suit and doesn't have taroks so any card can be played
    return players_cards_.at(current_player_);
  }

  if (selected_contract_->is_negative)
    return TakeSuitFromPlayerCardsInNegativeContracts(take_suit);
  else
    return TakeSuitFromPlayerCardsInPositiveContracts(take_suit);
}

std::tuple<bool, bool> TarokState::CanFollowSuitOrCantButHasTarok() const {
  const Card& opening_card = ActionToCard(trick_cards_.front());
  bool has_taroks = false;
  for (auto const& action : players_cards_.at(current_player_)) {
    const Card& current_card = ActionToCard(action);
    if (current_card.suit == opening_card.suit) {
      // note that the second return value is irrelevant in this case
      return {true, false};
    }
    if (current_card.suit == CardSuit::kTaroks) {
      has_taroks = true;
    }
  }
  return {false, has_taroks};
}

std::vector<Action> TarokState::TakeSuitFromPlayerCardsInNegativeContracts(
    CardSuit suit) const {
  bool player_has_pagat =
      ActionInActions(kPagatAction, players_cards_.at(current_player_));
  if (player_has_pagat && ActionInActions(kMondAction, trick_cards_) &&
      ActionInActions(kSkisAction, trick_cards_)) {
    // the emperor trick, i.e. pagat has to be played as it is the only card
    // that will win the trick
    return {kPagatAction};
  }

  absl::optional<Action> action_to_beat = ActionToBeatInNegativeContracts(suit);
  std::vector<Action> actions;

  if (action_to_beat) {
    const Card& card_to_beat = ActionToCard(*action_to_beat);
    auto const& player_cards = players_cards_.at(current_player_);
    // a higher card only has to be played when the player actually has a higher
    // card otherwise any card of the suit can be played
    bool has_higher_card = false;
    for (auto const& action : player_cards) {
      const Card& current_card = ActionToCard(action);
      if (current_card.suit == suit && current_card.rank > card_to_beat.rank) {
        has_higher_card = true;
        break;
      }
    }
    // collect the actual cards
    for (auto const& action : player_cards) {
      const Card& current_card = ActionToCard(action);
      if (current_card.suit == suit &&
          (!has_higher_card || current_card.rank > card_to_beat.rank)) {
        actions.push_back(action);
      }
    }
  } else {
    // no need to beat any card so simply return all cards of the correct suit
    actions = TakeSuitFromPlayerCardsInPositiveContracts(suit);
  }

  if (player_has_pagat)
    return RemovePagatIfNeeded(actions);
  else
    return actions;
}

absl::optional<Action> TarokState::ActionToBeatInNegativeContracts(
    CardSuit suit) const {
  // there are two cases where no card has to be beaten; the player is following
  // a colour suit and there is already at least one tarok in trick_cards_ or
  // the player is forced to play a tarok and there are no taroks in
  // trick_cards_
  bool tarok_in_trick_cards = false;
  for (auto const& action : trick_cards_) {
    if (ActionToCard(action).suit == CardSuit::kTaroks) {
      tarok_in_trick_cards = true;
      break;
    }
  }
  if ((suit != CardSuit::kTaroks && tarok_in_trick_cards) ||
      (suit == CardSuit::kTaroks && !tarok_in_trick_cards)) {
    return {};
  }
  // the specified suit should be present in trick_cards_ from here on because
  // it is either a suit of the opening card or CardSuit::kTaroks with existing
  // taroks in trick_cards_
  Action action_to_beat = trick_cards_.front();
  for (int i = 1; i < trick_cards_.size(); i++) {
    const Card& card_to_beat = ActionToCard(action_to_beat);
    const Card& current_card = ActionToCard(trick_cards_.at(i));
    if (current_card.suit == suit && current_card.rank > card_to_beat.rank)
      action_to_beat = trick_cards_.at(i);
  }
  return action_to_beat;
}

std::vector<Action> TarokState::RemovePagatIfNeeded(
    const std::vector<Action>& actions) const {
  if (actions.size() > 1) {
    // mustn't play pagat unless it's the only card, note that actions
    // can be all player's cards or a subset already filtered by the caller
    std::vector<Action> actions_no_pagat;
    for (auto const& action : actions) {
      if (action != kPagatAction) actions_no_pagat.push_back(action);
    }
    return actions_no_pagat;
  }
  return actions;
}

std::vector<Action> TarokState::TakeSuitFromPlayerCardsInPositiveContracts(
    CardSuit suit) const {
  std::vector<Action> actions;
  for (auto const& action : players_cards_.at(current_player_)) {
    if (ActionToCard(action).suit == suit) actions.push_back(action);
  }
  return actions;
}

std::string TarokState::ActionToString(Player player, Action action_id) const {
  switch (current_game_phase_) {
    case GamePhase::kCardDealing:
      // return a dummy action due to implicit stochasticity
      return "Deal";
    case GamePhase::kBidding:
      if (action_id == 0) return "Pass";
      return ContractNameToString(
          tarok_parent_game_->contracts_.at(action_id - 1).name);
    case GamePhase::kKingCalling:
    case GamePhase::kTricksPlaying:
      return CardActionToString(action_id);
    case GamePhase::kTalonExchange:
      if (talon_.size() == 6) return absl::StrCat("Talon set ", action_id + 1);
      return CardActionToString(action_id);
    case GamePhase::kFinished:
      return "";
  }
}

std::string TarokState::CardActionToString(Action action_id) const {
  return ActionToCard(action_id).ToString();
}

ActionsAndProbs TarokState::ChanceOutcomes() const {
  if (current_game_phase_ == GamePhase::kCardDealing) {
    // return a dummy action with probability 1 due to implicit stochasticity
    return {{0, 1.0}};
  }
  return {};
}

void TarokState::DoApplyAction(Action action_id) {
  if (!ActionInActions(action_id, LegalActions())) {
    SpielFatalError(absl::StrCat("Action ", action_id,
                                 " is not valid in the current state."));
  }
  switch (current_game_phase_) {
    case GamePhase::kCardDealing:
      DoApplyActionInCardDealing();
      break;
    case GamePhase::kBidding:
      DoApplyActionInBidding(action_id);
      break;
    case GamePhase::kKingCalling:
      DoApplyActionInKingCalling(action_id);
      break;
    case GamePhase::kTalonExchange:
      DoApplyActionInTalonExchange(action_id);
      break;
    case GamePhase::kTricksPlaying:
      DoApplyActionInTricksPlaying(action_id);
      break;
    case GamePhase::kFinished:
      SpielFatalError("Calling DoApplyAction in a terminal state.");
  }
}

void TarokState::DoApplyActionInCardDealing() {
  // do the actual sampling here due to implicit stochasticity
  do {
    // seed is persisted for serialization purposes
    card_dealing_seed_ = tarok_parent_game_->RNG();
    // hands without taroks are illegal
    std::tie(talon_, players_cards_) =
        DealCards(num_players_, card_dealing_seed_);
  } while (AnyPlayerWithoutTaroks());
  current_game_phase_ = GamePhase::kBidding;
  // lower player indices correspond to higher bidding priority,
  // i.e. 0 is the forehand, num_players - 1 is the dealer
  current_player_ = 1;
  AddPrivateCardsToInfoStates();
}

bool TarokState::AnyPlayerWithoutTaroks() const {
  // actions are sorted, i.e. taroks are always at the beginning
  for (int i = 0; i < num_players_; i++) {
    if (ActionToCard(players_cards_.at(i).front()).suit != CardSuit::kTaroks) {
      return true;
    }
  }
  return false;
}

void TarokState::AddPrivateCardsToInfoStates() {
  for (int i = 0; i < num_players_; i++) {
    AppendToInformationState(
        i, absl::StrCat(absl::StrJoin(players_cards_.at(i), ","), ";"));
  }
}

void TarokState::DoApplyActionInBidding(Action action_id) {
  players_bids_.at(current_player_) = action_id;
  AppendToAllInformationStates(std::to_string(action_id));
  if (AllButCurrentPlayerPassedBidding()) {
    FinishBiddingPhase(action_id);
    AppendToAllInformationStates(";");
  } else {
    do {
      NextPlayer();
    } while (players_bids_.at(current_player_) == kBidPassAction);
    AppendToAllInformationStates(",");
  }
}

bool TarokState::AllButCurrentPlayerPassedBidding() const {
  for (int i = 0; i < num_players_; i++) {
    if (i == current_player_) continue;
    if (players_bids_.at(i) != kBidPassAction) return false;
  }
  return true;
}

void TarokState::FinishBiddingPhase(Action action_id) {
  declarer_ = current_player_;
  selected_contract_ = &tarok_parent_game_->contracts_.at(action_id - 1);
  if (num_players_ == 4 && selected_contract_->needs_king_calling)
    current_game_phase_ = GamePhase::kKingCalling;
  else if (selected_contract_->NeedsTalonExchange())
    current_game_phase_ = GamePhase::kTalonExchange;
  else
    StartTricksPlayingPhase();
}

void TarokState::DoApplyActionInKingCalling(Action action_id) {
  called_king_ = action_id;
  if (ActionInActions(action_id, talon_)) {
    called_king_in_talon_ = true;
  } else {
    for (int i = 0; i < num_players_; i++) {
      if (i == current_player_) {
        continue;
      } else if (ActionInActions(action_id, players_cards_.at(i))) {
        declarer_partner_ = i;
        break;
      }
    }
  }
  current_game_phase_ = GamePhase::kTalonExchange;
  AppendToAllInformationStates(absl::StrCat(action_id, ";"));
}

void TarokState::DoApplyActionInTalonExchange(Action action_id) {
  auto& player_cards = players_cards_.at(current_player_);

  if (talon_.size() == 6) {
    // add all talon cards to info states
    AppendToAllInformationStates(absl::StrCat(absl::StrJoin(talon_, ","), ";"));

    // choosing one of the talon card sets
    int set_begin = action_id * selected_contract_->num_talon_exchanges;
    int set_end = set_begin + selected_contract_->num_talon_exchanges;

    bool mond_in_talon = ActionInActions(kMondAction, talon_);
    bool mond_in_selected_talon_set = false;
    for (int i = set_begin; i < set_end; i++) {
      player_cards.push_back(talon_.at(i));
      if (talon_.at(i) == kMondAction) mond_in_selected_talon_set = true;
    }
    if (mond_in_talon && !mond_in_selected_talon_set) {
      // the captured mond penalty applies if mond is in talon and not part of
      // the selected set
      captured_mond_player_ = current_player_;
    }

    // add the selected talon set to info states
    AppendToAllInformationStates(absl::StrCat(action_id, ";"));

    std::sort(player_cards.begin(), player_cards.end());
    talon_.erase(talon_.begin() + set_begin, talon_.begin() + set_end);
  } else {
    // discarding the cards
    MoveActionFromTo(action_id, &player_cards,
                     &players_collected_cards_.at(current_player_));

    bool talon_exchange_finished = player_cards.size() == 48 / num_players_;
    std::string info_state_delimiter = talon_exchange_finished ? ";" : ",";

    // note that all players see discarded tarok cards but only the discarder
    // knows about discarded non-taroks
    if (ActionToCard(action_id).suit == CardSuit::kTaroks) {
      AppendToAllInformationStates(
          absl::StrCat(action_id, info_state_delimiter));
    } else {
      AppendToInformationState(current_player_,
                               absl::StrCat(action_id, info_state_delimiter));
      for (Player p = 0; p < num_players_; p++) {
        if (p == current_player_) continue;
        AppendToInformationState(p, absl::StrCat("d", info_state_delimiter));
      }
    }

    if (talon_exchange_finished) StartTricksPlayingPhase();
  }
}

void TarokState::StartTricksPlayingPhase() {
  current_game_phase_ = GamePhase::kTricksPlaying;
  if (selected_contract_->declarer_starts)
    current_player_ = declarer_;
  else
    current_player_ = 0;
}

void TarokState::DoApplyActionInTricksPlaying(Action action_id) {
  MoveActionFromTo(action_id, &players_cards_.at(current_player_),
                   &trick_cards_);
  AppendToAllInformationStates(std::to_string(action_id));
  if (trick_cards_.size() == num_players_) {
    ResolveTrick();
    if (players_cards_.at(current_player_).empty() ||
        ((selected_contract_->name == ContractName::kBeggar ||
          selected_contract_->name == ContractName::kOpenBeggar) &&
         current_player_ == declarer_) ||
        ((selected_contract_->name == ContractName::kColourValatWithout ||
          selected_contract_->name == ContractName::kValatWithout) &&
         current_player_ != declarer_)) {
      current_game_phase_ = GamePhase::kFinished;
    } else {
      AppendToAllInformationStates(";");
    }
  } else {
    NextPlayer();
    AppendToAllInformationStates(",");
  }
}

void TarokState::ResolveTrick() {
  auto [trick_winner, winning_action] = ResolveTrickWinnerAndWinningAction();
  auto& trick_winner_collected_cards =
      players_collected_cards_.at(trick_winner);

  for (auto const& action : trick_cards_) {
    trick_winner_collected_cards.push_back(action);
  }

  if (selected_contract_->name == ContractName::kKlop && !talon_.empty()) {
    // add the "gift" talon card in klop
    trick_winner_collected_cards.push_back(talon_.front());
    AppendToAllInformationStates(absl::StrCat(",", talon_.front()));
    talon_.erase(talon_.begin());
  } else if (winning_action == called_king_ && called_king_in_talon_) {
    // declearer won the trick with the called king that was in talon so all
    // of the talon cards belong to the declearer (note that this is only
    // possible when talon exchange actually happened in the past)
    bool mond_in_talon = false;
    for (auto const& action : talon_) {
      trick_winner_collected_cards.push_back(action);
      if (action == kMondAction) mond_in_talon = true;
    }
    if (mond_in_talon) {
      // the called king and mond were in different parts of the talon and
      // declearer selected the set with the king plus won the mond as
      // part of the obtained talon remainder, negating the captured mond
      // penalty obtained during DoApplyActionInTalonExchange()
      captured_mond_player_ = kInvalidPlayer;
    }
    talon_.clear();
  } else if ((selected_contract_->NeedsTalonExchange() ||
              selected_contract_->name == ContractName::kSoloWithout) &&
             (winning_action == kSkisAction ||
              winning_action == kPagatAction)) {
    // check if mond is captured by skis or pagat (emperor's trick) and
    // penalise the player of the mond in certain contracts
    for (int i = 0; i < trick_cards_.size(); i++) {
      if (trick_cards_.at(i) == kMondAction) {
        captured_mond_player_ = TrickCardsIndexToPlayer(i);
      }
    }
  }

  trick_cards_.clear();
  current_player_ = trick_winner;
}

TrickWinnerAndAction TarokState::ResolveTrickWinnerAndWinningAction() const {
  // compute the winning action index within trick_cards_
  int winning_action_i;
  if ((ActionInActions(kPagatAction, trick_cards_) &&
       ActionInActions(kMondAction, trick_cards_) &&
       ActionInActions(kSkisAction, trick_cards_)) &&
      (selected_contract_->name != ContractName::kColourValatWithout ||
       ActionToCard(trick_cards_.front()).suit == CardSuit::kTaroks)) {
    // the emperor trick, i.e. pagat wins over mond and skis in all cases but
    // not in Contract::kColourValatWithout when a non-trump is led
    winning_action_i =
        std::find(trick_cards_.begin(), trick_cards_.end(), kPagatAction) -
        trick_cards_.begin();
  } else {
    winning_action_i = 0;
    for (int i = 1; i < trick_cards_.size(); i++) {
      const Card& winning_card =
          ActionToCard(trick_cards_.at(winning_action_i));
      const Card& current_card = ActionToCard(trick_cards_.at(i));

      if (((current_card.suit == CardSuit::kTaroks &&
            selected_contract_->name != ContractName::kColourValatWithout) ||
           current_card.suit == winning_card.suit) &&
          current_card.rank > winning_card.rank) {
        winning_action_i = i;
      }
    }
  }
  return {TrickCardsIndexToPlayer(winning_action_i),
          trick_cards_.at(winning_action_i)};
}

Player TarokState::TrickCardsIndexToPlayer(int index) const {
  Player player = current_player_;
  for (int i = 0; i < trick_cards_.size() - 1 - index; i++) {
    player -= 1;
    if (player == -1) player = num_players_ - 1;
  }
  return player;
}

std::vector<double> TarokState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  if (!IsTerminal()) return returns;

  std::vector<int> penalties = CapturedMondPenalties();
  std::vector<int> scores = ScoresWithoutCapturedMondPenalties();
  for (int i = 0; i < num_players_; i++) {
    returns.at(i) = penalties.at(i) + scores.at(i);
  }
  return returns;
}

std::vector<int> TarokState::CapturedMondPenalties() const {
  std::vector<int> penalties(num_players_, 0);
  if (captured_mond_player_ != kInvalidPlayer)
    penalties.at(captured_mond_player_) = -20;
  return penalties;
}

std::vector<int> TarokState::ScoresWithoutCapturedMondPenalties() const {
  if (!IsTerminal()) return std::vector<int>(num_players_, 0);
  if (selected_contract_->name == ContractName::kKlop) {
    return ScoresInKlop();
  } else if (selected_contract_->NeedsTalonExchange()) {
    return ScoresInNormalContracts();
  } else {
    // beggar and above
    return ScoresInHigherContracts();
  }
}

std::vector<int> TarokState::ScoresInKlop() const {
  std::vector<int> scores;
  scores.reserve(num_players_);

  bool any_player_won_or_lost = false;
  for (int i = 0; i < num_players_; i++) {
    int points = CardPoints(players_collected_cards_.at(i),
                            tarok_parent_game_->card_deck_);
    if (points > 35) {
      any_player_won_or_lost = true;
      scores.push_back(-70);
    } else if (points == 0) {
      any_player_won_or_lost = true;
      scores.push_back(70);
    } else {
      scores.push_back(-points);
    }
  }
  if (any_player_won_or_lost) {
    // only the winners and losers score
    for (int i = 0; i < num_players_; i++) {
      if (std::abs(scores.at(i)) != 70) scores.at(i) = 0;
    }
  }
  return scores;
}

std::vector<int> TarokState::ScoresInNormalContracts() const {
  auto [collected_cards, opposite_collected_cards] =
      SplitCollectedCardsPerTeams();

  int score;
  if (collected_cards.size() == 48) {
    // valat won
    score = 250;
  } else if (opposite_collected_cards.size() == 48) {
    // valat lost
    score = -250;
  } else {
    int card_points =
        CardPoints(collected_cards, tarok_parent_game_->card_deck_);
    score = card_points - 35;

    if (card_points > 35)
      score += selected_contract_->score;
    else
      score -= selected_contract_->score;

    // bonuses could be positive, negative or 0
    int bonuses = NonValatBonuses(collected_cards, opposite_collected_cards);
    score += bonuses;
  }

  std::vector<int> scores(num_players_, 0);
  scores.at(declarer_) = score;
  if (declarer_partner_ != kInvalidPlayer) scores.at(declarer_partner_) = score;
  return scores;
}

CollectedCardsPerTeam TarokState::SplitCollectedCardsPerTeams() const {
  std::vector<Action> collected_cards = players_collected_cards_.at(declarer_);
  std::vector<Action> opposite_collected_cards;
  for (Player p = 0; p < num_players_; p++) {
    if (p != declarer_ && p != declarer_partner_) {
      opposite_collected_cards.insert(opposite_collected_cards.end(),
                                      players_collected_cards_.at(p).begin(),
                                      players_collected_cards_.at(p).end());
    } else if (p == declarer_partner_) {
      collected_cards.insert(collected_cards.end(),
                             players_collected_cards_.at(p).begin(),
                             players_collected_cards_.at(p).end());
    }
  }
  return {collected_cards, opposite_collected_cards};
}

int TarokState::NonValatBonuses(
    const std::vector<Action>& collected_cards,
    const std::vector<Action>& opposite_collected_cards) const {
  int bonuses = 0;

  // last trick winner is the current player
  auto const& last_trick_winner_cards =
      players_collected_cards_.at(current_player_);
  // king ultimo and pagat ultimo
  int ultimo_bonus = 0;
  if (std::find(last_trick_winner_cards.end() - num_players_,
                last_trick_winner_cards.end(),
                called_king_) != last_trick_winner_cards.end()) {
    // king ultimo
    ultimo_bonus = 10;
  } else if (std::find(last_trick_winner_cards.end() - num_players_,
                       last_trick_winner_cards.end(),
                       0) != last_trick_winner_cards.end()) {
    // pagat ultimo
    ultimo_bonus = 25;
  }

  if (ultimo_bonus > 0 &&
      (current_player_ == declarer_ || current_player_ == declarer_partner_)) {
    bonuses = ultimo_bonus;
  } else if (ultimo_bonus > 0) {
    bonuses = -ultimo_bonus;
  }

  // collected kings or trula
  auto [collected_kings, collected_trula] =
      CollectedKingsAndOrTrula(collected_cards);
  auto [opposite_collected_kings, opposite_collected_trula] =
      CollectedKingsAndOrTrula(opposite_collected_cards);

  if (collected_kings)
    bonuses += 10;
  else if (opposite_collected_kings)
    bonuses -= 10;
  if (collected_trula)
    bonuses += 10;
  else if (opposite_collected_trula)
    bonuses -= 10;
  return bonuses;
}

std::tuple<bool, bool> TarokState::CollectedKingsAndOrTrula(
    const std::vector<Action>& collected_cards) const {
  int num_kings = 0, num_trula = 0;
  for (auto const& action : collected_cards) {
    if (action == kKingOfHeartsAction || kKingOfDiamondsAction == 37 ||
        action == kKingOfSpadesAction || kKingOfClubsAction == 53) {
      num_kings += 1;
    } else if (action == kPagatAction || action == kMondAction ||
               action == kSkisAction) {
      num_trula += 1;
    }
  }
  return {num_kings == 4, num_trula == 3};
}

std::vector<int> TarokState::ScoresInHigherContracts() const {
  bool declarer_won;
  if (selected_contract_->name == ContractName::kBeggar ||
      selected_contract_->name == ContractName::kOpenBeggar) {
    declarer_won = players_collected_cards_.at(declarer_).empty();
  } else if (selected_contract_->name == ContractName::kColourValatWithout ||
             selected_contract_->name == ContractName::kValatWithout) {
    declarer_won = players_collected_cards_.at(declarer_).size() == 48;
  } else {
    // solo without
    declarer_won = CardPoints(players_collected_cards_.at(declarer_),
                              tarok_parent_game_->card_deck_) > 35;
  }

  std::vector<int> scores(num_players_, 0);
  if (declarer_won)
    scores.at(declarer_) = selected_contract_->score;
  else
    scores.at(declarer_) = -selected_contract_->score;
  return scores;
}

std::string TarokState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return players_info_states_.at(player);
}

std::string TarokState::ToString() const {
  std::string str = "";
  GamePhase current_game_phase = CurrentGamePhase();
  absl::StrAppend(&str, "Game phase: ", GamePhaseToString(current_game_phase),
                  "\n");
  absl::StrAppend(&str, "Selected contract: ",
                  ContractNameToString(SelectedContractName()), "\n");

  Player current_player = CurrentPlayer();
  absl::StrAppend(&str, "Current player: ", current_player, "\n");
  if (current_game_phase != GamePhase::kCardDealing &&
      current_game_phase != GamePhase::kFinished) {
    absl::StrAppend(&str, "Player cards: ",
                    absl::StrJoin(PlayerCards(current_player), ","), "\n");
  }

  if (current_game_phase == GamePhase::kTalonExchange) {
    auto talon_sets = TalonSets();
    std::vector<std::string> talon_sets_strings;
    talon_sets_strings.reserve(talon_sets.size());
    for (auto const& set : talon_sets) {
      talon_sets_strings.push_back(absl::StrJoin(set, ","));
    }
    absl::StrAppend(
        &str, "Talon sets: ", absl::StrJoin(talon_sets_strings, ";"), "\n");
  } else if (current_game_phase == GamePhase::kTricksPlaying) {
    absl::StrAppend(&str, "Trick cards: ", absl::StrJoin(TrickCards(), ","),
                    "\n");
  }
  return str;
}

std::string TarokState::Serialize() const {
  if (current_game_phase_ == GamePhase::kCardDealing) return "";
  // replace the dummy stochastic action with the seed that was used
  // for dealing the cards
  std::vector<Action> history = History();
  history.front() = card_dealing_seed_;
  return absl::StrJoin(history, "\n");
}

std::unique_ptr<State> TarokState::Clone() const {
  return std::unique_ptr<State>(new TarokState(*this));
}

void TarokState::NextPlayer() {
  current_player_ += 1;
  if (current_player_ == num_players_) current_player_ = 0;
}

bool TarokState::ActionInActions(Action action_id,
                                 const std::vector<Action>& actions) {
  return std::find(actions.begin(), actions.end(), action_id) != actions.end();
}

void TarokState::MoveActionFromTo(Action action_id, std::vector<Action>* from,
                                  std::vector<Action>* to) {
  from->erase(std::remove(from->begin(), from->end(), action_id), from->end());
  to->push_back(action_id);
}

const Card& TarokState::ActionToCard(Action action_id) const {
  return tarok_parent_game_->card_deck_.at(action_id);
}

void TarokState::AppendToAllInformationStates(const std::string& appendix) {
  for (int i = 0; i < num_players_; i++) {
    absl::StrAppend(&players_info_states_.at(i), appendix);
  }
}

void TarokState::AppendToInformationState(Player player,
                                          const std::string& appendix) {
  absl::StrAppend(&players_info_states_.at(player), appendix);
}

std::ostream& operator<<(std::ostream& os, const GamePhase& game_phase) {
  os << GamePhaseToString(game_phase);
  return os;
}

std::string GamePhaseToString(const GamePhase& game_phase) {
  switch (game_phase) {
    case GamePhase::kCardDealing:
      return "Card dealing";
    case GamePhase::kBidding:
      return "Bidding";
    case GamePhase::kKingCalling:
      return "King calling";
    case GamePhase::kTalonExchange:
      return "Talon exchange";
    case GamePhase::kTricksPlaying:
      return "Tricks playing";
    case GamePhase::kFinished:
      return "Finished";
  }
}

}  // namespace tarok
}  // namespace open_spiel
