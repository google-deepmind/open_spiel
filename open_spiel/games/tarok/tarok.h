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

#ifndef OPEN_SPIEL_GAMES_TAROK_H_
#define OPEN_SPIEL_GAMES_TAROK_H_

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/tarok/cards.h"
#include "open_spiel/games/tarok/contracts.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tarok {

inline constexpr int kDefaultNumPLayers = 3;
// seed for shuffling the cards, -1 means seeded by clock
inline constexpr int kDefaultSeed = -1;

enum class GamePhase {
  kCardDealing,
  kBidding,
  kKingCalling,
  kTalonExchange,
  kTricksPlaying,
  kFinished
};

class TarokState;

// game definition
class TarokGame : public Game {
 public:
  explicit TarokGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  std::unique_ptr<TarokState> NewInitialTarokState() const;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  int MaxGameLength() const override;

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;
  std::string GetRNGState() const override;
  void SetRNGState(const std::string& rng_state) const override;

 private:
  friend class TarokState;
  // this function is const so that it can be called from state objects,
  // note that it nevertheless changes the state of the mutable rng_ used
  // for shuffling the cards, this is expected behaviour since the game
  // object has to maintain an internal RNG state due to implicit stochasticity,
  // see ChanceOutcomes() comments in open_spiel/spiel.h for more info
  int RNG() const;

  static inline const std::array<Card, 54> card_deck_ = InitializeCardDeck();
  static inline const std::array<Contract, 12> contracts_ =
      InitializeContracts();

  const int num_players_;
  mutable std::mt19937 rng_;
};

using TrickWinnerAndAction = std::tuple<Player, Action>;
using CollectedCardsPerTeam =
    std::tuple<std::vector<Action>, std::vector<Action>>;

// state definition
class TarokState : public State {
 public:
  explicit TarokState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  bool IsTerminal() const override;
  GamePhase CurrentGamePhase() const;
  std::vector<Action> PlayerCards(Player player) const;
  ContractName SelectedContractName() const;
  std::vector<Action> Talon() const;
  std::vector<std::vector<Action>> TalonSets() const;
  std::vector<Action> TrickCards() const;

  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string CardActionToString(Action action_id) const;
  ActionsAndProbs ChanceOutcomes() const override;

  // calculates the overall score for a finished game without radli, see
  // comments above CapturedMondPenalties() for more details
  std::vector<double> Returns() const override;
  // the following two methods are kept separately due to the captured mond
  // penalty not being affected by any multipliers for kontras or radli, note
  // that TarokState does not implement radli as they are, like cumulative
  // players' score, part of the global state that would have to be kept between
  // multiple NewInitialState() calls (i.e. TarokState only implements a single
  // round of the game and radli implementation is left to the owner of the game
  // instance who should keep track of multiple rounds if needed)
  std::vector<int> CapturedMondPenalties() const;
  std::vector<int> ScoresWithoutCapturedMondPenalties() const;

  // info state strings are of the following format (cards and actions are
  // delimited by a comma character, some parts of the string are omitted in
  // states where corresponding game phases are not played,
  // single_trick_played_actions also contains the gift talon card in klop):
  //
  // each_players_private_cards;bidding_actions;king_calling_action;
  // talon_cards;choosing_talon_set_action;discarding_cards_actions;
  // single_trick_played_actions;...;single_trick_played_actions
  std::string InformationStateString(Player player) const override;

  std::string ToString() const override;
  std::string Serialize() const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  friend class TarokGame;

  std::vector<Action> LegalActionsInBidding() const;
  std::vector<Action> LegalActionsInTalonExchange() const;
  std::vector<Action> LegalActionsInTricksPlaying() const;
  std::vector<Action> LegalActionsInTricksPlayingFollowing() const;

  // checks whether the current player can follow the opening card suit or
  // can't but still has at least one tarok, if the first value is true, the
  // second might be set incorrectly as it is irrelevant
  std::tuple<bool, bool> CanFollowSuitOrCantButHasTarok() const;

  std::vector<Action> TakeSuitFromPlayerCardsInNegativeContracts(
      CardSuit suit) const;
  absl::optional<Action> ActionToBeatInNegativeContracts(CardSuit suit) const;
  std::vector<Action> RemovePagatIfNeeded(
      const std::vector<Action>& actions) const;
  std::vector<Action> TakeSuitFromPlayerCardsInPositiveContracts(
      CardSuit suit) const;

  void DoApplyActionInCardDealing();
  bool AnyPlayerWithoutTaroks() const;
  void AddPrivateCardsToInfoStates();
  void DoApplyActionInBidding(Action action_id);
  bool AllButCurrentPlayerPassedBidding() const;
  void FinishBiddingPhase(Action action_id);
  void DoApplyActionInKingCalling(Action action_id);
  void DoApplyActionInTalonExchange(Action action_id);
  void StartTricksPlayingPhase();
  void DoApplyActionInTricksPlaying(Action action_id);
  void ResolveTrick();
  TrickWinnerAndAction ResolveTrickWinnerAndWinningAction() const;

  // computes which player belongs to the trick_cards_ index as the player
  // who opens the trick always belongs to index 0 within trick_cards_
  Player TrickCardsIndexToPlayer(int index) const;

  std::vector<int> ScoresInKlop() const;
  std::vector<int> ScoresInNormalContracts() const;
  CollectedCardsPerTeam SplitCollectedCardsPerTeams() const;
  int NonValatBonuses(
      const std::vector<Action>& collected_cards,
      const std::vector<Action>& opposite_collected_cards) const;
  std::tuple<bool, bool> CollectedKingsAndOrTrula(
      const std::vector<Action>& collected_cards) const;
  std::vector<int> ScoresInHigherContracts() const;

  void NextPlayer();
  static bool ActionInActions(Action action_id,
                              const std::vector<Action>& actions);
  static void MoveActionFromTo(Action action_id, std::vector<Action>* from,
                               std::vector<Action>* to);
  const Card& ActionToCard(Action action_id) const;
  void AppendToAllInformationStates(const std::string& appendix);
  void AppendToInformationState(Player player, const std::string& appendix);

  std::shared_ptr<const TarokGame> tarok_parent_game_;
  int card_dealing_seed_ = kDefaultSeed;

  GamePhase current_game_phase_ = GamePhase::kCardDealing;
  Player current_player_ = kInvalidPlayer;
  std::vector<Action> talon_;
  std::vector<std::vector<Action>> players_cards_;
  std::vector<Action> players_bids_;
  Player declarer_ = kInvalidPlayer;
  // contract pointed to is managed by the game instance
  const Contract* selected_contract_;
  Action called_king_ = kInvalidAction;
  bool called_king_in_talon_ = false;
  Player declarer_partner_ = kInvalidPlayer;
  std::vector<std::vector<Action>> players_collected_cards_;
  std::vector<Action> trick_cards_;
  Player captured_mond_player_ = kInvalidPlayer;
  std::vector<std::string> players_info_states_;
};

std::ostream& operator<<(std::ostream& os, const GamePhase& game_phase);

std::string GamePhaseToString(const GamePhase& game_phase);

}  // namespace tarok
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TAROK_H_
