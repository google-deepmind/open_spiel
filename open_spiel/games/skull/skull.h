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

#ifndef OPEN_SPIEL_GAMES_SKULL_H_
#define OPEN_SPIEL_GAMES_SKULL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Skull & Roses
// https://en.wikipedia.org/wiki/Skull_(card_game)
//
// A bluffing and deduction game for 3-6 players. Each player starts with
// a hand of Rose and Skull cards. Players take turns placing cards face-down
// on their mats, then bidding on how many total cards they can flip without
// revealing a Skull. The last remaining bidder becomes the challenger and
// must flip that many cards, starting from their own mat. Flipping a Skull
// costs the challenger a card permanently; flipping the full bid with only
// Roses wins the round. First player to win two rounds wins the game.

namespace open_spiel {
namespace skull {

// Parameter Defaults.
constexpr int kDefaultInitialHandSize = 4;
constexpr int kDefaultPlayers = 4;
constexpr int kDefaultWinningScore = 2;
constexpr bool kDefaultObserveDerivedInfo = true;
constexpr bool kDefaultEgocentric = true;
constexpr bool kDefaultPartialRecall = true; // Meaning: Infostate only has
                                             // exact history for current round

// Used for Absl::InlinedVector as well.
constexpr int kMaximumHandSize = 4;
constexpr int kMaxPlayers = 6;
constexpr int kMinPlayers = 3;

enum class GamePhase {
  kPlacement,
  kBidding,
  kFlipping,
  kCardLoss,
};
constexpr int kGamePhaseCount = 4;

enum class CardType {
  kRose = 0,
  kSkull = 1,
};
constexpr int kCardTypeCount = 2; // TODO: Style question: Should Counts be
                                  // included inside the end of enums,
                                  // or seperate like this?

// ---------------------------------------------------------------------------
// Action encoding
//
// The flat action space has this layout (the offsets are computed at runtime
// from num_players and max_total_cards = num_players * MaxHandSize()):
//
// [0]                              DisacrdRose       (Discard phase)
// [1]                              DiscardSkull      (Discard phase)
// [2]                              PlaceRose         (Placement phase)
// [3]                              PlaceSkull        (Placement phase)
// [4]                              Pass              (Bidding phase)
// [kBidBase +1, kBidBase + max_total_cards]
//                                  Bid N (>=1) cards (Placement/Bidding Phase)
// [kFlipBase, kFlipBase + num_players - 1]
//                                  Flip player #N's card (Flipping Phase)
//
// Total actions = 5 + max_bid + players
//               = 5 + (max_hand * players) + players
//
// ---------------------------------------------------------------------------

// These are made as 0 and 1 as they are taken also by chance players, and
// chance players actionspace must also begin at 0 according to the api, so
// having chance action and player action map to the same number makes for a
// simpler implemntation
constexpr Action kActionDiscardRose = 0;
constexpr Action kActionDiscardSkull = 1;

constexpr Action kActionPlaceRose = 2;
constexpr Action kActionPlaceSkull = 3;
constexpr Action kActionPass = 4;
constexpr Action kActionBidBase = kActionPass;

class SkullGame;
class SkullObserver;

class SkullState : public State {
public:
  explicit SkullState(std::shared_ptr<const Game> game);
  SkullState(const SkullState &) = default;

  Player CurrentPlayer() const override {
    // double checking, because spiel.h mentions it explicitly
    SPIEL_DCHECK_TRUE(!IsTerminal() || current_player_ == kTerminalPlayerId);
    return current_player_;
  }

  static constexpr std::string CardTypeToString(CardType c);
  static constexpr std::string PhaseToString(GamePhase p);
  std::string HandToString(Player p) const;
  std::string StackToString(Player p, bool full_info) const;
  std::string DerivedPublicInfoString(Player p) const;
  std::string FormatActionString(Player player, Action action,
                                 bool make_short) const;

  std::string ActionToShortString(Player player, Action action) const;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override { return winner_ != kInvalidPlayer; }
  std::vector<double> Returns() const override;

  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State>
  ResampleFromInfostate(int player_id, std::function<double()> rng) const;
  std::vector<Action> ActionsConsistentWithInformationFrom(Action action) const;
  std::unique_ptr<State> Clone() const override;

  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalBids() const;
  std::vector<Action> LegalActions() const override;

  GamePhase current_phase() const { return current_phase_; }
  int current_bid() const { return current_bid_; }
  Player challenger() const { return challenger_; }
  int total_cards_flipped() const { return total_cards_flipped_; }
  int score(Player p) const { return scores_[p]; }
  int stack_size(Player p) const { return static_cast<int>(stacks_[p].size()); }
  int hand_size(Player p) const { return static_cast<int>(hands_[p].size()); }
  bool is_active(Player p) const { return hand_size(p) > 0; }
  int flipped_stack_depth(Player p) const { return depth_flipped_[p]; }

  // (derived) public information, see `UpdateDerivedHandInfo()`
  bool known_has_rose(Player p) const { return known_has_rose_[p]; }
  bool known_has_skull(Player p) const { return known_has_skull_[p]; }
  bool known_has_only_roses(Player p) const { return known_has_only_roses_[p]; }
  bool known_has_only_skull(Player p) const { return known_has_only_skull_[p]; }

  // (private info)
  bool has_rose(Player p) const;
  bool has_skull(Player p) const;
  bool has_only_roses(Player p) const { return has_rose(p) && !has_skull(p); }
  bool has_only_skull(Player p) const { return has_skull(p) && !has_rose(p); }
  Action highest_safe_bid_or_pass(Player p) const;

  int total_cards_on_table() const;
  int num_active_players() const;

  size_t BeginningOfMostRecentRound() const { return start_of_last_round_idx_; }

protected:
  void DoApplyAction(Action action) override;

private:
  friend class SkullObserver;
  const absl::InlinedVector<CardType, kMaximumHandSize> InitialHand() const;
  void StartNewRound();
  void AdvanceToNextPlayer();
  void ApplyBidAction(Action action);
  void BeginFlipping();
  void ApplyFlipAction(Player flip_target);
  void CheckForWin();
  void DiscardCard(Player p, CardType type);
  void UpdateDerivedHandInfo(Player p);

  // action encoding helpers
  const bool IsActionBid(Action action) const {
    return action > kActionBidBase &&
           action <= kActionBidBase + max_total_cards_;
  }

  const bool IsActionFlip(Action action) const {
    return action >= flip_base_ && action < flip_base_ + num_players_;
  }

  const Action FlipTargetFromAction(Action action) const {
    return action - flip_base_;
  }

  // These are all derived from SkullGame, put here for easier access.
  const int max_hand_size_;
  const int wins_needed_;
  const Action flip_base_;
  const int max_total_cards_;

  // hands_[p]    : cards in player p's hand (private to p).
  // stacks_[p]   : cards placed face-down on player p's mat, bottom-to-top.
  // flipped_[p] :  number of cards flipped per pile
  // note cards aren't removed from hand vector when in stack for `has_skull()`.
  absl::InlinedVector<absl::InlinedVector<CardType, kMaximumHandSize>,
                      kMaxPlayers>
      hands_;
  absl::InlinedVector<absl::InlinedVector<CardType, kMaximumHandSize>,
                      kMaxPlayers>
      stacks_;
  absl::InlinedVector<int, kMaxPlayers> depth_flipped_;
  absl::InlinedVector<int, kMaxPlayers> scores_;

  // these are true when it could have been derived from public information
  // (when all cards they had were visible)
  absl::InlinedVector<bool, kMaxPlayers> known_has_rose_;
  absl::InlinedVector<bool, kMaxPlayers> known_has_only_roses_;
  absl::InlinedVector<bool, kMaxPlayers> known_has_skull_;
  absl::InlinedVector<bool, kMaxPlayers> known_has_only_skull_;

  GamePhase current_phase_;
  Player current_player_;
  Player first_player_;
  Player challenger_;

  Action current_bid_;      // Highest bid so far this round; 0 before any bid.
  int total_cards_flipped_; // Cards flipped by the challenger so far.

  int winner_;                     // kInvalidPlayer until the game ends.
  size_t start_of_last_round_idx_; // for partial recall, index into history
};

class SkullGame : public Game {
public:
  explicit SkullGame(const GameParameters &params);

  int NumDistinctActions() const override { return flip_base_ + num_players_; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 2; }
  int NumPlayers() const override { return num_players_; }
  int MaxGameLength() const override { return max_game_length_; }
  static constexpr int CalcMaxGameLength(int num_players, int max_hand_size,
                                         int wins_needed);

  int MaxChanceNodesInHistory() const override {
    return num_players_ * max_hand_size_;
  }

  double MinUtility() const override { return -wins_needed_; }
  double MaxUtility() const override {
    return (num_players_ - 1) * wins_needed_;
  }
  absl::optional<double> UtilitySum() const override { return 0; }

  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> TensorShapeFromIIGObsType(IIGObservationType iig_obs_type,
                                             bool include_public_derived_info,
                                             bool egocentric,
                                             bool partial_recall) const;

  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;
  std::shared_ptr<Observer> public_observer_;
  std::shared_ptr<Observer> private_observer_;
  std::shared_ptr<Observer> sees_all_private_observer_;
  const GameParameters obs_params;
  const bool obs_public_derived_info_;
  const bool obs_egocentric_;
  const bool obs_partial_recall_;

  /*
   * TODO: Implementing this corrupts the heap?
    std::shared_ptr<Observer>
    MakeObserver(absl::optional<IIGObservationType> iig_obs_type,
                 const GameParameters &params) const override;
  */

  const int MaxHandSize() const { return max_hand_size_; }
  const int WinsNeeded() const { return wins_needed_; }
  const int MaxTotalCards() const { return num_players_ * max_hand_size_; }

  const Action flip_base() const { return flip_base_; } // action integer offset

private:
  const int num_players_;
  const int max_hand_size_;
  const int wins_needed_;
  const Action flip_base_;
  const int max_game_length_; // cached, see CalcMaxGameLength()
};

class SkullObserver : public Observer {
public:
  explicit SkullObserver(IIGObservationType iig_obs_type,
                         bool include_public_derived_info, bool egocentric,
                         bool partial_recall);

  void WriteTensor(const State &observed_state, int player,
                   Allocator *allocator) const override;

  std::string StringFrom(const State &observed_state,
                         int player) const override;

private:
  inline void NextPlayer(int *count, Player *player, int num_players) const;
  inline Player AbsoluteToRelativePlayer(Player player, Player obs_player,
                                         int num_players) const;

  void WriteGlobalPublicInfo(Player obs_player, const SkullState &state,
                             Allocator *allocator) const;
  void WritePerPlayerPublicInfo(Player obs_player, bool skip_observing_player,
                                const SkullState &state,
                                Allocator *allocator) const;
  void WriteOnePlayerPrivateInfo(Player player, const SkullState &state,
                                 Allocator *allocator) const;
  void WriteAllPlayersPrivateInfo(Player obs_player, const SkullState &state,
                                  Allocator *allocator) const;
  void WritePartialHistory(Player obs_player, const SkullState &state,
                           Allocator *allocator) const;
  void WriteHistory(Player obs_player, const SkullState &state,
                    Allocator *allocator) const;

  int ObservedActionIndex(Player acting_player, Action action,
                          const SkullState &state, Player obs_player) const;

  inline const std::string ObservedActionString(Player acting_player,
                                                Action aciton,
                                                const SkullState &state,
                                                Player obs_player) const;
  IIGObservationType iig_obs_type_;
  const bool include_public_derived_info_;
  const bool egocentric_;
  const bool partial_recall_;
};

} // namespace skull
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_SKULL_H_
