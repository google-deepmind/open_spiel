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

#ifndef OPEN_SPIEL_GAMES_SPADES_H_
#define OPEN_SPIEL_GAMES_SPADES_H_

// The full game of partnership spades.
// See https://dkmgames.com/CardSharp/Spades/SpadesHelp.php
// This is played by four players in two partnerships; it consists of a bidding
// phase followed by a play phase. The bidding phase determines the contracts
// for the play phase. The contract consists of:
//    - Each player bidding how many tricks they can take.
//    - If a player bids 'Nil' (meaning '0'), then they have a special condition
//    for points
//      based on whether they can avoid taking any tricks.
//
// There is then a play phase, in which 13 tricks are allocated between the
// two partnerships. Each partnership gains 10 times their combined contract
// if the partners are able to collectively take at least as many tricks as that
// combined contract, otherwise the partnership loses 10 times their combined
// contract.
//
// Any tricks taken in excess of a partnership's combined contract are worth 1
// point and considered a 'bag' - for every 10 bags collected over the course of
// the game, the partnership is penalized 100 points.
//
// In the case of a Nil bid, if that partner avoids taking any tricks during the
// round, the partnership gains a 100 point bonus. Conversely, if that partner
// takes any tricks, the partnership will lose 100 points (but these tricks
// still count toward the other partner's contract).
//
// The action space is as follows:
//   0..51   Cards, used for both dealing (chance events) and play;
//   52+     Bids (Nil, 1-13) used during the bidding phase.
//
// During the bidding phase, every player will have 1 turn for making a bid.
// During the play phase, every play will have 13 turns for playing a card.

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "open_spiel/games/spades/spades_scoring.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace spades {

inline constexpr int kBiddingActionBase = kNumCards;  // First bidding action.
inline constexpr int kAuctionTensorSize =
    kNumPlayers * kNumBids + kNumCards;   // Our hand
inline constexpr int kPhaseInfoSize = 2;  // Bidding (auction) and Playing
inline constexpr int kPublicInfoTensorSize =
    kAuctionTensorSize  // The auction
    - kNumCards;        // But not any player's cards
inline constexpr int kMaxAuctionLength = 4;
inline constexpr Player kFirstPlayer = 0;
enum class Suit { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };

// State of a single trick.
class Trick {
 public:
  Trick() : Trick{kInvalidPlayer, 0} {}
  Trick(Player leader, int card);
  void Play(Player player, int card);
  Suit LedSuit() const { return led_suit_; }
  Player Winner() const { return winning_player_; }
  Player Leader() const { return leader_; }

 private:
  Suit led_suit_;
  Suit winning_suit_;
  int winning_rank_;
  Player leader_;
  Player winning_player_;
};

// State of an in-play game. Can be any phase of the game.
class SpadesState : public State {
 public:
  SpadesState(std::shared_ptr<const Game> game, bool use_mercy_rule,
              int mercy_threshold, int win_threshold, int win_or_loss_bonus,
              int num_tricks);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override { return returns_; }
  std::string ObservationString(Player player) const override;
  void WriteObservationTensor(Player player, absl::Span<float> values) const;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new SpadesState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string Serialize() const override;

  // If the state is terminal, returns the indexes of the final contracts, into
  // the arrays returned by PossibleFinalContracts and ScoreByContract.
  std::array<int, kNumPlayers> ContractIndexes() const;

  // Returns a mask indicating which final contracts are possible.
  std::array<bool, kNumContracts> PossibleContracts() const {
    return possible_contracts_;
  }

  // Private information tensor per player.
  std::vector<double> PrivateObservationTensor(Player player) const;

  // Public information.
  std::vector<double> PublicObservationTensor() const;

  // Current phase.
  int CurrentPhase() const { return static_cast<int>(phase_); }

  // Current overall partnership scores
  std::array<int, kNumPartnerships> GetCurrentScores() const {
    return current_scores_;
  }

  // Set partnership scores
  void SetCurrentScores(const std::array<int, kNumPartnerships>& new_scores) {
    current_scores_ = new_scores;
  }

  // Indicates if overall game is over (did a partnership meet win/lose
  // condition)
  bool IsGameOver() const { return is_game_over_; }

  // Manually set the current player (used to specify starting player)
  void SetCurrentPlayer(const int current_player) {
    current_player_ = current_player;
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  enum class Phase { kDeal, kAuction, kPlay, kGameOver };

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealAction(int card);
  void ApplyBiddingAction(int bid);
  void ApplyPlayAction(int card);

  void ScoreUp();
  Trick& CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }
  const Trick& CurrentTrick() const {
    return tricks_[num_cards_played_ / kNumPlayers];
  }
  std::array<absl::optional<Player>, kNumCards> OriginalDeal() const;
  std::string FormatDeal() const;
  std::string FormatAuction(bool trailing_query) const;
  std::string FormatPlay() const;
  std::string FormatResult() const;

  const bool use_mercy_rule_;
  const int mercy_threshold_;
  const int win_threshold_;
  const int win_or_loss_bonus_;
  const int num_tricks_;

  std::array<int, kNumPartnerships> current_scores_ = {0, 0};
  bool is_game_over_ = false;
  std::array<int, kNumPlayers> num_player_tricks_ = {0, 0, 0, 0};
  int num_cards_played_ = 0;
  Player current_player_ = 0;  // During the play phase, the hand to play.
  Phase phase_ = Phase::kDeal;
  std::array<int, kNumPlayers> contracts_ = {-1, -1, -1, -1};
  std::array<Trick, kNumTricks> tricks_{};
  std::vector<double> returns_ = std::vector<double>(kNumPlayers);
  std::array<absl::optional<Player>, kNumCards> holder_{};
  std::array<bool, kNumContracts>
      possible_contracts_;  // Array of bids 0-13 for each player (so 4x14 size)
  bool is_spades_broken_ = false;
};

class SpadesGame : public Game {
 public:
  explicit SpadesGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return kBiddingActionBase + kNumBids;
  }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new SpadesState(shared_from_this(), UseMercyRule(), MercyThreshold(),
                        WinThreshold(), WinOrLossBonus(), NumTricks()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -(kMaxScore + WinOrLossBonus()); }
  double MaxUtility() const override { return kMaxScore + WinOrLossBonus(); }

  static int GetPlayTensorSize(int num_tricks) {
    return kNumBids * kNumPlayers  // What each player's contract is
           + kNumCards             // Our remaining cards
           + num_tricks * kNumPlayers * kNumCards  // Number of played tricks
           + kNumTricks * kNumPlayers;  // Number of tricks each player has won
  }

  std::vector<int> ObservationTensorShape() const override {
    return {kPhaseInfoSize +
            std::max(GetPlayTensorSize(NumTricks()), kAuctionTensorSize)};
  }

  int MaxGameLength() const override { return kMaxAuctionLength + kNumCards; }
  int MaxChanceNodesInHistory() const override { return kNumCards; }

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

  // How many contracts there are.
  int NumPossibleContracts() const { return kNumContracts; }

  // A string representation of a contract.
  std::string ContractString(int bid) const;

  // Extra observation tensors.
  int PrivateObservationTensorSize() const { return kNumCards; }
  int PublicObservationTensorSize() const { return kPublicInfoTensorSize; }

 private:
  bool UseMercyRule() const {
    return ParameterValue<bool>("use_mercy_rule", true);
  }

  int MercyThreshold() const {
    return ParameterValue<int>("mercy_threshold", -350);
  }

  int WinThreshold() const { return ParameterValue<int>("win_threshold", 500); }

  int WinOrLossBonus() const {
    return ParameterValue<int>("win_or_loss_bonus", 200);
  }

  int NumTricks() const { return ParameterValue<int>("num_tricks", 2); }
};

}  // namespace spades
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SPADES_H_
