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

#ifndef OPEN_SPIEL_GAMES_BRIDGE_H_
#define OPEN_SPIEL_GAMES_BRIDGE_H_

// The full game of contract bridge.
// See https://en.wikipedia.org/wiki/Contract_bridge
// This is played by four players in two partnerships; it consists of a bidding
// phase followed by a play phase. The bidding phase determines the contract for
// the play phase. The contract has four components:
//    - Which of the four players is the 'declarer'. (The first play is made by
//      the player on declarer's left. Declarer's partner (the 'dummy') then
//      places their cards face-up for everyone to see; their plays are chosen
//      by declarer.)
//    - The trump suit (or no-trumps).
//    - The level, i.e. the trick target for the declaring partnership.
//    - Whether the contract is doubled or redoubled (increasing the stakes).
//
// There is then a play phase, in which 13 tricks are allocated between the
// two partnerships. The declaring side gets a positive score if they take
// at least as many tricks as contracted for, otherwise their score is negative.
//
// We support an option to replace the play phase with a perfect-information
// solution (the 'double dummy result' in bridge jargon).
//
// The action space is as follows:
//   0..51   Cards, used for both dealing (chance events) and play;
//   52+     Calls (Pass, Dbl, RDbl, and bids), used during the auction phase.
//
// During the play phase, the dummy's cards are played by the declarer (their
// partner). There will thus be 26 turns for declarer, and 13 turns for each
// of the defenders during the play.

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/bridge/double_dummy_solver/include/dll.h"
#include "open_spiel/games/bridge/bridge_scoring.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace bridge {

inline constexpr int kBiddingActionBase = kNumCards;  // First bidding action.
inline constexpr int kNumObservationTypes = 4;  // Bid, lead, declare, defend
// Because bids always increase, any individual bid can be made at most once.
// Thus for each bid, we only need to track (a) who bid it (if anyone), (b) who
// doubled it (if anyone), and (c) who redoubled it (if anyone).
// We also report the number of passes before the first bid; we could
// equivalently report which player made the first call.
// This is much more compact than storing the auction call-by-call, which
// requires 318 turns * 38 possible calls per turn = 12084 bits (although
// in practice almost all auctions have fewer than 80 calls).
inline constexpr int kAuctionTensorSize =
    kNumPlayers * (1           // Did this player pass before the opening bid?
                   + kNumBids  // Did this player make each bid?
                   + kNumBids  // Did this player double each bid?
                   + kNumBids  // Did this player redouble each bid?
                   ) +
    kNumCards                                  // Our hand
    + kNumVulnerabilities * kNumPartnerships;  // Vulnerability of each side
inline constexpr int kPublicInfoTensorSize =
    kAuctionTensorSize  // The auction
    - kNumCards         // But not any player's cards
    + kNumPlayers;      // Plus trailing passes
inline constexpr int kPlayTensorSize =
    kNumBidLevels              // What the contract is
    + kNumDenominations        // What trumps are
    + kNumOtherCalls           // Undoubled / doubled / redoubled
    + kNumPlayers              // Who declarer is
    + kNumVulnerabilities      // Vulnerability of the declaring side
    + kNumCards                // Our remaining cards
    + kNumCards                // Dummy's remaining cards
    + kNumPlayers * kNumCards  // Cards played to the previous trick
    + kNumPlayers * kNumCards  // Cards played to the current trick
    + kNumTricks               // Number of tricks we have won
    + kNumTricks;              // Number of tricks they have won
inline constexpr int kObservationTensorSize =
    kNumObservationTypes + std::max(kPlayTensorSize, kAuctionTensorSize);
inline constexpr int kMaxAuctionLength =
    kNumBids * (1 + kNumPlayers * 2) + kNumPlayers;
inline constexpr Player kFirstPlayer = 0;
enum class Suit { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };

// State of a single trick.
class Trick {
 public:
  Trick() : Trick{kInvalidPlayer, kNoTrump, 0} {}
  Trick(Player leader, Denomination trumps, int card);
  void Play(Player player, int card);
  Suit LedSuit() const { return led_suit_; }
  Player Winner() const { return winning_player_; }
  Player Leader() const { return leader_; }

 private:
  Denomination trumps_;
  Suit led_suit_;
  Suit winning_suit_;
  int winning_rank_;
  Player leader_;
  Player winning_player_;
};

// State of an in-play game. Can be any phase of the game.
class BridgeState : public State {
 public:
  BridgeState(std::shared_ptr<const Game> game, bool use_double_dummy_result,
              bool is_dealer_vulnerable, bool is_non_dealer_vulnerable);
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
    return std::unique_ptr<State>(new BridgeState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string Serialize() const override;
  void SetDoubleDummyResults(ddTableResults double_dummy_results);

  // If the state is terminal, returns the index of the final contract, into the
  // arrays returned by PossibleFinalContracts and ScoreByContract.
  int ContractIndex() const;

  // Returns a mask indicating which final contracts are possible.
  std::array<bool, kNumContracts> PossibleContracts() const {
    return possible_contracts_;
  }

  // Returns the score for each possible final contract. This is computed once
  // at the start of the deal, so will include scores for contracts which are
  // now impossible.
  std::array<int, kNumContracts> ScoreByContract() const {
    SPIEL_CHECK_TRUE(double_dummy_results_.has_value());
    return score_by_contract_;
  }

  // Returns the double-dummy score for a list of contracts from the point
  // of view of the specified player.
  // Will compute the double-dummy results if needed.
  std::vector<int> ScoreForContracts(int player,
                                     const std::vector<int>& contracts) const;

  // Private information tensor per player.
  std::vector<double> PrivateObservationTensor(Player player) const;

  // Public information.
  std::vector<double> PublicObservationTensor() const;

  // Current phase.
  int CurrentPhase() const { return static_cast<int>(phase_); }

 protected:
  void DoApplyAction(Action action) override;

 private:
  enum class Phase { kDeal, kAuction, kPlay, kGameOver };

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealAction(int card);
  void ApplyBiddingAction(int call);
  void ApplyPlayAction(int card);
  void ComputeDoubleDummyTricks() const;
  void ComputeScoreByContract() const;
  void ScoreUp();
  Trick& CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }
  const Trick& CurrentTrick() const {
    return tricks_[num_cards_played_ / kNumPlayers];
  }
  std::array<absl::optional<Player>, kNumCards> OriginalDeal() const;
  std::string FormatDeal() const;
  std::string FormatVulnerability() const;
  std::string FormatAuction(bool trailing_query) const;
  std::string FormatPlay() const;
  std::string FormatResult() const;

  const bool use_double_dummy_result_;
  const bool is_vulnerable_[kNumPartnerships];

  int num_passes_ = 0;  // Number of consecutive passes since the last non-pass.
  int num_declarer_tricks_ = 0;
  int num_cards_played_ = 0;
  Player current_player_ = 0;  // During the play phase, the hand to play.
  Phase phase_ = Phase::kDeal;
  Contract contract_{0};
  std::array<std::array<absl::optional<Player>, kNumDenominations>,
             kNumPartnerships>
      first_bidder_{};
  std::array<Trick, kNumTricks> tricks_{};
  std::vector<double> returns_ = std::vector<double>(kNumPlayers);
  std::array<absl::optional<Player>, kNumCards> holder_{};
  mutable absl::optional<ddTableResults> double_dummy_results_{};
  std::array<bool, kNumContracts> possible_contracts_;
  mutable std::array<int, kNumContracts> score_by_contract_;
};

class BridgeGame : public Game {
 public:
  explicit BridgeGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return kBiddingActionBase + kNumCalls;
  }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new BridgeState(shared_from_this(), UseDoubleDummyResult(),
                        IsDealerVulnerable(), IsNonDealerVulnerable()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -kMaxScore; }
  double MaxUtility() const override { return kMaxScore; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
  int MaxGameLength() const override {
    return UseDoubleDummyResult() ? kMaxAuctionLength
                                  : kMaxAuctionLength + kNumCards;
  }
  int MaxChanceNodesInHistory() const override { return kNumCards; }

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

  // How many contracts there are (including declarer and double status).
  int NumPossibleContracts() const { return kNumContracts; }

  // A string representation of a contract.
  std::string ContractString(int index) const;

  // Extra observation tensors.
  int PrivateObservationTensorSize() const { return kNumCards; }
  int PublicObservationTensorSize() const { return kPublicInfoTensorSize; }

 private:
  bool UseDoubleDummyResult() const {
    return ParameterValue<bool>("use_double_dummy_result", true);
  }
  bool IsDealerVulnerable() const {
    return ParameterValue<bool>("dealer_vul", false);
  }
  bool IsNonDealerVulnerable() const {
    return ParameterValue<bool>("non_dealer_vul", false);
  }
};

}  // namespace bridge
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BRIDGE_H_
