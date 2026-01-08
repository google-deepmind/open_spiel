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

#ifndef OPEN_SPIEL_GAMES_NEGOTIATION_H_
#define OPEN_SPIEL_GAMES_NEGOTIATION_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple negotiation game where agents propose splits of a group of items,
// until the maximum number of turns runs out or an offer is accepted.

// This game is inspired by the following papers:
//   - DeVault et al., Toward natural turn-taking in a virtual human negotiation
//     agent, 2015.
//   - Lewis et al., Deal or no deal? End-to-end learning of negotiation
//     dialogues, 2017.
//   - Cao et al., Emergent Communication through Negotiation, 2018.
//     https://arxiv.org/abs/1804.03980
//
// We use the specific description in Cao et al. 2018. However, we choose
// default settings that lead to a smaller game since the values used in the
// paper (utterance_dim = 6, num_symbols = 10) could lead to legal action sizes
// of 2.17 * 10^8.
// TODO(author5): fix this restriction by either (i) adding support for legal
// action iterators rather than lists, or (ii) supporting structured actions,
// (or both!).
//
// Parameters:
//     "enable_proposals"  bool    open the proposal channel (default = true)
//     "enable_utterances" bool    open the linguistic channel (default = true)
//     "num_items"         int     number of distinct items (default = 5)
//     "num_symbols"       int     number of distinct symbols (default = 5)
//     "rng_seed"          int     seed for the random number generator
//                                 (default -1 = not set, seeded by clock)
//     "utterance_dim"     int     dimensionality of the utterances, i.e. number
//                                 of symbols per utterance (default = 3)

namespace open_spiel {
namespace negotiation {

inline constexpr bool kDefaultEnableProposals = true;
inline constexpr bool kDefaultEnableUtterances = true;
inline constexpr int kDefaultNumSymbols = 5;
inline constexpr int kDefaultUtteranceDim = 3;
inline constexpr int kMaxQuantity = 5;
inline constexpr int kMaxValue = 10;
inline constexpr int kMaxSteps = 10;
inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultNumItems = 3;
inline constexpr int kDefaultSeed = -1;

// The utterances and proposals are done in separate repeated turns by the same
// agent. This enum is used to keep track what the type of turn it is.
enum class TurnType { kUtterance, kProposal };

class NegotiationGame;

class NegotiationState : public State {
 public:
  NegotiationState(std::shared_ptr<const Game> game);
  NegotiationState(const NegotiationState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  const std::vector<int>& ItemPool() const { return item_pool_; }
  const std::vector<std::vector<int>>& AgentUtils() const {
    return agent_utils_;
  }
  std::vector<int>& ItemPool() { return item_pool_; }
  std::vector<std::vector<int>>& AgentUtils() { return agent_utils_; }
  void SetCurrentPlayer(Player p) { cur_player_ = p; }
  int MaxSteps() const { return max_steps_; }
  void SetMaxSteps(int max_steps) { max_steps_ = max_steps; }
  std::string Serialize() const override;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  // Initialize pool of items and agent utilities.
  void DetermineItemPoolAndUtilities();

  // Initialize state variables to start an episode.
  void InitializeEpisode();

  // Get the next valid proposal; returns false when there are no more.
  bool NextProposal(std::vector<int>* proposal) const;

  // Action encoding and decoding helpers. Actions are encoded as follows:
  // the first values { 0, 1, ... , NumDistinctProposals() - 1 } are reserved
  // for proposals, encoded in the usual way (fixed base). The next
  // NumDistinctUtterances() values are reserved for utterances, so these begin
  // at an offset of NumDistinctProposals().
  Action EncodeProposal(const std::vector<int>& proposal) const;
  Action EncodeUtterance(const std::vector<int>& utterance) const;
  std::vector<int> DecodeProposal(int encoded_proposal) const;
  std::vector<int> DecodeUtterance(int encoded_utterance) const;

  std::vector<int> DecodeInteger(int encoded_value, int dimensions,
                                 int num_digit_values) const;
  int EncodeInteger(const std::vector<int>& container,
                    int num_digit_values) const;

  const NegotiationGame& parent_game_;
  bool enable_proposals_;
  bool enable_utterances_;
  int num_items_;
  int num_symbols_;
  int utterance_dim_;
  int num_steps_;
  int max_steps_;
  bool agreement_reached_;
  Player cur_player_;
  TurnType turn_type_;

  // Current quantities of items 0, 1, 2..
  std::vector<int> item_pool_;

  // Utilities for each item of each player: agent_utils_[i][j] represents
  // player i's utility for the jth item.
  std::vector<std::vector<int>> agent_utils_;

  // History of proposals.
  std::vector<std::vector<int>> proposals_;

  // History of utterances.
  std::vector<std::vector<int>> utterances_;
};

class NegotiationGame : public Game {
 public:
  explicit NegotiationGame(const GameParameters& params);
  explicit NegotiationGame(const NegotiationGame& other);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new NegotiationState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return 1; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override;
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override {
    return kMaxQuantity * kMaxValue * num_items_;
  }
  double MinUtility() const override { return -MaxUtility(); }
  std::vector<int> ObservationTensorShape() const override;

  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;
  std::string GetRNGState() const;
  void SetRNGState(const std::string& rng_state) const;

  std::mt19937* RNG() const { return rng_.get(); }
  bool EnableProposals() const { return enable_proposals_; }
  bool EnableUtterances() const { return enable_utterances_; }
  int NumItems() const { return num_items_; }
  int NumSymbols() const { return num_symbols_; }
  int UtteranceDim() const { return utterance_dim_; }

  int NumDistinctUtterances() const;
  int NumDistinctProposals() const;

  const std::vector<Action>& LegalUtterances() const {
    return legal_utterances_;
  }

 private:
  void ConstructLegalUtterances();

  bool enable_proposals_;
  bool enable_utterances_;
  int num_items_;
  int num_symbols_;
  int utterance_dim_;
  int seed_;
  std::vector<Action> legal_utterances_;
  mutable std::unique_ptr<std::mt19937> rng_;
};

}  // namespace negotiation
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_NEGOTIATION_H_
