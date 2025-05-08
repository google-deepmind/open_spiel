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

#ifndef OPEN_SPIEL_GAMES_BARGAINING_H_
#define OPEN_SPIEL_GAMES_BARGAINING_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/game_parameters.h"

// A simple multi-issue bargaining game, based on [1,2]. The rules are based
// on the description of Section 2.2 of [1]:
//
// "Each agent is given a different randomly generated value function, which
//  gives a non-negative value for each item. The value functions are
//  constrained so that: (1) the total value for a user of all items is 10;
//  (2) each item has non-zero value to at least one user; and (3) some items
//  have nonzero value to both users. These constraints enforce that it is not
//  possible for both agents to receive a maximum score, and that no item is
//  worthless to both agents, so the negotiation will be competitive. After 10
//  turns, we allow agents the option to complete the negotiation with no
//  agreement, which is worth 0 points to both users. We use 3 item types
//  (books, hats, balls), and between 5 and 7 total items in the pool."
//
// [1] Lewis et al., Deal or no deal? End-to-end learning of negotiation
//     dialogues, 2017. https://arxiv.org/abs/1706.05125
// [2] David DeVault, Johnathan Mell, and Jonathan Gratch.
//     2015. Toward Natural Turn-taking in a Virtual Human Negotiation Agent
//
// Parameters:
//     "instances_file" string    The file containing the boards (default: "")
//     "discount"       double    Discount factor multiplied each turn after
//                                turn 2, applied to (multiplied to reduce) the
//                                returns (default = 1.0).
//     "max_turns"      integer   Maximum total turns before the game ends
//                                (default = 10).
//     "prob_end"       double    Probability of the game ending after each
//                                action (only after each player has taken
//                                one turn each)  (default = 0.0).

namespace open_spiel {
namespace bargaining {

constexpr int kNumItemTypes = 3;
constexpr int kPoolMinNumItems = 5;
constexpr int kPoolMaxNumItems = 7;
constexpr int kTotalValueAllItems = 10;
constexpr int kNumPlayers = 2;
constexpr double kDefaultDiscount = 1.0;
constexpr int kDefaultMaxTurns = 10;
constexpr double kDefaultProbEnd = 0.0;
constexpr int kDefaultNumInstances = 1000;
// Default 1000-instance database. See
// bargaining_instances1000.cc to create your own.
// Format is: pool items, p1 values, p2 values.
const char* BargainingInstances1000();

struct Instance {
  std::vector<std::vector<int>> values;
  std::vector<int> pool;
  Instance()
      : values({std::vector<int>(kNumItemTypes, 0),
                std::vector<int>(kNumItemTypes, 0)}),
        pool(kNumItemTypes, 0) {}
  std::string ToString() const;
  std::string ToPrettyString() const;
};

struct Offer {
  std::vector<int> quantities;
  Offer() : quantities(kNumItemTypes, 0) {}
  Offer(const std::vector<int>& _quantities) : quantities(_quantities) {}
  std::string ToString() const;
};

class BargainingGame;  // Forward definition necessary for parent pointer.

class BargainingState : public State {
 public:
  BargainingState(std::shared_ptr<const Game> game);
  BargainingState(const BargainingState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string InformationStateString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;

  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  // Extra methods not part of the general API.
  Instance GetInstance() const { return instance_; }
  void SetInstance(Instance instance);

  std::vector<Offer> Offers() const { return offers_; }

  Action AgreeAction() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool IsLegalOffer(const Offer& offer) const;

  Player cur_player_;
  bool agreement_reached_;
  const BargainingGame* parent_game_;
  Instance instance_;
  std::vector<Offer> offers_;
  Player next_player_;
  double discount_;
  bool game_ended_;
};

class BargainingGame : public Game {
 public:
  explicit BargainingGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BargainingState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return all_instances_.size() + 2; }
  std::string ActionToString(Player player, Action move_id) const override;

  int MaxGameLength() const override { return max_turns_; }
  int MaxChanceNodesInHistory() const override { return 1 + (max_turns_ - 2); }

  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override { return kTotalValueAllItems; }
  double MinUtility() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;

  int max_turns() const { return max_turns_; }
  double discount() const { return discount_; }
  double prob_end() const { return prob_end_; }

  Action ContinueOutcome() const { return all_instances_.size(); }
  Action EndOutcome() const { return all_instances_.size() + 1; }

  const std::vector<Instance>& AllInstances() const { return all_instances_; }
  const std::vector<Offer>& AllOffers() const { return all_offers_; }
  const Instance& GetInstance(int num) const { return all_instances_[num]; }
  const Offer& GetOffer(int num) const { return all_offers_[num]; }
  std::pair<Offer, Action> GetOfferByQuantities(
      const std::vector<int>& quantities) const;
  int GetInstanceIndex(const Instance& instance) const {
    if (!instance_map_.contains(instance.ToString())) {
      return -1;
    }
    return instance_map_.at(instance.ToString());
  }
  int GetOfferIndex(const Offer& offer) const {
    if (!offer_map_.contains(offer.ToString())) {
      return -1;
    }
    return offer_map_.at(offer.ToString());
  }
  std::vector<std::vector<int>> GetPossibleOpponentValues(
      int player_id, const std::vector<int>& pool,
      const std::vector<int>& values) const;

 private:
  void ParseInstancesFile(const std::string& filename);
  void ParseInstancesString(const std::string& instances_string);
  void CreateOffers();

  std::vector<Instance> all_instances_;
  std::vector<Offer> all_offers_;
  absl::flat_hash_map<std::string, int> offer_map_;
  absl::flat_hash_map<std::string, int> instance_map_;
  absl::flat_hash_map<std::string, std::vector<std::vector<int>>>
      possible_opponent_values_;
  const int max_turns_;
  const double discount_;
  const double prob_end_;
};

}  // namespace bargaining
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BARGAINING_H_
