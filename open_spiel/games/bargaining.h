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

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/spiel.h"

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
//     "instances_file"     string  The file containing the boards (default: "")

namespace open_spiel {
namespace bargaining {

constexpr int kNumItemTypes = 3;
constexpr int kPoolMinNumItems = 5;
constexpr int kPoolMaxNumItems = 7;
constexpr int kTotalValueAllItems = 10;
constexpr int kNumPlayers = 2;
constexpr int kMaxTurns = 10;

// Default 10-instance database used for tests. See
// bargaining_instance_generator.cc to create your own.
// Format is: pool items, p1 values, p2 values.
constexpr const char* kDefaultInstancesString =
    "1,2,3 8,1,0 4,0,2\n"
    "1,4,1 4,1,2 2,2,0\n"
    "2,2,1 1,1,6 0,4,2\n"
    "1,4,1 9,0,1 2,2,0\n"
    "1,4,1 5,1,1 0,1,6\n"
    "4,1,1 2,1,1 1,0,6\n"
    "3,1,1 1,4,3 0,2,8\n"
    "1,1,3 0,1,3 1,3,2\n"
    "1,3,1 2,2,2 10,0,0\n"
    "1,2,2 2,3,1 4,0,3\n";

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

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool IsLegalOffer(const Offer& offer) const;

  Player cur_player_;
  bool agreement_reached_;
  const BargainingGame* parent_game_;
  Instance instance_;
  std::vector<Offer> offers_;
};

class BargainingGame : public Game {
 public:
  explicit BargainingGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BargainingState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override { return all_instances_.size(); }
  std::string ActionToString(Player player, Action move_id) const override;

  int MaxGameLength() const override { return kMaxTurns; }
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return kNumPlayers; }
  double MaxUtility() const override { return kTotalValueAllItems; }
  double MinUtility() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;

  const std::vector<Instance>& AllInstances() const { return all_instances_; }
  const std::vector<Offer>& AllOffers() const { return all_offers_; }
  const Instance& GetInstance(int num) const { return all_instances_[num]; }
  const Offer& GetOffer(int num) const { return all_offers_[num]; }

 private:
  void ParseInstancesFile(const std::string& filename);
  void ParseInstancesString(const std::string& instances_string);
  void CreateOffers();

  std::vector<Instance> all_instances_;
  std::vector<Offer> all_offers_;
};

}  // namespace bargaining
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BARGAINING_H_
