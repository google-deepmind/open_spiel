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

#ifndef OPEN_SPIEL_PUBLIC_STATES_GAMES_KUHN_POKER_H_
#define OPEN_SPIEL_PUBLIC_STATES_GAMES_KUHN_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/public_states/public_states.h"

// This is a public state API version of Kuhn Poker:
// http://en.wikipedia.org/wiki/Kuhn_poker
//
// While public state API describes imperfect recall abstractions, these
// actually coincide with perfect recall on this game.
//
// There is a visualization of world/public/private trees available in [1]
// for the two-player variant.
//
// This implementation works for N players (N >= 2).
//
// The multiplayer (n>2) version is the one described in
// http://mlanctot.info/files/papers/aamas14sfrd-cfr-kuhn.pdf
//
// [1] https://arxiv.org/abs/1906.11110
// TODO: Base Game API observations description.

namespace open_spiel {
namespace public_states {
namespace kuhn_poker {


class KuhnGameWithPublicStates : public GameWithPublicStates {
 public:
  KuhnGameWithPublicStates(std::shared_ptr<const Game> base_game);
  std::unique_ptr<PublicState> NewInitialPublicState() const override;
  std::vector<ReachProbs> NewInitialReachProbs() const override;
  int NumPublicFeatures() const override;
  std::vector<int> MaxDistinctPrivateInformationsCount() const override;
};

class KuhnPrivateInformation : public PrivateInformation {
 public:
  KuhnPrivateInformation(std::shared_ptr<const Game> base_game, Player player,
                         int player_card);
  Player GetPlayer() const override;
  int GetPlayerCard() const;
  int ReachProbsIndex() const override;
  int NetworkIndex() const override;
  bool IsStateCompatible(const State& state) const override;
  std::string ToString() const override;
  std::unique_ptr<PrivateInformation> Clone() const override;
  std::string Serialize() const override;
  bool operator==(const PrivateInformation& other) const override;

 private:
  const Player player_;
  const int player_card_;
};

class KuhnPublicState : public PublicState {
 public:
  explicit KuhnPublicState(
      std::shared_ptr<const GameWithPublicStates> public_game);
  KuhnPublicState(const KuhnPublicState&) = default;
  KuhnPublicState(std::shared_ptr<const GameWithPublicStates> public_game,
                  std::vector<PublicTransition> public_obs);

  // Perspectives over the public state.
  std::vector<int> NumDistinctPrivateInformations() const override;
  std::vector<PrivateInformation> GetPrivateInformations(
      Player player) const override;
  std::vector<std::unique_ptr<State>> GetPublicSet() const override;
  std::string GetInformationState(
      const PrivateInformation& information) const override;
  std::vector<std::unique_ptr<State>> GetInformationSet(
      const PrivateInformation& information) const override;
  std::unique_ptr<State> GetWorldState(
      const std::vector<PrivateInformation*>& informations) const override;
  std::vector<Action> GetPrivateActions(
      const PrivateInformation& information) const override;
  std::unique_ptr<PrivateInformation> GetPrivateInformation(
      const State& state, Player p) const override;


  // Fetch a random subset from a perspective
  std::unique_ptr<State> ResampleFromPublicSet(Random* random) const override;
  std::unique_ptr<State> ResampleFromInformationSet(
      const PrivateInformation&, Random* random) const override;

  // Traversal of public state
  std::vector<PublicTransition> LegalTransitions() const override;
  std::vector<int> CountPrivateActions(Player player) const override;
  void UndoTransition(const PublicTransition& transition) override;

  // Public state types
  bool IsChance() const override;
  bool IsTerminal() const override;
  bool IsPlayer() const override;
  std::vector<Player> ActingPlayers() const override;
  bool IsPlayerActing(Player) const override;

  // CFR-related computations
  ReachProbs ComputeReachProbs(const PublicTransition& transition,
                               const std::vector<ArrayXd>& strategy,
                               ReachProbs reach_probs) const override;
  CfPrivValues TerminalCfValues(
      const std::vector<ReachProbs>& reach_probs, Player player) const override;
  CfPrivValues ComputeCfPrivValues(
      const std::vector<CfActionValues>& children_values,
      const std::vector<ArrayXd>& privates_policies) const override;
  std::vector<CfActionValues> ComputeCfActionValues(
      const std::vector<CfPrivValues>& children_values) const override;

  // Neural networks
  std::vector<double> PublicFeaturesTensor() const override;

  // Miscellaneous
  std::unique_ptr<PublicState> Clone() const override;

  bool PlayerReceivesItsCard(Player p) const;
  bool PlayerHasSeenItsCard(Player p) const;
  bool AllPlayerHaveSeenTheirCards() const;
  int NumPlayers() const;
  int NumCards() const;
  int NumPassesWithoutBet() const;

 private:
  // Public action of each player.
  std::vector<open_spiel::kuhn_poker::ActionType> public_actions_;
  void DoApplyPublicTransition(const PublicTransition& transition) override;
};

}  // namespace kuhn_poker
}  // namespace public_states
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PUBLIC_STATES_GAMES_KUHN_POKER_H_
