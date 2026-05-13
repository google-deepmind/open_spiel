// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_SOCIAL_DEDUCTION_SOCIAL_DEDUCTION_H_
#define OPEN_SPIEL_GAMES_SOCIAL_DEDUCTION_SOCIAL_DEDUCTION_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple configurable hidden-role social deduction benchmark.
//
// Players are assigned hidden roles: innocents and imposters. Imposters know
// the full imposter team, while innocents only know their own role. Each round
// has an observation chance event, one tokenized communication action per
// living player, and one elimination vote per living player. Innocents win by
// eliminating all imposters. Imposters win by reaching parity or by surviving
// until the round limit.

namespace open_spiel {
namespace social_deduction {

enum class Role { kInnocent = 0, kImposter = 1 };

enum class Phase {
  kAssignRoles = 0,
  kObservation = 1,
  kCommunication = 2,
  kVoting = 3,
  kTerminal = 4
};

class SocialDeductionGame;

struct RoundRecord {
  int signal_target = kInvalidPlayer;
  bool signal_suspicious = false;
  std::vector<Action> messages;
  std::vector<Player> votes;
  Player eliminated = kInvalidPlayer;
};

class SocialDeductionState : public State {
 public:
  explicit SocialDeductionState(std::shared_ptr<const Game> game);
  SocialDeductionState(const SocialDeductionState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool IsAlive(Player player) const;
  bool IsImposter(Player player) const;
  int AlivePlayers() const;
  int AliveImposters() const;
  int AliveInnocents() const;
  Player NextAlivePlayer(int start_player) const;
  void StartRound();
  void FinishCommunication();
  void ResolveVotes();
  void UpdateWinner();
  void AppendPublicHistory(std::ostream& out) const;
  std::string RoleString(Player player) const;

  const SocialDeductionGame& parent_game_;

  Phase phase_ = Phase::kAssignRoles;
  int round_ = 0;
  int actor_cursor_ = 0;
  int winner_team_ = kInvalidPlayer;
  bool roles_assigned_ = false;
  std::vector<Role> roles_;
  std::vector<bool> alive_;
  std::vector<RoundRecord> rounds_;
};

class SocialDeductionGame : public Game {
 public:
  explicit SocialDeductionGame(const GameParameters& params);

  int NumDistinctActions() const override;
  int MaxChanceOutcomes() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return num_players_; }
  double MaxUtility() const override { return 1.0; }
  double MinUtility() const override { return -1.0; }
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override { return 1 + max_rounds_; }

  int NumImposters() const { return num_imposters_; }
  int MaxRounds() const { return max_rounds_; }
  double ObservationNoise() const { return observation_noise_; }

 private:
  int num_players_;
  int num_imposters_;
  int max_rounds_;
  double observation_noise_;
};

}  // namespace social_deduction
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SOCIAL_DEDUCTION_SOCIAL_DEDUCTION_H_
