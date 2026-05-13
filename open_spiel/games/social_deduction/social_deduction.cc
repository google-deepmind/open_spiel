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

#include "open_spiel/games/social_deduction/social_deduction.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace social_deduction {
namespace {

constexpr int kDefaultPlayers = 6;
constexpr int kDefaultImposters = 1;
constexpr int kDefaultMaxRounds = 10;
constexpr double kDefaultObservationNoise = 0.1;

constexpr Action kSkip = 0;
constexpr Action kClaimInnocent = 1;
constexpr Action kAccuseOffset = 2;

constexpr int kInnocentTeam = 0;
constexpr int kImposterTeam = 1;

const GameType kGameType{
    /*short_name=*/"social_deduction",
    /*long_name=*/"Social Deduction",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/3,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"players", GameParameter(kDefaultPlayers)},
     {"imposters", GameParameter(kDefaultImposters)},
     {"max_rounds", GameParameter(kDefaultMaxRounds)},
     {"observation_noise", GameParameter(kDefaultObservationNoise)}}};  // NOLINT

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SocialDeductionGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

int PopCount(int mask) {
  int count = 0;
  while (mask != 0) {
    count += mask & 1;
    mask >>= 1;
  }
  return count;
}

std::vector<Action> RoleAssignments(int num_players, int num_imposters) {
  std::vector<Action> actions;
  const int num_masks = 1 << num_players;
  for (int mask = 0; mask < num_masks; ++mask) {
    if (PopCount(mask) == num_imposters) actions.push_back(mask);
  }
  return actions;
}

std::vector<Action> ObservationSignals(const std::vector<bool>& alive) {
  std::vector<Action> actions;
  for (Player player = 0; player < alive.size(); ++player) {
    if (alive[player]) {
      actions.push_back(2 * player);
      actions.push_back(2 * player + 1);
    }
  }
  return actions;
}

std::string PhaseString(Phase phase) {
  switch (phase) {
    case Phase::kAssignRoles:
      return "AssignRoles";
    case Phase::kObservation:
      return "Observation";
    case Phase::kCommunication:
      return "Communication";
    case Phase::kVoting:
      return "Voting";
    case Phase::kTerminal:
      return "Terminal";
    default:
      SpielFatalError("Unknown phase.");
  }
}

std::string MessageString(Action action, int num_players) {
  if (action == kSkip) return "SKIP";
  if (action == kClaimInnocent) return "CLAIM_INNOCENT";
  if (action >= kAccuseOffset && action < kAccuseOffset + num_players) {
    return absl::StrCat("ACCUSE_PLAYER_", action - kAccuseOffset);
  }
  const Action defend_offset = kAccuseOffset + num_players;
  if (action >= defend_offset && action < defend_offset + num_players) {
    return absl::StrCat("DEFEND_PLAYER_", action - defend_offset);
  }
  return absl::StrCat("UNKNOWN_MESSAGE_", action);
}

std::string SignalString(Action action) {
  return absl::StrCat("SIGNAL_PLAYER_", action / 2,
                      action % 2 == 1 ? "_SUSPICIOUS" : "_CLEAR");
}

}  // namespace

SocialDeductionState::SocialDeductionState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(static_cast<const SocialDeductionGame&>(*game)),
      roles_(game->NumPlayers(), Role::kInnocent),
      alive_(game->NumPlayers(), true) {}  // NOLINT

Player SocialDeductionState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (phase_ == Phase::kAssignRoles || phase_ == Phase::kObservation) {
    return kChancePlayerId;
  }
  const Player next = NextAlivePlayer(actor_cursor_);
  SPIEL_CHECK_NE(next, kInvalidPlayer);
  return next;
}

std::vector<Action> SocialDeductionState::LegalActions() const {
  if (IsTerminal()) return {};
  if (phase_ == Phase::kAssignRoles || phase_ == Phase::kObservation) {
    return LegalChanceOutcomes();
  }

  std::vector<Action> actions;
  if (phase_ == Phase::kCommunication) {
    actions.push_back(kSkip);
    actions.push_back(kClaimInnocent);
    for (Player target = 0; target < num_players_; ++target) {
      if (IsAlive(target)) actions.push_back(kAccuseOffset + target);
    }
    for (Player target = 0; target < num_players_; ++target) {
      if (IsAlive(target)) {
        actions.push_back(kAccuseOffset + num_players_ + target);
      }
    }
    return actions;
  }

  if (phase_ == Phase::kVoting) {
    const Player voter = CurrentPlayer();
    for (Player target = 0; target < num_players_; ++target) {
      if (IsAlive(target) && target != voter) actions.push_back(target);
    }
    return actions;
  }

  SpielFatalError("Unknown phase.");
}

ActionsAndProbs SocialDeductionState::ChanceOutcomes() const {
  if (phase_ == Phase::kAssignRoles) {
    std::vector<Action> assignments =
        RoleAssignments(num_players_, parent_game_.NumImposters());
    ActionsAndProbs outcomes;
    outcomes.reserve(assignments.size());
    const double probability = 1.0 / assignments.size();
    for (Action action : assignments) outcomes.push_back({action, probability});
    return outcomes;
  }

  SPIEL_CHECK_TRUE(phase_ == Phase::kObservation);
  ActionsAndProbs outcomes;
  const double target_probability = 1.0 / AlivePlayers();
  for (Player target = 0; target < num_players_; ++target) {
    if (!IsAlive(target)) continue;
    const double suspicious_probability =
        IsImposter(target) ? 1.0 - parent_game_.ObservationNoise()
                           : parent_game_.ObservationNoise();
    const double clear_probability = 1.0 - suspicious_probability;
    if (clear_probability > 0.0) {
      outcomes.push_back({2 * target, target_probability * clear_probability});
    }
    if (suspicious_probability > 0.0) {
      outcomes.push_back(
          {2 * target + 1, target_probability * suspicious_probability});
    }
  }
  return outcomes;
}

void SocialDeductionState::DoApplyAction(Action action) {
  if (phase_ == Phase::kAssignRoles) {
    SPIEL_CHECK_EQ(PopCount(action), parent_game_.NumImposters());
    for (Player player = 0; player < num_players_; ++player) {
      roles_[player] =
          ((action >> player) & 1) ? Role::kImposter : Role::kInnocent;
    }
    roles_assigned_ = true;
    StartRound();
    return;
  }

  if (phase_ == Phase::kObservation) {
    SPIEL_CHECK_TRUE(roles_assigned_);
    const Player target = action / 2;
    SPIEL_CHECK_TRUE(IsAlive(target));
    SPIEL_CHECK_LT(action, 2 * num_players_);
    rounds_.back().signal_target = target;
    rounds_.back().signal_suspicious = action % 2 == 1;
    phase_ = Phase::kCommunication;
    actor_cursor_ = 0;
    return;
  }

  const Player player = CurrentPlayer();
  if (phase_ == Phase::kCommunication) {
    rounds_.back().messages[player] = action;
    actor_cursor_ = player + 1;
    if (NextAlivePlayer(actor_cursor_) == kInvalidPlayer) {
      FinishCommunication();
    }
    return;
  }

  SPIEL_CHECK_TRUE(phase_ == Phase::kVoting);
  SPIEL_CHECK_TRUE(IsAlive(action));
  SPIEL_CHECK_NE(player, action);
  rounds_.back().votes[player] = action;
  actor_cursor_ = player + 1;
  if (NextAlivePlayer(actor_cursor_) == kInvalidPlayer) ResolveVotes();
}

std::string SocialDeductionState::ActionToString(Player player,
                                                 Action action_id) const {
  if (player == kChancePlayerId) {
    if (!roles_assigned_) {
      std::vector<std::string> imposters;
      for (Player p = 0; p < num_players_; ++p) {
        if ((action_id >> p) & 1) imposters.push_back(absl::StrCat(p));
      }
      return absl::StrCat("ASSIGN_IMPOSTERS_", absl::StrJoin(imposters, "_"));
    }
    return SignalString(action_id);
  }
  if (phase_ == Phase::kVoting) {
    return absl::StrCat("VOTE_PLAYER_", action_id);
  }
  return MessageString(action_id, num_players_);
}

std::string SocialDeductionState::ToString() const {
  std::ostringstream out;
  out << "Round: " << round_ << "\n";
  out << "Phase: " << PhaseString(phase_) << "\n";
  if (roles_assigned_) {
    out << "Roles:";
    for (Player player = 0; player < num_players_; ++player) {
      out << " " << player << "=" << RoleString(player);
    }
    out << "\n";
  }
  AppendPublicHistory(out);
  return out.str();
}

bool SocialDeductionState::IsTerminal() const {
  return winner_team_ != kInvalidPlayer;
}

std::vector<double> SocialDeductionState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  if (!IsTerminal()) return returns;
  for (Player player = 0; player < num_players_; ++player) {
    const int team = IsImposter(player) ? kImposterTeam : kInnocentTeam;
    returns[player] = team == winner_team_ ? 1.0 : -1.0;
  }
  return returns;
}

std::unique_ptr<State> SocialDeductionState::Clone() const {
  return std::unique_ptr<State>(new SocialDeductionState(*this));
}

std::string SocialDeductionState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::ostringstream out;
  out << ObservationString(player);
  out << "InformationStatePlayer: " << player << "\n";
  return out.str();
}

std::string SocialDeductionState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::ostringstream out;
  out << "Player: " << player << "\n";
  out << "Role: " << RoleString(player) << "\n";
  if (roles_assigned_ && IsImposter(player)) {
    out << "ImposterTeam:";
    for (Player other = 0; other < num_players_; ++other) {
      if (IsImposter(other)) out << " " << other;
    }
    out << "\n";
  }
  AppendPublicHistory(out);
  return out.str();
}

bool SocialDeductionState::IsAlive(Player player) const {
  return player >= 0 && player < alive_.size() && alive_[player];
}

bool SocialDeductionState::IsImposter(Player player) const {
  return roles_[player] == Role::kImposter;
}

int SocialDeductionState::AlivePlayers() const {
  return std::count(alive_.begin(), alive_.end(), true);
}

int SocialDeductionState::AliveImposters() const {
  int count = 0;
  for (Player player = 0; player < num_players_; ++player) {
    if (IsAlive(player) && IsImposter(player)) ++count;
  }
  return count;
}

int SocialDeductionState::AliveInnocents() const {
  return AlivePlayers() - AliveImposters();
}

Player SocialDeductionState::NextAlivePlayer(int start_player) const {
  for (Player player = std::max(0, start_player); player < num_players_;
       ++player) {
    if (IsAlive(player)) return player;
  }
  return kInvalidPlayer;
}

void SocialDeductionState::StartRound() {
  UpdateWinner();
  if (IsTerminal()) return;
  phase_ = Phase::kObservation;
  actor_cursor_ = 0;
  rounds_.push_back(RoundRecord{
      /*signal_target=*/kInvalidPlayer,
      /*signal_suspicious=*/false,
      /*messages=*/std::vector<Action>(num_players_, kInvalidAction),
      /*votes=*/std::vector<Player>(num_players_, kInvalidPlayer),
      /*eliminated=*/kInvalidPlayer});
}

void SocialDeductionState::FinishCommunication() {
  phase_ = Phase::kVoting;
  actor_cursor_ = 0;
}

void SocialDeductionState::ResolveVotes() {
  std::vector<int> vote_counts(num_players_, 0);
  for (Player voter = 0; voter < num_players_; ++voter) {
    if (rounds_.back().votes[voter] != kInvalidPlayer) {
      ++vote_counts[rounds_.back().votes[voter]];
    }
  }

  Player eliminated = kInvalidPlayer;
  int best_votes = 0;
  bool tied = false;
  for (Player target = 0; target < num_players_; ++target) {
    if (!IsAlive(target) || vote_counts[target] == 0) continue;
    if (vote_counts[target] > best_votes) {
      best_votes = vote_counts[target];
      eliminated = target;
      tied = false;
    } else if (vote_counts[target] == best_votes) {
      tied = true;
    }
  }

  if (!tied && eliminated != kInvalidPlayer) {
    alive_[eliminated] = false;
    rounds_.back().eliminated = eliminated;
  }

  ++round_;
  UpdateWinner();
  if (IsTerminal()) return;
  if (round_ >= parent_game_.MaxRounds()) {
    winner_team_ = kImposterTeam;
    phase_ = Phase::kTerminal;
    return;
  }
  StartRound();
}

void SocialDeductionState::UpdateWinner() {
  if (AliveImposters() == 0) {
    winner_team_ = kInnocentTeam;
    phase_ = Phase::kTerminal;
  } else if (AliveImposters() >= AliveInnocents()) {
    winner_team_ = kImposterTeam;
    phase_ = Phase::kTerminal;
  }
}

void SocialDeductionState::AppendPublicHistory(std::ostream& out) const {
  out << "Alive:";
  for (Player player = 0; player < num_players_; ++player) {
    if (IsAlive(player)) out << " " << player;
  }
  out << "\n";
  for (int r = 0; r < rounds_.size(); ++r) {
    const RoundRecord& record = rounds_[r];
    out << "Round " << r << "\n";
    if (record.signal_target != kInvalidPlayer) {
      out << "Signal: player " << record.signal_target
          << (record.signal_suspicious ? " suspicious" : " clear") << "\n";
    }
    out << "Messages:";
    for (Player player = 0; player < num_players_; ++player) {
      if (record.messages[player] != kInvalidAction) {
        out << " " << player << "="
            << MessageString(record.messages[player], num_players_);
      }
    }
    out << "\n";
    out << "Votes:";
    for (Player player = 0; player < num_players_; ++player) {
      if (record.votes[player] != kInvalidPlayer) {
        out << " " << player << "->" << record.votes[player];
      }
    }
    out << "\n";
    if (record.eliminated != kInvalidPlayer) {
      out << "Eliminated: " << record.eliminated << "\n";
    }
  }
}

std::string SocialDeductionState::RoleString(Player player) const {
  if (!roles_assigned_) return "Unknown";
  return roles_[player] == Role::kImposter ? "Imposter" : "Innocent";
}

SocialDeductionGame::SocialDeductionGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      num_imposters_(ParameterValue<int>("imposters")),
      max_rounds_(ParameterValue<int>("max_rounds")),
      observation_noise_(ParameterValue<double>("observation_noise")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  SPIEL_CHECK_GE(num_imposters_, 1);
  SPIEL_CHECK_LT(num_imposters_, num_players_);
  SPIEL_CHECK_LT(num_imposters_, num_players_ - num_imposters_);
  SPIEL_CHECK_GE(max_rounds_, 1);
  SPIEL_CHECK_GE(observation_noise_, 0.0);
  SPIEL_CHECK_LE(observation_noise_, 1.0);
}

int SocialDeductionGame::NumDistinctActions() const {
  return std::max(num_players_, static_cast<int>(kAccuseOffset) +
                                    2 * num_players_);
}

int SocialDeductionGame::MaxChanceOutcomes() const {
  return std::max(1 << num_players_, 2 * num_players_);
}

std::unique_ptr<State> SocialDeductionGame::NewInitialState() const {
  return std::unique_ptr<State>(new SocialDeductionState(shared_from_this()));
}

int SocialDeductionGame::MaxGameLength() const {
  return 1 + max_rounds_ * (1 + 2 * num_players_);
}

}  // namespace social_deduction
}  // namespace open_spiel
