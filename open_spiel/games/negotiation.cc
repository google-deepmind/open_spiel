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

#include "open_spiel/games/negotiation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/random/poisson_distribution.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace negotiation {

namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"negotiation",
    /*long_name=*/"Negotiation",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"enable_proposals", GameParameter(kDefaultEnableProposals)},
     {"enable_utterances", GameParameter(kDefaultEnableUtterances)},
     {"num_items", GameParameter(kDefaultNumItems)},
     {"num_symbols", GameParameter(kDefaultNumSymbols)},
     {"rng_seed", GameParameter(kDefaultSeed)},
     {"utterance_dim", GameParameter(kDefaultUtteranceDim)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new NegotiationGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string TurnTypeToString(TurnType turn_type) {
  if (turn_type == TurnType::kProposal) {
    return "Proposal";
  } else if (turn_type == TurnType::kUtterance) {
    return "Utterance";
  } else {
    SpielFatalError("Unrecognized turn type");
  }
}
}  // namespace

std::string NegotiationState::ActionToString(Player player,
                                             Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("chance outcome ", move_id);
  } else {
    std::string action_string = "";
    if (turn_type_ == TurnType::kProposal) {
      if (move_id == parent_game_.NumDistinctProposals() - 1) {
        absl::StrAppend(&action_string, "Proposal: Agreement reached!");
      } else {
        std::vector<int> proposal = DecodeProposal(move_id);
        std::string prop_str = absl::StrJoin(proposal, ", ");
        absl::StrAppend(&action_string, "Proposal: [", prop_str, "]");
      }
    } else {
      std::vector<int> utterance = DecodeUtterance(move_id);
      std::string utt_str = absl::StrJoin(utterance, ", ");
      absl::StrAppend(&action_string, ", Utterance: [", utt_str, "]");
    }
    return action_string;
  }
}

bool NegotiationState::IsTerminal() const {
  // If utterances are enabled, force the agent to utter something even when
  // they accept the proposal or run out of steps (i.e. on ther last turn).
  bool utterance_check =
      (enable_utterances_ ? utterances_.size() == proposals_.size() : true);
  return (agreement_reached_ || proposals_.size() >= max_steps_) &&
         utterance_check;
}

std::vector<double> NegotiationState::Returns() const {
  if (!IsTerminal() || !agreement_reached_) {
    return std::vector<double>(num_players_, 0.0);
  }

  int proposing_player = proposals_.size() % 2 == 1 ? 0 : 1;
  int other_player = 1 - proposing_player;
  const std::vector<int>& final_proposal = proposals_.back();

  std::vector<double> returns(num_players_, 0.0);
  for (int j = 0; j < num_items_; ++j) {
    returns[proposing_player] +=
        agent_utils_[proposing_player][j] * final_proposal[j];
    returns[other_player] +=
        agent_utils_[other_player][j] * (item_pool_[j] - final_proposal[j]);
  }

  return returns;
}

std::string NegotiationState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (IsChanceNode()) {
    return "ChanceNode -- no observation";
  }

  std::string str = absl::StrCat("Max steps: ", max_steps_, "\n");
  absl::StrAppend(&str, "Item pool: ", absl::StrJoin(item_pool_, " "), "\n");

  if (!agent_utils_.empty()) {
    absl::StrAppend(&str, "Agent ", player,
                    " util vec: ", absl::StrJoin(agent_utils_[player], " "),
                    "\n");
  }

  absl::StrAppend(&str, "Current player: ", CurrentPlayer(), "\n");
  absl::StrAppend(&str, "Turn Type: ", TurnTypeToString(turn_type_), "\n");

  if (!proposals_.empty()) {
    absl::StrAppend(&str, "Most recent proposal: [",
                    absl::StrJoin(proposals_.back(), ", "), "]\n");
  }

  if (!utterances_.empty()) {
    absl::StrAppend(&str, "Most recent utterance: [",
                    absl::StrJoin(utterances_.back(), ", "), "]\n");
  }

  return str;
}

// 1D vector with shape:
//   - Current player: kNumPlayers bits
//   - Current turn type: 2 bits
//   - Terminal status: 2 bits: (Terminal? and Agreement reached?)
//   - Context:
//     - item pool      (num_items * (max_quantity + 1) bits)
//     - my utilities   (num_items * (max_value + 1) bits)
//   - Last proposal:   (num_items * (max_quantity + 1) bits)
// If utterances are enabled, another:
//   - Last utterance:  (utterance_dim * num_symbols) bits)
std::vector<int> NegotiationGame::ObservationTensorShape() const {
  return {kNumPlayers + 2 + 2 + (num_items_ * (kMaxQuantity + 1)) +
          (num_items_ * (kMaxValue + 1)) + (num_items_ * (kMaxQuantity + 1)) +
          (enable_utterances_ ? utterance_dim_ * num_symbols_ : 0)};
}

void NegotiationState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), parent_game_.ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0);

  // No observations at chance nodes.
  if (IsChanceNode()) {
    return;
  }

  // 1D vector with shape:
  //   - Current player: 2 bits
  //   - Turn type: 2 bits
  //   - Terminal status: 2 bits: (Terminal? and Agreement reached?)
  //   - Context:
  //     - item pool      (num_items * (max_quantity + 1) bits)
  //     - my utilities   (num_items * (max_value + 1) bits)
  //   - Last proposal:   (num_items * (max_quantity + 1) bits)
  // If utterances are enabled, another:
  //   - Last utterance:  (utterance_dim * num_symbols) bits)

  // Current player.
  int offset = 0;
  if (!IsTerminal()) {
    values[offset + CurrentPlayer()] = 1;
  }
  offset += kNumPlayers;

  // Current turn type.
  if (turn_type_ == TurnType::kProposal) {
    values[offset] = 1;
  } else {
    values[offset + 1] = 1;
  }
  offset += 2;

  // Terminal status: 2 bits
  values[offset] = IsTerminal() ? 1 : 0;
  values[offset + 1] = agreement_reached_ ? 1 : 0;
  offset += 2;

  // Item pool.
  for (int item = 0; item < num_items_; ++item) {
    values[offset + item_pool_[item]] = 1;
    offset += kMaxQuantity + 1;
  }

  // Utilities.
  for (int item = 0; item < num_items_; ++item) {
    values[offset + agent_utils_[player][item]] = 1;
    offset += kMaxValue + 1;
  }

  // Last proposal.
  if (!proposals_.empty()) {
    for (int item = 0; item < num_items_; ++item) {
      values[offset + proposals_.back()[item]] = 1;
      offset += kMaxQuantity + 1;
    }
  } else {
    offset += num_items_ * (kMaxQuantity + 1);
  }

  // Last utterance.
  if (enable_utterances_) {
    if (!utterances_.empty()) {
      for (int dim = 0; dim < utterance_dim_; ++dim) {
        values[offset + utterances_.back()[dim]] = 1;
        offset += num_symbols_;
      }
    } else {
      offset += utterance_dim_ * num_symbols_;
    }
  }

  SPIEL_CHECK_EQ(offset, values.size());
}

NegotiationState::NegotiationState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(static_cast<const NegotiationGame&>(*game)),
      enable_proposals_(parent_game_.EnableProposals()),
      enable_utterances_(parent_game_.EnableUtterances()),
      num_items_(parent_game_.NumItems()),
      num_symbols_(parent_game_.NumSymbols()),
      utterance_dim_(parent_game_.UtteranceDim()),
      num_steps_(0),
      max_steps_(-1),
      agreement_reached_(false),
      cur_player_(kChancePlayerId),
      turn_type_(TurnType::kProposal),
      item_pool_({}),
      agent_utils_({}),
      proposals_({}),
      utterances_({}) {}

int NegotiationState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

// From Sec 2.1 of the paper: "At each round (i) an item pool is sampled
// uniformly, instantiating a quantity (between 0 and 5) for each of the types
// and represented as a vector i \in {0...5}^3 and (ii) each agent j receives a
// utility function sampled uniformly, which specifies how rewarding one unit of
// each item is (with item rewards between 0 and 10, and with the constraint
// that there is at least one item with non-zero utility), represented as a
// vector u_j \in {0...10}^3".
void NegotiationState::DetermineItemPoolAndUtilities() {
  // Generate max number of rounds (max number of steps for the episode): we
  // sample N between 4 and 10 at the start of each episode, according to a
  // truncated Poissondistribution with mean 7, as done in the Cao et al. '18
  // paper.
  max_steps_ = -1;
  absl::poisson_distribution<int> steps_dist(7.0);
  while (!(max_steps_ >= 4 && max_steps_ <= 10)) {
    max_steps_ = steps_dist(*parent_game_.RNG());
  }

  // Generate the pool of items.
  absl::uniform_int_distribution<int> quantity_dist(0, kMaxQuantity);
  for (int i = 0; i < num_items_; ++i) {
    item_pool_.push_back(quantity_dist(*parent_game_.RNG()));
  }

  // Generate agent utilities.
  absl::uniform_int_distribution<int> util_dist(0, kMaxValue);
  for (int i = 0; i < num_players_; ++i) {
    agent_utils_.push_back({});
    int sum_util = 0;
    while (sum_util == 0) {
      for (int j = 0; j < num_items_; ++j) {
        agent_utils_[i].push_back(util_dist(*parent_game_.RNG()));
        sum_util += agent_utils_[i].back();
      }
    }
  }
}

void NegotiationState::InitializeEpisode() {
  cur_player_ = 0;
  turn_type_ = TurnType::kProposal;
}

void NegotiationState::DoApplyAction(Action move_id) {
  if (IsChanceNode()) {
    DetermineItemPoolAndUtilities();
    InitializeEpisode();
  } else {
    if (turn_type_ == TurnType::kProposal) {
      if (move_id == parent_game_.NumDistinctProposals() - 1) {
        // Agreement!
        agreement_reached_ = true;
      } else {
        std::vector<int> proposal = DecodeProposal(move_id);
        proposals_.push_back(proposal);
      }

      if (enable_utterances_) {
        // Note: do not move to next player yet.
        turn_type_ = TurnType::kUtterance;
      } else {
        // Keep it at kProposal, but move to next player.
        cur_player_ = 1 - cur_player_;
      }
    } else {
      SPIEL_CHECK_TRUE(enable_utterances_);
      std::vector<int> utterance = DecodeUtterance(move_id);
      utterances_.push_back(utterance);
      turn_type_ = TurnType::kProposal;
      cur_player_ = 1 - cur_player_;
    }
  }
}

bool NegotiationState::NextProposal(std::vector<int>* proposal) const {
  // Starting from the right, move left trying to increase the value. When
  // successful, increment the value and set all the right digits back to 0.
  for (int i = num_items_ - 1; i >= 0; --i) {
    if ((*proposal)[i] + 1 <= item_pool_[i]) {
      // Success!
      (*proposal)[i]++;
      for (int j = i + 1; j < num_items_; ++j) {
        (*proposal)[j] = 0;
      }
      return true;
    }
  }

  return false;
}

std::vector<int> NegotiationState::DecodeInteger(int encoded_value,
                                                 int dimensions,
                                                 int num_digit_values) const {
  std::vector<int> decoded(dimensions, 0);
  int i = dimensions - 1;
  while (encoded_value > 0) {
    SPIEL_CHECK_GE(i, 0);
    SPIEL_CHECK_LT(i, dimensions);
    decoded[i] = encoded_value % num_digit_values;
    encoded_value /= num_digit_values;
    i--;
  }
  return decoded;
}

int NegotiationState::EncodeInteger(const std::vector<int>& container,
                                    int num_digit_values) const {
  int encoded_value = 0;
  for (int digit : container) {
    encoded_value = encoded_value * num_digit_values + digit;
  }
  return encoded_value;
}

Action NegotiationState::EncodeProposal(
    const std::vector<int>& proposal) const {
  SPIEL_CHECK_EQ(proposal.size(), num_items_);
  return EncodeInteger(proposal, kMaxQuantity + 1);
}

Action NegotiationState::EncodeUtterance(
    const std::vector<int>& utterance) const {
  SPIEL_CHECK_EQ(utterance.size(), utterance_dim_);
  // Utterance ids are offset from zero (starting at NumDistinctProposals()).
  return parent_game_.NumDistinctProposals() +
         EncodeInteger(utterance, num_symbols_);
}

std::vector<int> NegotiationState::DecodeProposal(int encoded_proposal) const {
  return DecodeInteger(encoded_proposal, num_items_, kMaxQuantity + 1);
}

std::vector<int> NegotiationState::DecodeUtterance(
    int encoded_utterance) const {
  // Utterance ids are offset from zero (starting at NumDistinctProposals()).
  return DecodeInteger(encoded_utterance - parent_game_.NumDistinctProposals(),
                       utterance_dim_, num_symbols_);
}

std::vector<Action> NegotiationState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (turn_type_ == TurnType::kProposal) {
    std::vector<Action> legal_actions;

    // Proposals are always enabled, so first contruct them.
    std::vector<int> proposal(num_items_, 0);
    legal_actions.push_back(EncodeProposal(proposal));

    while (NextProposal(&proposal)) {
      legal_actions.push_back(EncodeProposal(proposal));
    }

    if (!proposals_.empty()) {
      // Add the agreement action only if there's been at least one proposal.
      legal_actions.push_back(parent_game_.NumDistinctProposals() - 1);
    }

    return legal_actions;
  } else {
    SPIEL_CHECK_TRUE(enable_utterances_);
    SPIEL_CHECK_FALSE(parent_game_.LegalUtterances().empty());
    return parent_game_.LegalUtterances();
  }
}

std::vector<std::pair<Action, double>> NegotiationState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  // The game has chance mode kSampledStochastic, so there is only a single
  // outcome, and it's all randomized in the ApplyAction.
  std::vector<std::pair<Action, double>> outcomes = {std::make_pair(0, 1.0)};
  return outcomes;
}

std::string NegotiationState::ToString() const {
  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str = absl::StrCat("Max steps: ", max_steps_, "\n");
  absl::StrAppend(&str, "Item pool: ", absl::StrJoin(item_pool_, " "), "\n");

  if (!agent_utils_.empty()) {
    for (int i = 0; i < num_players_; ++i) {
      absl::StrAppend(&str, "Agent ", i,
                      " util vec: ", absl::StrJoin(agent_utils_[i], " "), "\n");
    }
  }

  absl::StrAppend(&str, "Current player: ", cur_player_, "\n");
  absl::StrAppend(&str, "Turn Type: ", TurnTypeToString(turn_type_), "\n");

  for (int i = 0; i < proposals_.size(); ++i) {
    absl::StrAppend(&str, "Player ", i % 2, " proposes: [",
                    absl::StrJoin(proposals_[i], ", "), "]");
    if (enable_utterances_ && i < utterances_.size()) {
      absl::StrAppend(&str, " utters: [", absl::StrJoin(utterances_[i], ", "),
                      "]");
    }
    absl::StrAppend(&str, "\n");
  }

  if (agreement_reached_) {
    absl::StrAppend(&str, "Agreement reached!\n");
  }

  return str;
}

std::unique_ptr<State> NegotiationState::Clone() const {
  return std::unique_ptr<State>(new NegotiationState(*this));
}

NegotiationGame::NegotiationGame(const GameParameters& params)
    : Game(kGameType, params),
      enable_proposals_(
          ParameterValue<bool>("enable_proposals", kDefaultEnableProposals)),
      enable_utterances_(
          ParameterValue<bool>("enable_utterances", kDefaultEnableUtterances)),
      num_items_(ParameterValue<int>("num_items", kDefaultNumItems)),
      num_symbols_(ParameterValue<int>("num_symbols", kDefaultNumSymbols)),
      utterance_dim_(
          ParameterValue<int>("utterance_dim", kDefaultUtteranceDim)),
      seed_(ParameterValue<int>("rng_seed", kDefaultSeed)),
      legal_utterances_({}),
      rng_(new std::mt19937(seed_ >= 0 ? seed_ : std::mt19937::default_seed)) {
  ConstructLegalUtterances();
}

void NegotiationGame::ConstructLegalUtterances() {
  if (enable_utterances_) {
    legal_utterances_.resize(NumDistinctUtterances());
    for (int i = 0; i < NumDistinctUtterances(); ++i) {
      legal_utterances_[i] = NumDistinctProposals() + i;
    }
  }
}

int NegotiationGame::MaxGameLength() const {
  if (enable_utterances_) {
    return 2 * kMaxSteps;  // Every step is two turns: proposal, then utterance.
  } else {
    return kMaxSteps;
  }
}

int NegotiationGame::NumDistinctUtterances() const {
  return static_cast<int>(std::pow(num_symbols_, utterance_dim_));
}

int NegotiationGame::NumDistinctProposals() const {
  // Every slot can hold { 0, 1, ..., MaxQuantity }, and there is an extra
  // one at the end for the special "agreement reached" action.
  return static_cast<int>(std::pow(kMaxQuantity + 1, num_items_)) + 1;
}

// See the header for an explanation of how these are encoded.
int NegotiationGame::NumDistinctActions() const {
  if (enable_utterances_) {
    return NumDistinctProposals() + NumDistinctUtterances();
  } else {
    // Proposals are always possible.
    return NumDistinctProposals();
  }
}

std::string NegotiationState::Serialize() const {
  if (IsChanceNode()) {
    return "chance";
  } else {
    std::string state_str = "";
    absl::StrAppend(&state_str, MaxSteps(), "\n");
    absl::StrAppend(&state_str, absl::StrJoin(ItemPool(), " "), "\n");
    for (int p = 0; p < NumPlayers(); ++p) {
      absl::StrAppend(&state_str, absl::StrJoin(AgentUtils()[p], " "), "\n");
    }
    absl::StrAppend(&state_str, HistoryString(), "\n");
    return state_str;
  }
}

std::unique_ptr<State> NegotiationGame::DeserializeState(
    const std::string& str) const {
  if (str == "chance") {
    return NewInitialState();
  } else {
    std::vector<std::string> lines = absl::StrSplit(str, '\n');
    std::unique_ptr<State> state = NewInitialState();
    SPIEL_CHECK_EQ(lines.size(), 5);
    NegotiationState& nstate = static_cast<NegotiationState&>(*state);
    // Take the chance action, but then reset the quantities. Make sure game's
    // RNG state is not advanced during deserialization so copy it beforehand
    // in order to be able to restore after the chance action.
    std::unique_ptr<std::mt19937> rng = std::make_unique<std::mt19937>(*rng_);
    nstate.ApplyAction(0);
    rng_ = std::move(rng);
    nstate.ItemPool().clear();
    nstate.AgentUtils().clear();
    // Max steps
    nstate.SetMaxSteps(std::stoi(lines[0]));
    // Item pool.
    std::vector<std::string> parts = absl::StrSplit(lines[1], ' ');
    for (const auto& part : parts) {
      nstate.ItemPool().push_back(std::stoi(part));
    }
    // Agent utilities.
    for (Player player : {0, 1}) {
      parts = absl::StrSplit(lines[2 + player], ' ');
      nstate.AgentUtils().push_back({});
      for (const auto& part : parts) {
        nstate.AgentUtils()[player].push_back(std::stoi(part));
      }
    }
    nstate.SetCurrentPlayer(0);
    // Actions.
    if (lines.size() == 5) {
      parts = absl::StrSplit(lines[4], ' ');
      // Skip the first one since it is the chance node.
      for (int i = 1; i < parts.size(); ++i) {
        Action action = static_cast<Action>(std::stoi(parts[i]));
        nstate.ApplyAction(action);
      }
    }
    return state;
  }
}

std::string NegotiationGame::GetRNGState() const {
  std::ostringstream rng_stream;
  rng_stream << *rng_;
  return rng_stream.str();
}

void NegotiationGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) return;
  std::istringstream rng_stream(rng_state);
  rng_stream >> *rng_;
}

}  // namespace negotiation
}  // namespace open_spiel
