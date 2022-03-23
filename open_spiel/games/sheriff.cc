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

#include "open_spiel/games/sheriff.h"

#include <algorithm>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace sheriff {
inline constexpr const Player kSmuggler = Player{0};
inline constexpr const Player kSheriff = Player{1};

namespace {
// Facts about the game
const GameType kGameType{
    /* short_name = */ "sheriff",
    /* long_name = */ "Sheriff",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /* max_num_players = */ 2,
    /* min_num_players = */ 2,
    /* provides_information_state_string = */ true,
    /* provides_information_state_tensor = */ true,
    /* provides_observation_string = */ false,
    /* provides_observation_tensor = */ false,
    /* parameter_specification = */
    {{"item_penalty", GameParameter(kDefaultItemPenalty)},
     {"item_value", GameParameter(kDefaultItemValue)},
     {"sheriff_penalty", GameParameter(kDefaultSheriffPenalty)},
     {"max_bribe", GameParameter(kDefaultMaxBribe)},
     {"max_items", GameParameter(kDefaultMaxItems)},
     {"num_rounds", GameParameter(kDefaultNumRounds)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<const SheriffGame>(params);
}
REGISTER_SPIEL_GAME(kGameType, Factory);

template <typename T>
void StrAppendVector(std::string* s, const std::vector<T>& v) {
  absl::StrAppend(s, "[");
  for (size_t index = 0; index < v.size(); ++index) {
    if (index > 0) absl::StrAppend(s, ",");
    absl::StrAppend(s, v[index]);
  }
  absl::StrAppend(s, "]");
}
}  // namespace

SheriffState::SheriffState(
    const std::shared_ptr<const SheriffGame> sheriff_game)
    : State(sheriff_game), sheriff_game_(sheriff_game) {}

Player SheriffState::CurrentPlayer() const {
  if (!num_illegal_items_) {
    // The smuggler still hasn't decided the number of illegal items to
    // place in the cargo. The game has just begun.
    return kSmuggler;
  } else if (bribes_.size() == inspection_feedback_.size()) {
    // The smuggler has received feedback for all the bribes.
    //
    // If the number of bribes is equal to the number of bribing turns in the
    // game, the game is over.
    if (bribes_.size() == sheriff_game_->conf.num_rounds) {
      return kTerminalPlayerId;
    } else {
      // Otherwise, a new bribing round begins.
      return kSmuggler;
    }
  } else {
    // The smuggles has made a bribe, but no feedback has been given out yet.
    return kSheriff;
  }
}

std::vector<Action> SheriffState::LegalActions() const {
  const SheriffGame::SheriffGameConfiguration& conf = sheriff_game_->conf;

  std::vector<Action> action_ids;
  if (IsTerminal()) {
    return {};
  } else if (!num_illegal_items_) {
    // This is the beginning of the game. The smuggles must decide how many
    // illegal items to place in the cargo, which must be an integer in the
    // range [0, conf.max_items]. The action id will correspond to the number
    // of illegal items placed in the cargo.

    action_ids.reserve(conf.max_items + 1);
    for (uint32_t num_illegal_items = 0; num_illegal_items <= conf.max_items;
         ++num_illegal_items) {
      action_ids.push_back(
          sheriff_game_->SerializeItemPlacementAction(num_illegal_items));
    }
  } else {
    // If we are here, we are inside of a bribing round. There are two cases:
    // - it is the *smuggler's* turn. This means that the bribing round has
    //   just started. The actions that the player can use correspond to the
    //   set of valid bribes, which is the range [0, conf.max_bribe].
    //
    //   The action id corresponds to the bribe amount.
    // - it is the *sheriff's* turn. The sheriff can decide to say they will
    //   _not_ inspect (action id: 0), or that they _will_ inspect the cargo
    //   (action id: 1).
    const Player player = CurrentPlayer();

    if (player == kSmuggler) {
      action_ids.reserve(conf.max_bribe + 1);
      for (uint32_t bribe = 0; bribe <= conf.max_bribe; ++bribe) {
        action_ids.push_back(sheriff_game_->SerializeBribe(bribe));
      }
    } else {
      action_ids = {sheriff_game_->SerializeInspectionFeedback(false),
                    sheriff_game_->SerializeInspectionFeedback(true)};
    }
  }

  return action_ids;
}

std::string SheriffState::ActionToString(Player player,
                                         Action action_id) const {
  return sheriff_game_->ActionToString(player, action_id);
}

std::string SheriffState::ToString() const {
  if (!num_illegal_items_) {
    return "Initial game state (smuggler hasn't decided the number of illegal "
           "cargo items yet)";
  } else {
    std::string state_str;

    absl::StrAppend(&state_str,
                    "Num illegal items in cargo: ", *num_illegal_items_, "\n");
    absl::StrAppend(&state_str, "Bribes  : ");
    StrAppendVector(&state_str, bribes_);
    absl::StrAppend(&state_str, "\nFeedback: ");
    StrAppendVector(&state_str, inspection_feedback_);

    return state_str;
  }
}

bool SheriffState::IsTerminal() const {
  return CurrentPlayer() == kTerminalPlayerId;
}

std::vector<double> SheriffState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  } else {
    SPIEL_CHECK_EQ(inspection_feedback_.size(), bribes_.size());
    SPIEL_CHECK_GT(inspection_feedback_.size(), 0);
    SPIEL_CHECK_TRUE(num_illegal_items_);

    const uint32_t num_illegal_items = *num_illegal_items_;
    const uint32_t bribe = bribes_.back();
    const bool sheriff_inspects = inspection_feedback_.back();
    const SheriffGame::SheriffGameConfiguration& conf = sheriff_game_->conf;

    if (sheriff_inspects) {
      if (num_illegal_items > 0) {
        // The smuggler was caught red-handed.
        return {-static_cast<double>(num_illegal_items) * conf.item_penalty,
                static_cast<double>(num_illegal_items) * conf.item_penalty};
      } else {
        // The sheriff must pay up for inspecting a legal cargo.
        return {conf.sheriff_penalty, -conf.sheriff_penalty};
      }
    } else {
      return {static_cast<double>(num_illegal_items) * conf.item_value - bribe,
              static_cast<double>(bribe)};
    }
  }
}

std::unique_ptr<State> SheriffState::Clone() const {
  return std::make_unique<SheriffState>(*this);
}

std::string SheriffState::InformationStateString(Player player) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < NumPlayers());

  std::string infostring = absl::StrCat("T=", MoveNumber(), " ");
  if (player == kSmuggler) {
    absl::StrAppend(&infostring, "num_illegal_items:");
    if (num_illegal_items_) {
      absl::StrAppend(&infostring, *num_illegal_items_);
    } else {
      absl::StrAppend(&infostring, "none");
    }
  }

  SPIEL_CHECK_GE(inspection_feedback_.size() + 1, bribes_.size());
  SPIEL_CHECK_LE(inspection_feedback_.size(), bribes_.size());
  for (size_t index = 0; index < bribes_.size(); ++index) {
    absl::StrAppend(&infostring, "/bribe:", bribes_.at(index));

    if (index < inspection_feedback_.size()) {
      absl::StrAppend(&infostring,
                      "/feedback:", inspection_feedback_.at(index));
    }
  }

  return infostring;
}

std::vector<int> SheriffGame::InformationStateTensorShape() const {
  return {
    2 +                                      // Whose turn?
    2 +                                      // Who is observing?
    static_cast<int>(conf.num_rounds) + 1 +  // Move number (0 to rounds)
    static_cast<int>(conf.max_items) + 1 +   // Number of items (0 to max)
    // Each round, a bribe in { 0, 1, ...,  max_bribe } plus one bit for yes/no
    static_cast<int>(conf.num_rounds) *
        (static_cast<int>(conf.max_bribe) + 1 + 1)
  };
}

void SheriffState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < NumPlayers());

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorSize());
  std::fill(values.begin(), values.end(), 0);

  // Two-player game.
  SPIEL_CHECK_TRUE(player == 0 || player == 1);

  int offset = 0;
  const int num_players = game_->NumPlayers();
  const Player cur_player = CurrentPlayer();
  const auto* parent_game = down_cast<const SheriffGame*>(game_.get());

  // Set a bit to indicate whose turn it is.
  if (cur_player != kTerminalPlayerId) {
    values[cur_player] = 1;
  }
  offset += num_players;

  // Set a bit to indicate whose is observing
  values[offset + player] = 1;
  offset += num_players;

  // Move number
  values[offset + MoveNumber()] = 1;
  offset += parent_game->num_rounds() + 1;

  // Number of items chosen by the smuggler
  if (player == kSmuggler) {
    int index = (num_illegal_items_ ? *num_illegal_items_ : 0);
    values[offset + index] = 1;
  }
  offset += parent_game->max_items() + 1;

  SPIEL_CHECK_GE(inspection_feedback_.size() + 1, bribes_.size());
  SPIEL_CHECK_LE(inspection_feedback_.size(), bribes_.size());
  for (size_t index = 0; index < bribes_.size(); ++index) {
    int inner_offset = index * (parent_game->max_bribe() + 2);
    values[offset + inner_offset + bribes_.at(index)] = 1;

    if (index < inspection_feedback_.size()) {
      int bool_bit = inspection_feedback_.at(index) ? 0 : 1;
      values[offset + inner_offset + parent_game->max_bribe() + 1] = bool_bit;
    }
  }
  offset += parent_game->num_rounds() * (parent_game->max_bribe() + 2);

  SPIEL_CHECK_EQ(offset, values.size());
}

void SheriffState::UndoAction(Player player, Action action_id) {
  SPIEL_CHECK_TRUE(!history_.empty() &&
                   (history_.back() == PlayerAction{player, action_id}));
  history_.pop_back();
  --move_number_;

  if (!bribes_.empty()) {
    if (bribes_.size() == inspection_feedback_.size()) {
      // The last action must have been for the sheriff to return feedback about
      // whether or not the sheriff would inspect the cargo.
      inspection_feedback_.pop_back();
    } else {
      // The last action must have been for the smuggler to offer a new bribe.
      bribes_.pop_back();
    }
  } else {
    // If there are no bribes yet, then the only possibility is that the game
    // has just started and the only action so far was for the smuggler to
    // select the number of illegal items to place into the smuggler's cargo.
    SPIEL_CHECK_TRUE(num_illegal_items_);
    num_illegal_items_ = absl::nullopt;
  }
}

void SheriffState::DoApplyAction(Action action_id) {
  SPIEL_CHECK_FALSE(IsTerminal());

  if (!num_illegal_items_) {
    // The action must represent the selection of the number of illegal items in
    // the cargo.

    SPIEL_CHECK_EQ(CurrentPlayer(), kSmuggler);
    num_illegal_items_ =
        sheriff_game_->DeserializeItemPlacementAction(action_id);
  } else if (bribes_.size() == inspection_feedback_.size()) {
    // The action must represent a new bribe made by the smuggler.
    SPIEL_CHECK_EQ(CurrentPlayer(), kSmuggler);
    bribes_.push_back(sheriff_game_->DeserializeBribe(action_id));
  } else {
    // The action must represent the inspection feedback returned by the
    // sheriff.
    SPIEL_CHECK_EQ(CurrentPlayer(), kSheriff);
    inspection_feedback_.push_back(
        sheriff_game_->DeserializeInspectionFeedback(action_id));
  }
}

SheriffGame::SheriffGame(const GameParameters& params)
    : Game(kGameType, params) {
  conf.item_penalty = ParameterValue<double>("item_penalty");
  SPIEL_CHECK_GE(conf.item_penalty, 0.0);

  conf.item_value = ParameterValue<double>("item_value");
  SPIEL_CHECK_GE(conf.item_value, 0.0);

  conf.sheriff_penalty = ParameterValue<double>("sheriff_penalty");
  SPIEL_CHECK_GE(conf.sheriff_penalty, 0.0);

  conf.max_bribe = ParameterValue<int>("max_bribe");
  SPIEL_CHECK_GE(conf.max_bribe, 0);

  conf.max_items = ParameterValue<int>("max_items");
  SPIEL_CHECK_GE(conf.max_items, 1);

  conf.num_rounds = ParameterValue<int>("num_rounds");
  SPIEL_CHECK_GE(conf.num_rounds, 1);
}

int SheriffGame::NumDistinctActions() const {
  return 4 + conf.max_items + conf.max_bribe;
}

std::unique_ptr<State> SheriffGame::NewInitialState() const {
  const auto ptr =
      std::dynamic_pointer_cast<const SheriffGame>(shared_from_this());
  return std::make_unique<SheriffState>(ptr);
}

double SheriffGame::MinUtility() const {
  return std::min({-static_cast<double>(conf.max_items) * conf.item_penalty,
                   -static_cast<double>(conf.max_bribe),
                   -conf.sheriff_penalty});
}

double SheriffGame::MaxUtility() const {
  return std::max({conf.sheriff_penalty, static_cast<double>(conf.max_bribe),
                   static_cast<double>(conf.max_items) * conf.item_value,
                   static_cast<double>(conf.max_items) * conf.item_penalty});
}

double SheriffGame::UtilitySum() const {
  SpielFatalError("Called `UtilitySum()` on a general sum Sheriff game.");
}

int SheriffGame::MaxGameLength() const { return 2 * conf.num_rounds + 1; }

std::string SheriffGame::ActionToString(Player player, Action action_id) const {
  std::string action_string;

  if (action_id < 2) {
    SPIEL_CHECK_EQ(player, kSheriff);
    const bool feedback = DeserializeInspectionFeedback(action_id);
    if (!feedback) {
      action_string = "InspectionFeedback(will_inspect=False)";
    } else {
      action_string = "InspectionFeedback(will_inspect=True)";
    }
  } else if (action_id < 3 + conf.max_items) {
    SPIEL_CHECK_EQ(player, kSmuggler);

    const uint32_t num_illegal_items =
        DeserializeItemPlacementAction(action_id);
    absl::StrAppend(&action_string, "PlaceIllegalItems(num=", num_illegal_items,
                    ")");
  } else {
    SPIEL_CHECK_EQ(player, kSmuggler);

    const uint32_t bribe = DeserializeBribe(action_id);
    absl::StrAppend(&action_string, "Bribe(amount=", bribe, ")");
  }

  return action_string;
}

Action SheriffGame::SerializeItemPlacementAction(
    const uint32_t num_illegal_items) const {
  SPIEL_CHECK_LE(num_illegal_items, conf.max_items);
  return 2 + num_illegal_items;
}

Action SheriffGame::SerializeBribe(const uint32_t bribe) const {
  SPIEL_CHECK_LE(bribe, conf.max_bribe);
  return 3 + conf.max_items + bribe;
}

Action SheriffGame::SerializeInspectionFeedback(const bool feedback) const {
  if (!feedback) {
    return 0;
  } else {
    return 1;
  }
}

uint32_t SheriffGame::DeserializeItemPlacementAction(
    const Action action_id) const {
  SPIEL_CHECK_GE(action_id, 2);
  SPIEL_CHECK_LE(action_id, 2 + conf.max_items);

  return action_id - 2;
}

uint32_t SheriffGame::DeserializeBribe(const Action action_id) const {
  SPIEL_CHECK_GE(action_id, 3 + conf.max_items);
  SPIEL_CHECK_LE(action_id, 3 + conf.max_items + conf.max_bribe);

  return action_id - 3 - conf.max_items;
}

bool SheriffGame::DeserializeInspectionFeedback(const Action action_id) const {
  SPIEL_CHECK_TRUE(action_id == 0 || action_id == 1);

  if (action_id == 0) {
    return false;
  } else {
    return true;
  }
}

}  // namespace sheriff
}  // namespace open_spiel
