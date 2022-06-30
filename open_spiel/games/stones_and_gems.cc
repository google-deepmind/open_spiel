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

#include "open_spiel/games/stones_and_gems.h"

#include <sys/types.h>

#include <algorithm>  // std::find, min
#include <utility>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace stones_and_gems {

namespace {

// Property bit flags
enum ElementProperties {
  kNone = 0,
  kConsumable = 1 << 0,
  kCanExplode = 1 << 1,
  kRounded = 1 << 2,
  kTraversable = 1 << 3,
};

// All possible elements
const Element kElAgent = {
    HiddenCellType::kAgent, VisibleCellType::kAgent,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, '@'};
const Element kElAgentInExit = {HiddenCellType::kAgentInExit,
                                VisibleCellType::kAgentInExit,
                                ElementProperties::kNone, '!'};
const Element kElExitOpen = {HiddenCellType::kExitOpen,
                             VisibleCellType::kExitOpen,
                             ElementProperties::kTraversable, '#'};
const Element kElExitClosed = {HiddenCellType::kExitClosed,
                               VisibleCellType::kExitClosed,
                               ElementProperties::kNone, 'C'};
const Element kElEmpty = {
    HiddenCellType::kEmpty, VisibleCellType::kEmpty,
    ElementProperties::kConsumable | ElementProperties::kTraversable, ' '};
const Element kElDirt = {
    HiddenCellType::kDirt, VisibleCellType::kDirt,
    ElementProperties::kConsumable | ElementProperties::kTraversable, '.'};
const Element kElStone = {
    HiddenCellType::kStone, VisibleCellType::kStone,
    ElementProperties::kConsumable | ElementProperties::kRounded, 'o'};
const Element kElStoneFalling = {HiddenCellType::kStoneFalling,
                                 VisibleCellType::kStone,
                                 ElementProperties::kConsumable, 'o'};
const Element kElDiamond = {HiddenCellType::kDiamond, VisibleCellType::kDiamond,
                            ElementProperties::kConsumable |
                                ElementProperties::kRounded |
                                ElementProperties::kTraversable,
                            '*'};
const Element kElDiamondFalling = {HiddenCellType::kDiamondFalling,
                                   VisibleCellType::kDiamond,
                                   ElementProperties::kConsumable, '*'};
const Element kElFireflyUp = {
    HiddenCellType::kFireflyUp, VisibleCellType::kFirefly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'};
const Element kElFireflyLeft = {
    HiddenCellType::kFireflyLeft, VisibleCellType::kFirefly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'};
const Element kElFireflyDown = {
    HiddenCellType::kFireflyDown, VisibleCellType::kFirefly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'};
const Element kElFireflyRight = {
    HiddenCellType::kFireflyRight, VisibleCellType::kFirefly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'};
const Element kElButterflyUp = {
    HiddenCellType::kButterflyUp, VisibleCellType::kButterfly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'};
const Element kElButterflyLeft = {
    HiddenCellType::kButterflyLeft, VisibleCellType::kButterfly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'};
const Element kElButterflyDown = {
    HiddenCellType::kButterflyDown, VisibleCellType::kButterfly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'};
const Element kElButterflyRight = {
    HiddenCellType::kButterflyRight, VisibleCellType::kButterfly,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'};
const Element kElBlob = {HiddenCellType::kBlob, VisibleCellType::kBlob,
                         ElementProperties::kConsumable, 'A'};
const Element kElWallBrick = {
    HiddenCellType::kWallBrick, VisibleCellType::kWallBrick,
    ElementProperties::kConsumable | ElementProperties::kRounded, 'H'};
const Element kElWallSteel = {HiddenCellType::kWallSteel,
                              VisibleCellType::kWallSteel,
                              ElementProperties::kNone, 'S'};
const Element kElWallMagicOn = {HiddenCellType::kWallMagicOn,
                                VisibleCellType::kWallMagicOn,
                                ElementProperties::kConsumable, 'M'};
const Element kElWallMagicDormant = {HiddenCellType::kWallMagicDormant,
                                     VisibleCellType::kWallMagicOff,
                                     ElementProperties::kConsumable, 'Q'};
const Element kElWallMagicExpired = {HiddenCellType::kWallMagicExpired,
                                     VisibleCellType::kWallMagicOff,
                                     ElementProperties::kConsumable, 'Q'};
const Element kElExplosionDiamond = {HiddenCellType::kExplosionDiamond,
                                     VisibleCellType::kExplosion,
                                     ElementProperties::kNone, 'E'};
const Element kElExplosionBoulder = {HiddenCellType::kExplosionBoulder,
                                     VisibleCellType::kExplosion,
                                     ElementProperties::kNone, 'E'};
const Element kElExplosionEmpty = {HiddenCellType::kExplosionEmpty,
                                   VisibleCellType::kExplosion,
                                   ElementProperties::kNone, 'E'};
const Element kElGateRedClosed = {HiddenCellType::kGateRedClosed,
                                  VisibleCellType::kGateRedClosed,
                                  ElementProperties::kNone, 'r'};
const Element kElGateRedOpen = {HiddenCellType::kGateRedOpen,
                                VisibleCellType::kGateRedOpen,
                                ElementProperties::kNone, 'R'};
const Element kElKeyRed = {HiddenCellType::kKeyRed, VisibleCellType::kKeyRed,
                           ElementProperties::kTraversable, '1'};
const Element kElGateBlueClosed = {HiddenCellType::kGateBlueClosed,
                                   VisibleCellType::kGateBlueClosed,
                                   ElementProperties::kNone, 'b'};
const Element kElGateBlueOpen = {HiddenCellType::kGateBlueOpen,
                                 VisibleCellType::kGateBlueOpen,
                                 ElementProperties::kNone, 'B'};
const Element kElKeyBlue = {HiddenCellType::kKeyBlue, VisibleCellType::kKeyBlue,
                            ElementProperties::kTraversable, '2'};
const Element kElGateGreenClosed = {HiddenCellType::kGateGreenClosed,
                                    VisibleCellType::kGateGreenClosed,
                                    ElementProperties::kNone, 'g'};
const Element kElGateGreenOpen = {HiddenCellType::kGateGreenOpen,
                                  VisibleCellType::kGateGreenOpen,
                                  ElementProperties::kNone, 'G'};
const Element kElKeyGreen = {HiddenCellType::kKeyGreen,
                             VisibleCellType::kKeyGreen,
                             ElementProperties::kTraversable, '3'};
const Element kElGateYellowClosed = {HiddenCellType::kGateYellowClosed,
                                     VisibleCellType::kGateYellowClosed,
                                     ElementProperties::kNone, 'y'};
const Element kElGateYellowOpen = {HiddenCellType::kGateYellowOpen,
                                   VisibleCellType::kGateYellowOpen,
                                   ElementProperties::kNone, 'Y'};
const Element kElKeyYellow = {HiddenCellType::kKeyYellow,
                              VisibleCellType::kKeyYellow,
                              ElementProperties::kTraversable, '4'};
const Element kElNut = {
    HiddenCellType::kNut, VisibleCellType::kNut,
    ElementProperties::kRounded | ElementProperties::kConsumable, '+'};
const Element kElNutFalling = {
    HiddenCellType::kNutFalling, VisibleCellType::kNut,
    ElementProperties::kRounded | ElementProperties::kConsumable, '+'};
const Element kElBomb = {HiddenCellType::kBomb, VisibleCellType::kBomb,
                         ElementProperties::kRounded |
                             ElementProperties::kConsumable |
                             ElementProperties::kCanExplode,
                         '^'};
const Element kElBombFalling = {
    HiddenCellType::kBombFalling, VisibleCellType::kBomb,
    ElementProperties::kRounded | ElementProperties::kConsumable |
        ElementProperties::kCanExplode,
    '^'};
const Element kElOrangeUp = {
    HiddenCellType::kOrangeUp, VisibleCellType::kOrange,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'X'};
const Element kElOrangeLeft = {
    HiddenCellType::kOrangeLeft, VisibleCellType::kOrange,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'X'};
const Element kElOrangeDown = {
    HiddenCellType::kOrangeDown, VisibleCellType::kOrange,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'X'};
const Element kElOrangeRight = {
    HiddenCellType::kOrangeRight, VisibleCellType::kOrange,
    ElementProperties::kConsumable | ElementProperties::kCanExplode, 'X'};

// Hash for Element, so we can use as a map key
struct ElementHash {
  std::size_t operator()(const Element &e) const {
    return static_cast<int>(e.cell_type) -
           static_cast<int>(HiddenCellType::kNull);
  }
};

// ----- Conversion maps -----
// Swap map for DeserializeState
const absl::flat_hash_map<int, Element> kCellTypeToElement{
    {static_cast<int>(HiddenCellType::kNull), kNullElement},
    {static_cast<int>(HiddenCellType::kAgent), kElAgent},
    {static_cast<int>(HiddenCellType::kEmpty), kElEmpty},
    {static_cast<int>(HiddenCellType::kDirt), kElDirt},
    {static_cast<int>(HiddenCellType::kStone), kElStone},
    {static_cast<int>(HiddenCellType::kStoneFalling), kElStoneFalling},
    {static_cast<int>(HiddenCellType::kDiamond), kElDiamond},
    {static_cast<int>(HiddenCellType::kDiamondFalling), kElDiamondFalling},
    {static_cast<int>(HiddenCellType::kExitClosed), kElExitClosed},
    {static_cast<int>(HiddenCellType::kExitOpen), kElExitOpen},
    {static_cast<int>(HiddenCellType::kAgentInExit), kElAgentInExit},
    {static_cast<int>(HiddenCellType::kFireflyUp), kElFireflyUp},
    {static_cast<int>(HiddenCellType::kFireflyLeft), kElFireflyLeft},
    {static_cast<int>(HiddenCellType::kFireflyDown), kElFireflyDown},
    {static_cast<int>(HiddenCellType::kFireflyRight), kElFireflyRight},
    {static_cast<int>(HiddenCellType::kButterflyUp), kElButterflyUp},
    {static_cast<int>(HiddenCellType::kButterflyLeft), kElButterflyLeft},
    {static_cast<int>(HiddenCellType::kButterflyDown), kElButterflyDown},
    {static_cast<int>(HiddenCellType::kButterflyRight), kElButterflyRight},
    {static_cast<int>(HiddenCellType::kWallBrick), kElWallBrick},
    {static_cast<int>(HiddenCellType::kWallSteel), kElWallSteel},
    {static_cast<int>(HiddenCellType::kWallMagicOn), kElWallMagicOn},
    {static_cast<int>(HiddenCellType::kWallMagicDormant), kElWallMagicDormant},
    {static_cast<int>(HiddenCellType::kWallMagicExpired), kElWallMagicExpired},
    {static_cast<int>(HiddenCellType::kBlob), kElBlob},
    {static_cast<int>(HiddenCellType::kExplosionBoulder), kElExplosionBoulder},
    {static_cast<int>(HiddenCellType::kExplosionDiamond), kElExplosionDiamond},
    {static_cast<int>(HiddenCellType::kExplosionEmpty), kElExplosionEmpty},
    {static_cast<int>(HiddenCellType::kGateRedClosed), kElGateRedClosed},
    {static_cast<int>(HiddenCellType::kGateRedOpen), kElGateRedOpen},
    {static_cast<int>(HiddenCellType::kKeyRed), kElKeyRed},
    {static_cast<int>(HiddenCellType::kGateBlueClosed), kElGateBlueClosed},
    {static_cast<int>(HiddenCellType::kGateBlueOpen), kElGateBlueOpen},
    {static_cast<int>(HiddenCellType::kKeyBlue), kElKeyBlue},
    {static_cast<int>(HiddenCellType::kGateGreenClosed), kElGateGreenClosed},
    {static_cast<int>(HiddenCellType::kGateGreenOpen), kElGateGreenOpen},
    {static_cast<int>(HiddenCellType::kKeyGreen), kElKeyGreen},
    {static_cast<int>(HiddenCellType::kGateYellowClosed), kElGateYellowClosed},
    {static_cast<int>(HiddenCellType::kGateYellowOpen), kElGateYellowOpen},
    {static_cast<int>(HiddenCellType::kKeyYellow), kElKeyYellow},
    {static_cast<int>(HiddenCellType::kNut), kElNut},
    {static_cast<int>(HiddenCellType::kNutFalling), kElNutFalling},
    {static_cast<int>(HiddenCellType::kBomb), kElBomb},
    {static_cast<int>(HiddenCellType::kBombFalling), kElBombFalling},
    {static_cast<int>(HiddenCellType::kOrangeUp), kElOrangeUp},
    {static_cast<int>(HiddenCellType::kOrangeLeft), kElOrangeLeft},
    {static_cast<int>(HiddenCellType::kOrangeDown), kElOrangeDown},
    {static_cast<int>(HiddenCellType::kOrangeRight), kElOrangeRight},
};

// Rotate actions right
const absl::flat_hash_map<int, int> kRotateRight{
    {Directions::kUp, Directions::kRight},
    {Directions::kRight, Directions::kDown},
    {Directions::kDown, Directions::kLeft},
    {Directions::kLeft, Directions::kUp},
    {Directions::kNone, Directions::kNone},
};

// Rotate actions left
const absl::flat_hash_map<int, int> kRotateLeft{
    {Directions::kUp, Directions::kLeft},
    {Directions::kLeft, Directions::kDown},
    {Directions::kDown, Directions::kRight},
    {Directions::kRight, Directions::kUp},
    {Directions::kNone, Directions::kNone},
};

// actions to strings
const absl::flat_hash_map<int, std::string> kActionsToString{
    {Directions::kUp, "up"},     {Directions::kLeft, "left"},
    {Directions::kDown, "down"}, {Directions::kRight, "right"},
    {Directions::kNone, "none"},
};

// directions to offsets (col, row)
const absl::flat_hash_map<int, std::pair<int, int>> kDirectionOffsets{
    {Directions::kUp, {0, -1}},   {Directions::kUpLeft, {-1, -1}},
    {Directions::kLeft, {-1, 0}}, {Directions::kDownLeft, {-1, 1}},
    {Directions::kDown, {0, 1}},  {Directions::kDownRight, {1, 1}},
    {Directions::kRight, {1, 0}}, {Directions::kUpRight, {1, -1}},
    {Directions::kNone, {0, 0}},
};

// Directions to fireflys
const absl::flat_hash_map<int, Element> kDirectionToFirefly{
    {Directions::kUp, kElFireflyUp},
    {Directions::kLeft, kElFireflyLeft},
    {Directions::kDown, kElFireflyDown},
    {Directions::kRight, kElFireflyRight},
};

// Firefly to directions
const absl::flat_hash_map<Element, int, ElementHash> kFireflyToDirection{
    {kElFireflyUp, Directions::kUp},
    {kElFireflyLeft, Directions::kLeft},
    {kElFireflyDown, Directions::kDown},
    {kElFireflyRight, Directions::kRight},
};

// Directions to butterflys
const absl::flat_hash_map<int, Element> kDirectionToButterfly{
    {Directions::kUp, kElButterflyUp},
    {Directions::kLeft, kElButterflyLeft},
    {Directions::kDown, kElButterflyDown},
    {Directions::kRight, kElButterflyRight},
};

// Butterfly to directions
const absl::flat_hash_map<Element, int, ElementHash> kButterflyToDirection{
    {kElButterflyUp, Directions::kUp},
    {kElButterflyLeft, Directions::kLeft},
    {kElButterflyDown, Directions::kDown},
    {kElButterflyRight, Directions::kRight},
};

// Orange to directions
const absl::flat_hash_map<Element, int, ElementHash> kOrangeToDirection{
    {kElOrangeUp, Directions::kUp},
    {kElOrangeLeft, Directions::kLeft},
    {kElOrangeDown, Directions::kDown},
    {kElOrangeRight, Directions::kRight},
};

// Direction to Orange
const absl::flat_hash_map<int, Element> kDirectionToOrange{
    {Directions::kUp, kElOrangeUp},
    {Directions::kLeft, kElOrangeLeft},
    {Directions::kDown, kElOrangeDown},
    {Directions::kRight, kElOrangeRight},
};

// Element explosion maps
const absl::flat_hash_map<Element, Element, ElementHash> kElementToExplosion{
    {kElFireflyUp, kElExplosionEmpty},
    {kElFireflyLeft, kElExplosionEmpty},
    {kElFireflyDown, kElExplosionEmpty},
    {kElFireflyRight, kElExplosionEmpty},
    {kElButterflyUp, kElExplosionDiamond},
    {kElButterflyLeft, kElExplosionDiamond},
    {kElButterflyDown, kElExplosionDiamond},
    {kElButterflyRight, kElExplosionDiamond},
    {kElAgent, kElExplosionEmpty},
    {kElBomb, kElExplosionEmpty},
    {kElBombFalling, kElExplosionEmpty},
    {kElOrangeUp, kElExplosionEmpty},
    {kElOrangeLeft, kElExplosionEmpty},
    {kElOrangeDown, kElExplosionEmpty},
    {kElOrangeRight, kElExplosionEmpty},
};

// Explosions back to elements
const absl::flat_hash_map<Element, Element, ElementHash> kExplosionToElement{
    {kElExplosionDiamond, kElDiamond},
    {kElExplosionBoulder, kElStone},
    {kElExplosionEmpty, kElEmpty},
};

// Magic wall conversion map
const absl::flat_hash_map<Element, Element, ElementHash> kMagicWallConversion{
    {kElStoneFalling, kElDiamondFalling},
    {kElDiamondFalling, kElStoneFalling},
};

// Gem point maps
const absl::flat_hash_map<Element, int, ElementHash> kGemPoints{
    {kElDiamond, 10},
    {kElDiamondFalling, 10},
};

// Gate open conversion map
const absl::flat_hash_map<Element, Element, ElementHash> kGateOpenMap{
    {kElGateRedClosed, kElGateRedOpen},
    {kElGateBlueClosed, kElGateBlueOpen},
    {kElGateGreenClosed, kElGateGreenOpen},
    {kElGateYellowClosed, kElGateYellowOpen},
};
// Gate key map
const absl::flat_hash_map<Element, Element, ElementHash> kKeyToGate{
    {kElKeyRed, kElGateRedClosed},
    {kElKeyBlue, kElGateBlueClosed},
    {kElKeyGreen, kElGateGreenClosed},
    {kElKeyYellow, kElGateYellowClosed},
};

// Stationary to falling
const absl::flat_hash_map<Element, Element, ElementHash> kElToFalling{
    {kElDiamond, kElDiamondFalling},
    {kElStone, kElStoneFalling},
    {kElNut, kElNutFalling},
    {kElBomb, kElBombFalling},
};

// Default parameters.
constexpr int kDefaultMagicWallSteps =
    140;  // Number of steps before magic walls expire
constexpr int kDefaultBlobChance =
    20;  // Chance to spawn another blob (out of 256)
constexpr double kDefaultBlobMaxPercentage =
    0.16;  // Maximum number of blob before they collapse (percentage of map
           // size)
constexpr bool kDefaultObsShowIDs =
    false;  // Flag to show IDs instead of one-hot encoding

// Facts about the game
const GameType kGameType{
    /*short_name=*/"stones_and_gems",
    /*long_name=*/"Stones and Gems",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/1,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"obs_show_ids", GameParameter(kDefaultObsShowIDs)},
     {"magic_wall_steps", GameParameter(kDefaultMagicWallSteps)},
     {"blob_chance", GameParameter(kDefaultBlobChance)},
     {"blob_max_percentage", GameParameter(kDefaultBlobMaxPercentage)},
     {"rng_seed", GameParameter(0)},
     {"grid", GameParameter(std::string(kDefaultGrid))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new StonesNGemsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string StonesNGemsState::ActionToString(Player player,
                                             Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome: ", move_id);
  } else {
    SPIEL_CHECK_GE(move_id, 0);
    SPIEL_CHECK_LT(move_id, kNumActions);
    if (kActionsToString.find(move_id) == kActionsToString.end()) {
      SpielFatalError("Unknown move_id");
    }
    return kActionsToString.at(move_id);
  }
}

bool StonesNGemsState::IsTerminal() const {
  // Time complete or the agent exploded
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElAgent);
  return steps_remaining_ <= 0 || it == grid_.elements.end();
}

std::vector<double> StonesNGemsState::Returns() const {
  // Sum of rewards, and should agree with Rewards()
  return {static_cast<double>(sum_reward_)};
}

std::vector<double> StonesNGemsState::Rewards() const {
  // reward for most recent state transition
  return {static_cast<double>(current_reward_)};
}

std::string StonesNGemsState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsChanceNode()) {
    return "ChanceNode -- no observation";
  }
  return ToString();
}

void StonesNGemsState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 3-d tensor.
  TensorView<3> view(
      values, {kNumVisibleCellType, grid_.num_rows, grid_.num_cols}, true);

  // No observations at chance nodes.
  if (IsChanceNode()) {
    std::fill(values.begin(), values.end(), 0);
    return;
  }

  int i = 0;
  for (int row = 0; row < grid_.num_rows; ++row) {
    for (int col = 0; col < grid_.num_cols; ++col) {
      int channel = static_cast<int>(grid_.elements[i].visible_type);
      view[{channel, row, col}] = obs_show_ids_ ? grid_.ids[i] : 1.0;
      ++i;
    }
  }
}

int StonesNGemsState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

std::mt19937 *StonesNGemsState::rng() {
  return static_cast<const StonesNGemsGame *>(game_.get())->rng();
}

// Element helper functions
namespace {

bool IsActionHorz(int action) {
  return action == Directions::kLeft || action == Directions::kRight;
}

bool IsFirefly(const Element &element) {
  return element == kElFireflyUp || element == kElFireflyLeft ||
         element == kElFireflyDown || element == kElFireflyRight;
}

bool IsButterfly(const Element &element) {
  return element == kElButterflyUp || element == kElButterflyLeft ||
         element == kElButterflyDown || element == kElButterflyRight;
}

bool IsOrange(const Element &element) {
  return element == kElOrangeUp || element == kElOrangeLeft ||
         element == kElOrangeDown || element == kElOrangeRight;
}

bool IsExplosion(const Element &element) {
  return element == kElExplosionBoulder || element == kElExplosionDiamond ||
         element == kElExplosionEmpty;
}

bool IsMagicWall(const Element &element) {
  return element == kElWallMagicDormant || element == kElWallMagicExpired ||
         element == kElWallMagicOn;
}

bool IsOpenGate(const Element &element) {
  return element == kElGateRedOpen || element == kElGateBlueOpen ||
         element == kElGateGreenOpen || element == kElGateYellowOpen;
}

bool IsKey(const Element &element) {
  return element == kElKeyRed || element == kElKeyBlue ||
         element == kElKeyGreen || element == kElKeyYellow;
}

}  // namespace

// ---------- Game dynamic function ----------

// Given an index and action, get the new flat index
int StonesNGemsState::IndexFromAction(int index, int action) const {
  int col = index % grid_.num_cols;
  int row = (index - col) / grid_.num_cols;
  std::pair<int, int> offsets = kDirectionOffsets.at(action);
  col += offsets.first;
  row += offsets.second;
  return (grid_.num_cols * row) + col;
}

// Check if the index with a given action step will remain in bounds
bool StonesNGemsState::InBounds(int index, int action) const {
  int col = index % grid_.num_cols;
  int row = (index - col) / grid_.num_cols;
  std::pair<int, int> offsets = kDirectionOffsets.at(action);
  col += offsets.first;
  row += offsets.second;
  return col >= 0 && col < grid_.num_cols && row >= 0 && row < grid_.num_rows;
}

// Check if the index after applying action contains the given element
bool StonesNGemsState::IsType(int index, Element element, int action) const {
  int new_index = IndexFromAction(index, action);
  return InBounds(index, action) && grid_.elements[new_index] == element;
}

// Check if the index after applying action has an element with the given
// property
bool StonesNGemsState::HasProperty(int index, int property, int action) const {
  int new_index = IndexFromAction(index, action);
  return InBounds(index, action) &&
         ((grid_.elements[new_index].properties & property) > 0);
}

// Move the element given the action, and set the old index to empty
void StonesNGemsState::MoveItem(int index, int action) {
  int new_index = IndexFromAction(index, action);
  grid_.elements[new_index] = grid_.elements[index];
  grid_.ids[new_index] = grid_.ids[index];
  grid_.elements[new_index].has_updated = true;
  grid_.elements[index] = kElEmpty;
  grid_.ids[index] = ++id_counter_;
}

// Set the new index to the given element
void StonesNGemsState::SetItem(int index, Element element, int id, int action) {
  int new_index = IndexFromAction(index, action);
  grid_.elements[new_index] = element;
  grid_.ids[new_index] = id;
  grid_.elements[new_index].has_updated = true;
}

// Get the item after applying the action to the index
Element StonesNGemsState::GetItem(int index, int action) const {
  return grid_.elements[IndexFromAction(index, action)];
}

// Check if the element is adjacent to and cell around the given index
bool StonesNGemsState::IsTypeAdjacent(int index, Element element) const {
  return IsType(index, element, Directions::kUp) ||
         IsType(index, element, Directions::kLeft) ||
         IsType(index, element, Directions::kDown) ||
         IsType(index, element, Directions::kRight);
}

// Can roll left if sitting on rounded element, left and bottom left clear
bool StonesNGemsState::CanRollLeft(int index) const {
  return HasProperty(index, ElementProperties::kRounded, Directions::kDown) &&
         IsType(index, kElEmpty, Directions::kLeft) &&
         IsType(index, kElEmpty, Directions::kDownLeft);
}

// Can roll right if sitting on rounded element, right and bottom right clear
bool StonesNGemsState::CanRollRight(int index) const {
  return HasProperty(index, ElementProperties::kRounded, Directions::kDown) &&
         IsType(index, kElEmpty, Directions::kRight) &&
         IsType(index, kElEmpty, Directions::kDownRight);
}

// Roll the item to the left
void StonesNGemsState::RollLeft(int index, Element element) {
  SetItem(index, element, grid_.ids[index]);
  MoveItem(index, Directions::kLeft);
}

// Roll the item to the right
void StonesNGemsState::RollRight(int index, Element element) {
  SetItem(index, element, grid_.ids[index]);
  MoveItem(index, Directions::kRight);
}

// Push the item
void StonesNGemsState::Push(int index, Element stationary, Element falling,
                            int action) {
  int new_index = IndexFromAction(index, action);
  // Check if same direction past element is empty so that theres room to push
  if (IsType(new_index, kElEmpty, action)) {
    // Check if the element will become stationary or falling
    int next_index = IndexFromAction(new_index, action);
    bool is_empty = IsType(next_index, kElEmpty, Directions::kDown);
    SetItem(new_index, is_empty ? falling : stationary, grid_.ids[new_index],
            action);
    // Move the agent
    MoveItem(index, action);
  }
}

// Move the item through the magic wall
void StonesNGemsState::MoveThroughMagic(int index, Element element) {
  // Check if magic wall is still active
  if (magic_wall_steps_ <= 0) {
    return;
  }
  magic_active_ = true;
  int index_below = IndexFromAction(index, Directions::kDown);
  // Need to ensure cell below magic wall is empty (so item can pass through)
  if (IsType(index_below, kElEmpty, Directions::kDown)) {
    SetItem(index, kElEmpty, ++id_counter_);
    SetItem(index_below, element, ++id_counter_, Directions::kDown);
  }
}

// Explode the item
void StonesNGemsState::Explode(int index, Element element, int action) {
  int new_index = IndexFromAction(index, action);
  auto it = kElementToExplosion.find(GetItem(new_index));
  Element ex =
      (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
  SetItem(new_index, element, ++id_counter_);
  // Recursively check all directions for chain explosions
  for (int dir = 0; dir < kNumDirections; ++dir) {
    if (dir == Directions::kNone || !InBounds(new_index, dir)) {
      continue;
    }
    if (HasProperty(new_index, ElementProperties::kCanExplode, dir)) {
      Explode(new_index, ex, dir);
    } else if (HasProperty(new_index, ElementProperties::kConsumable, dir)) {
      SetItem(new_index, ex, ++id_counter_, dir);
    }
  }
}

void StonesNGemsState::OpenGate(Element element) {
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), element);
  if (it != grid_.elements.end()) {
    int index = std::distance(grid_.elements.begin(), it);
    SetItem(index, kGateOpenMap.at(GetItem(index)), grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateStone(int index) {
  // Boulder falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElStoneFalling, grid_.ids[index]);
    UpdateStoneFalling(index);
  } else if (CanRollLeft(index)) {  // Roll left/right if possible
    RollLeft(index, kElStoneFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElStoneFalling);
  }
}

void StonesNGemsState::UpdateStoneFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (HasProperty(index, ElementProperties::kCanExplode,
                         Directions::kDown)) {
    // Falling stones can cause elements to explode
    auto it = kElementToExplosion.find(GetItem(index, Directions::kDown));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex, Directions::kDown);
  } else if (IsType(index, kElWallMagicOn, Directions::kDown) ||
             IsType(index, kElWallMagicDormant, Directions::kDown)) {
    MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
  } else if (IsType(index, kElNut, Directions::kDown)) {
    // Falling on a nut, crack it open to reveal a diamond!
    SetItem(index, kElDiamond, ++id_counter_, Directions::kDown);
  } else if (IsType(index, kElNut, Directions::kDown)) {
    // Falling on a bomb, explode!
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else if (CanRollLeft(index)) {  // Roll left/right
    RollLeft(index, kElStoneFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElStoneFalling);
  } else {
    // Default options is for falling stones to become stationary
    SetItem(index, kElStone, grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateDiamond(int index) {
  // Diamond falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElDiamondFalling, grid_.ids[index]);
    UpdateDiamondFalling(index);
  } else if (CanRollLeft(index)) {  // Roll left/right if possible
    RollLeft(index, kElDiamondFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElDiamondFalling);
  }
}

void StonesNGemsState::UpdateDiamondFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (HasProperty(index, ElementProperties::kCanExplode,
                         Directions::kDown) &&
             !IsType(index, kElBomb, Directions::kDown) &&
             !IsType(index, kElBombFalling, Directions::kDown)) {
    // Falling diamonds can cause elements to explode (but not bombs)
    auto it = kElementToExplosion.find(GetItem(index, Directions::kDown));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex, Directions::kDown);
  } else if (IsType(index, kElWallMagicOn, Directions::kDown) ||
             IsType(index, kElWallMagicDormant, Directions::kDown)) {
    MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
  } else if (CanRollLeft(index)) {  // Roll left/right
    RollLeft(index, kElDiamondFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElDiamondFalling);
  } else {
    // Default options is for falling diamond to become stationary
    SetItem(index, kElDiamond, grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateNut(int index) {
  // Nut falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElNutFalling, grid_.ids[index]);
    UpdateNutFalling(index);
  } else if (CanRollLeft(index)) {  // Roll left/right
    RollLeft(index, kElNutFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElNutFalling);
  }
}

void StonesNGemsState::UpdateNutFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (CanRollLeft(index)) {  // Roll left/right
    RollLeft(index, kElNutFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElNutFalling);
  } else {
    // Default options is for falling nut to become stationary
    SetItem(index, kElNut, grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateBomb(int index) {
  // Bomb falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElBombFalling, grid_.ids[index]);
    UpdateBombFalling(index);
  } else if (CanRollLeft(index)) {  // Roll left/right
    RollLeft(index, kElBomb);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBomb);
  }
}

void StonesNGemsState::UpdateBombFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (CanRollLeft(index)) {  // Roll left/right
    RollLeft(index, kElBombFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBombFalling);
  } else {
    // Default options is for bomb to explode if stopped falling
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  }
}

void StonesNGemsState::UpdateExit(int index) {
  // Open exit if enough gems collected
  if (gems_collected_ >= gems_required_) {
    SetItem(index, kElExitOpen, grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateAgent(int index, int action) {
  if (IsType(index, kElEmpty, action) || IsType(index, kElDirt, action)) {
    // Move if empty/dirt
    MoveItem(index, action);
  } else if (IsType(index, kElDiamond, action) ||
             IsType(index, kElDiamondFalling, action)) {
    // Collect gems
    ++gems_collected_;
    current_reward_ += kGemPoints.at(GetItem(index, action));
    sum_reward_ += kGemPoints.at(GetItem(index, action));
    MoveItem(index, action);
  } else if (IsActionHorz(action) && (IsType(index, kElStone, action) ||
                                      IsType(index, kElNut, action) ||
                                      IsType(index, kElBomb, action))) {
    // Push stone, nut, or bomb if action is horizontal
    Push(index, GetItem(index, action), kElToFalling.at(GetItem(index, action)),
         action);
  } else if (IsKey(GetItem(index, action))) {
    // Collecting key, set gate open
    OpenGate(kKeyToGate.at(GetItem(index, action)));
    MoveItem(index, action);
  } else if (IsOpenGate(GetItem(index, action))) {
    // Walking through an open gate, with traversable element on other side
    int index_gate = IndexFromAction(index, action);
    if (HasProperty(index_gate, ElementProperties::kTraversable, action)) {
      // Correct for landing on traversable elements
      if (IsType(index_gate, kElDiamond, action)) {
        ++gems_collected_;
        current_reward_ += kGemPoints.at(GetItem(index_gate, action));
        sum_reward_ += kGemPoints.at(GetItem(index_gate, action));
      } else if (IsKey(GetItem(index_gate, action))) {
        OpenGate(kKeyToGate.at(GetItem(index_gate, action)));
      }
      SetItem(index_gate, kElAgent, grid_.ids[index], action);
      SetItem(index, kElEmpty, ++id_counter_);
    }
  } else if (IsType(index, kElExitOpen, action)) {
    // Walking into exit after collecting enough gems
    MoveItem(index, action);
    SetItem(index, kElAgentInExit, ++id_counter_, action);
    current_reward_ += steps_remaining_;
    sum_reward_ += steps_remaining_;
  }
}

void StonesNGemsState::UpdateFirefly(int index, int action) {
  int new_dir = kRotateLeft.at(action);
  if (IsTypeAdjacent(index, kElAgent) || IsTypeAdjacent(index, kElBlob)) {
    // Explode if touching the agent/blob
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else if (IsType(index, kElEmpty, new_dir)) {
    // Fireflies always try to rotate left, otherwise continue forward
    SetItem(index, kDirectionToFirefly.at(new_dir), grid_.ids[index]);
    MoveItem(index, new_dir);
  } else if (IsType(index, kElEmpty, action)) {
    SetItem(index, kDirectionToFirefly.at(action), grid_.ids[index]);
    MoveItem(index, action);
  } else {
    // No other options, rotate right
    SetItem(index, kDirectionToFirefly.at(kRotateRight.at(action)),
            grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateButterfly(int index, int action) {
  int new_dir = kRotateRight.at(action);
  if (IsTypeAdjacent(index, kElAgent) || IsTypeAdjacent(index, kElBlob)) {
    // Explode if touching the agent/blob
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else if (IsType(index, kElEmpty, new_dir)) {
    // Butterflies always try to rotate right, otherwise continue forward
    SetItem(index, kDirectionToButterfly.at(new_dir), grid_.ids[index]);
    MoveItem(index, new_dir);
  } else if (IsType(index, kElEmpty, action)) {
    SetItem(index, kDirectionToButterfly.at(action), grid_.ids[index]);
    MoveItem(index, action);
  } else {
    // No other options, rotate right
    SetItem(index, kDirectionToButterfly.at(kRotateLeft.at(action)),
            grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateOrange(int index, int action) {
  if (IsType(index, kElEmpty, action)) {
    // Continue moving in direction
    MoveItem(index, action);
  } else if (IsTypeAdjacent(index, kElAgent)) {
    // Run into the agent, explode!
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex =
        (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else {
    // Blocked, roll for new direction
    std::vector<int> open_dirs;
    for (int dir = 0; dir < kNumActions; ++dir) {
      if (dir == Directions::kNone || !InBounds(index, dir)) {
        continue;
      }
      if (IsType(index, kElEmpty, dir)) {
        open_dirs.push_back(dir);
      }
    }
    // Roll available directions
    if (!open_dirs.empty()) {
      int new_dir = open_dirs[(*rng())() % open_dirs.size()];
      SetItem(index, kDirectionToOrange.at(new_dir), grid_.ids[index]);
    }
  }
}

void StonesNGemsState::UpdateMagicWall(int index) {
  // Dorminant, active, then expired once time runs out
  if (magic_active_) {
    SetItem(index, kElWallMagicOn, grid_.ids[index]);
  } else if (magic_wall_steps_ > 0) {
    SetItem(index, kElWallMagicDormant, grid_.ids[index]);
  } else {
    SetItem(index, kElWallMagicExpired, grid_.ids[index]);
  }
}

void StonesNGemsState::UpdateBlob(int index) {
  // Replace blobs if swap element set
  if (blob_swap_ != kNullElement) {
    SetItem(index, blob_swap_, ++id_counter_);
    return;
  }
  ++blob_size_;
  // Check if at least one tile blob can grow to
  if (IsTypeAdjacent(index, kElEmpty) || IsTypeAdjacent(index, kElDirt)) {
    blob_enclosed_ = false;
  }
  // Roll if to grow and direction
  bool will_grow = ((*rng())() % 256) < blob_chance_;
  int grow_dir = (*rng())() % kNumActions;
  if (will_grow &&
      (IsType(index, kElEmpty, grow_dir) || IsType(index, kElDirt, grow_dir))) {
    SetItem(index, kElBlob, grow_dir, ++id_counter_);
  }
}

void StonesNGemsState::UpdateExplosions(int index) {
  SetItem(index, kExplosionToElement.at(GetItem(index)), ++id_counter_);
}

void StonesNGemsState::StartScan() {
  // Update global flags
  --steps_remaining_;
  current_reward_ = 0;
  blob_size_ = 0;
  blob_enclosed_ = true;
  // Reset element flags
  for (auto &e : grid_.elements) {
    e.has_updated = false;
  }
}

void StonesNGemsState::EndScan() {
  // Check if blob dead/closed/size
  if (blob_swap_ == kNullElement) {
    if (blob_enclosed_) {
      // blobs become diamonds if enclosed
      blob_swap_ = kElDiamond;
    } else if (blob_size_ > blob_max_size_) {
      // blobs become stones is they grow too large
      blob_swap_ = kElStone;
    }
  }
  // Reduce magic wall steps if active
  if (magic_active_) {
    magic_wall_steps_ = std::max(magic_wall_steps_ - 1, 0);
  }
  // Check if still active
  magic_active_ = (magic_active_ && magic_wall_steps_ > 0);
}

void StonesNGemsState::DoApplyAction(Action move) {
  if (cur_player_ == kChancePlayerId) {
    // Check each cell and apply respective dynamics function
    for (int index = 0; index < grid_.num_cols * grid_.num_rows; ++index) {
      Element &e = grid_.elements[index];
      if (e.has_updated) {
        continue;
      } else if (e == kElStone) {
        UpdateStone(index);
      } else if (e == kElStoneFalling) {
        UpdateStoneFalling(index);
      } else if (e == kElDiamond) {
        UpdateDiamond(index);
      } else if (e == kElDiamondFalling) {
        UpdateDiamondFalling(index);
      } else if (e == kElNut) {
        UpdateNut(index);
      } else if (e == kElNutFalling) {
        UpdateNutFalling(index);
      } else if (e == kElBomb) {
        UpdateBomb(index);
      } else if (e == kElBombFalling) {
        UpdateBombFalling(index);
      } else if (e == kElExitClosed) {
        UpdateExit(index);
      } else if (IsButterfly(e)) {
        UpdateButterfly(index, kButterflyToDirection.at(e));
      } else if (IsFirefly(e)) {
        UpdateFirefly(index, kFireflyToDirection.at(e));
      } else if (IsOrange(e)) {
        UpdateOrange(index, kOrangeToDirection.at(e));
      } else if (IsMagicWall(e)) {
        UpdateMagicWall(index);
      } else if (e == kElBlob) {
        UpdateBlob(index);
      } else if (IsExplosion(e)) {
        UpdateExplosions(index);
      }
    }
    EndScan();
    cur_player_ = 0;
  } else {
    StartScan();
    // Find where the agent is, and update its position
    auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElAgent);
    int index = std::distance(grid_.elements.begin(), it);
    UpdateAgent(index, move);
    cur_player_ = kChancePlayerId;
  }
}

std::vector<Action> StonesNGemsState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    return {Directions::kNone, Directions::kUp, Directions::kRight,
            Directions::kDown, Directions::kLeft};
  }
}

std::vector<std::pair<Action, double>> StonesNGemsState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes = {std::make_pair(0, 1.0)};
  return outcomes;
}

std::string StonesNGemsState::ToString() const {
  if (IsChanceNode()) {
    return "chance node";
  }
  std::string out_str;
  int col_counter = 0;
  for (const auto el : grid_.elements) {
    ++col_counter;
    out_str += el.id;
    if (col_counter == grid_.num_cols) {
      absl::StrAppend(&out_str, "\n");
      col_counter = 0;
    }
  }
  absl::StrAppend(&out_str, "time left: ", steps_remaining_, ", ");
  absl::StrAppend(&out_str, "gems required: ", gems_required_, ", ");
  absl::StrAppend(&out_str, "gems collectred: ", gems_collected_);
  return out_str;
}

std::string StonesNGemsState::Serialize() const {
  std::string out_str;
  // grid properties
  absl::StrAppend(&out_str, grid_.num_cols, ",");
  absl::StrAppend(&out_str, grid_.num_rows, ",");
  absl::StrAppend(&out_str, steps_remaining_, ",");
  absl::StrAppend(&out_str, magic_wall_steps_, ",");
  absl::StrAppend(&out_str, magic_active_, ",");
  absl::StrAppend(&out_str, blob_max_size_, ",");
  absl::StrAppend(&out_str, blob_size_, ",");
  absl::StrAppend(&out_str, blob_chance_, ",");
  absl::StrAppend(&out_str, static_cast<int>(blob_swap_.cell_type), ",");
  absl::StrAppend(&out_str, blob_enclosed_, ",");
  absl::StrAppend(&out_str, gems_required_, ",");
  absl::StrAppend(&out_str, gems_collected_, ",");
  absl::StrAppend(&out_str, current_reward_, ",");
  absl::StrAppend(&out_str, sum_reward_, ",");
  absl::StrAppend(&out_str, obs_show_ids_, ",");
  absl::StrAppend(&out_str, id_counter_, ",");
  absl::StrAppend(&out_str, cur_player_, "\n");
  // grid contents
  int col_counter = 0;
  for (std::size_t i = 0; i < grid_.elements.size(); ++i) {
    ++col_counter;
    absl::StrAppend(&out_str, static_cast<int>(grid_.elements[i].cell_type),
                    ",");
    absl::StrAppend(&out_str, grid_.ids[i], ",");
    if (col_counter == grid_.num_cols) {
      out_str.pop_back();
      absl::StrAppend(&out_str, "\n");
      col_counter = 0;
    }
  }
  // remove trailing newline
  out_str.pop_back();
  return out_str;
}

std::unique_ptr<State> StonesNGemsState::Clone() const {
  return std::unique_ptr<State>(new StonesNGemsState(*this));
}

StonesNGemsState::StonesNGemsState(
    std::shared_ptr<const Game> game, int steps_remaining, int magic_wall_steps,
    bool magic_active, int blob_max_size, int blob_size, int blob_chance,
    Element blob_swap, bool blob_enclosed, int gems_required,
    int gems_collected, int current_reward, int sum_reward, Grid grid,
    bool obs_show_ids, int id_counter, Player player)
    : State(game),
      steps_remaining_(steps_remaining),
      magic_wall_steps_(magic_wall_steps),
      magic_active_(magic_active),
      blob_max_size_(blob_max_size),
      blob_size_(blob_size),
      blob_chance_(blob_chance),
      blob_swap_(blob_swap),
      blob_enclosed_(blob_enclosed),
      gems_required_(gems_required),
      gems_collected_(gems_collected),
      current_reward_(current_reward),
      sum_reward_(sum_reward),
      grid_(grid),
      obs_show_ids_(obs_show_ids),
      id_counter_(id_counter),
      cur_player_(player) {}

// ------ Game -------

std::unique_ptr<State> StonesNGemsGame::DeserializeState(
    const std::string &str) const {
  // empty string
  if (str.empty()) {
    return NewInitialState();
  }
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  if (lines.size() < 2) {
    SpielFatalError("Empty map string passed.");
  }
  // Read grid properties
  std::vector<std::string> property_line = absl::StrSplit(lines[0], ',');
  Grid grid;
  int steps_remaining, magic_wall_steps, blob_max_size, blob_size, blob_chance,
      gems_required, gems_collected, current_reward, sum_reward, id_counter,
      cur_player, magic_active, blob_enclosed, obs_show_ids, blob_swap;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[0], &grid.num_cols));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[1], &grid.num_rows));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[2], &steps_remaining));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[3], &magic_wall_steps));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[4], &magic_active));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[5], &blob_max_size));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[6], &blob_size));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[7], &blob_chance));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[8], &blob_swap));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[9], &blob_enclosed));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[10], &gems_required));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[11], &gems_collected));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[12], &current_reward));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[13], &sum_reward));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[14], &obs_show_ids));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[15], &id_counter));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[16], &cur_player));
  // Set grid elements
  for (std::size_t i = 1; i < lines.size(); ++i) {
    std::vector<std::string> grid_line = absl::StrSplit(lines[i], ',');
    // Check for proper number of columns
    if (grid_line.size() != grid.num_cols * 2) {
      SpielFatalError(absl::StrCat("Grid line ", i - 1,
                                   "doesn't have correct number of elements."));
    }
    // Check each element in row
    // for (const auto &type : grid_line) {
    for (std::size_t i = 0; i < grid_line.size() / 2; ++i) {
      // Element
      auto it = kCellTypeToElement.find(std::stoi(grid_line[2 * i]));
      if (it != kCellTypeToElement.end()) {
        grid.elements.push_back(it->second);
      } else {
        SpielFatalError(absl::StrCat("Unknown element id: ", grid_line[2 * i]));
      }
      // ID
      grid.ids.push_back(std::stoi(grid_line[2 * i + 1]));
    }
  }
  // Ensure we read proper number of rows
  if (lines.size() - 1 != grid.num_rows) {
    SpielFatalError(absl::StrCat("Incorrect number of rows, got ",
                                 lines.size() - 1, " but need ",
                                 grid.num_rows));
  }
  // Ensure the agent exists in the map
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElAgent);
  if (it == grid_.elements.end()) {
    SpielFatalError("Grid string doesn't contain the agent.");
  }

  return std::unique_ptr<State>(new StonesNGemsState(
      shared_from_this(), steps_remaining, magic_wall_steps, magic_active,
      blob_max_size, blob_size, blob_chance, kCellTypeToElement.at(blob_swap),
      blob_enclosed, gems_required, gems_collected, current_reward, sum_reward,
      grid, obs_show_ids, id_counter, cur_player));
}

std::string StonesNGemsGame::GetRNGState() const {
  std::ostringstream rng_stream;
  rng_stream << rng_;
  return rng_stream.str();
}

void StonesNGemsGame::SetRNGState(const std::string &rng_state) const {
  if (rng_state.empty()) return;
  std::istringstream rng_stream(rng_state);
  rng_stream >> rng_;
}

int StonesNGemsGame::NumDistinctActions() const { return kNumActions; }

// There is arbitrarily chosen number to ensure the game is finite.
int StonesNGemsGame::MaxGameLength() const { return max_steps_; }

int StonesNGemsGame::NumPlayers() const { return 1; }

double StonesNGemsGame::MinUtility() const { return 0; }

double StonesNGemsGame::MaxUtility() const {
  // Max utility really depends on the number of gems in the map,
  // so we have a lose upper bound.
  // Diamonds give points
  // Boulders can be converted to diamonds
  // Butterflies can drop diamonds
  // Nuts drop diamonds if cracked
  double max_util = max_steps_;
  max_util +=
      kGemPoints.at(kElDiamond) *
      std::count(grid_.elements.begin(), grid_.elements.end(), kElDiamond);
  max_util += kGemPoints.at(kElDiamond) * std::count(grid_.elements.begin(),
                                                     grid_.elements.end(),
                                                     kElDiamondFalling);
  max_util +=
      std::count(grid_.elements.begin(), grid_.elements.end(), kElStone);
  max_util +=
      std::count(grid_.elements.begin(), grid_.elements.end(), kElStoneFalling);
  max_util += 9 * std::count(grid_.elements.begin(), grid_.elements.end(),
                             kElButterflyUp);
  max_util += 9 * std::count(grid_.elements.begin(), grid_.elements.end(),
                             kElButterflyLeft);
  max_util += 9 * std::count(grid_.elements.begin(), grid_.elements.end(),
                             kElButterflyDown);
  max_util += 9 * std::count(grid_.elements.begin(), grid_.elements.end(),
                             kElButterflyRight);
  max_util += std::count(grid_.elements.begin(), grid_.elements.end(), kElNut);
  max_util +=
      std::count(grid_.elements.begin(), grid_.elements.end(), kElNutFalling);
  return max_util;
}

std::vector<int> StonesNGemsGame::ObservationTensorShape() const {
  return {kNumVisibleCellType, grid_.num_rows, grid_.num_cols};
}

Grid StonesNGemsGame::ParseGrid(const std::string &grid_string,
                                double blob_max_percentage) {
  Grid grid;

  std::vector<std::string> lines = absl::StrSplit(grid_string, '\n');
  if (lines.size() < 2) {
    SpielFatalError("Empty map string passed.");
  }
  // Parse first line which contains level properties
  std::vector<std::string> property_line = absl::StrSplit(lines[0], '|');
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[0], &grid.num_cols));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[1], &grid.num_rows));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[2], &max_steps_));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(property_line[3], &gems_required_));

  // Parse grid contents
  for (std::size_t i = 1; i < lines.size(); ++i) {
    // Check for proper number of columns
    std::vector<std::string> grid_line = absl::StrSplit(lines[i], '|');
    if (grid_line.size() != grid.num_cols) {
      SpielFatalError(absl::StrCat(
          "Grid line ", i - 1, " doesn't have correct number of elements.",
          " Received ", grid_line.size(), ", expected ", grid.num_cols));
    }
    // Check each element in row
    for (const auto &type : grid_line) {
      auto it = kCellTypeToElement.find(std::stoi(type));
      if (it != kCellTypeToElement.end()) {
        grid.elements.push_back(it->second);
      } else {
        SpielFatalError(absl::StrCat("Unknown element id: ", type));
      }
    }
  }
  // Ensure we read proper number of rows
  if (lines.size() - 1 != grid.num_rows) {
    SpielFatalError(absl::StrCat("Incorrect number of rows, received ",
                                 lines.size() - 1, ", expected ",
                                 grid.num_rows));
  }
  // Ensure the agent exists in the map
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElAgent);
  if (it == grid_.elements.end()) {
    SpielFatalError("Grid string doesn't contain the agent.");
  }
  blob_max_size_ = (int)(grid_.num_cols * grid_.num_rows * blob_max_percentage);

  // Initialize the grid element IDs
  grid_.ids.clear();
  for (std::size_t i = 0; i < grid.elements.size(); ++i) {
    grid_.ids.push_back(i + 1);
  }

  return grid;
}

StonesNGemsGame::StonesNGemsGame(const GameParameters &params)
    : Game(kGameType, params),
      obs_show_ids_(ParameterValue<bool>("obs_show_ids")),
      magic_wall_steps_(ParameterValue<int>("magic_wall_steps")),
      blob_chance_(ParameterValue<int>("blob_chance")),
      rng_seed_(ParameterValue<int>("rng_seed")),
      grid_(ParseGrid(ParameterValue<std::string>("grid"),
                      ParameterValue<double>("blob_max_percentage"))) {}

}  // namespace stones_and_gems
}  // namespace open_spiel
