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

#include "open_spiel/games/bd_mines.h"

#include <unordered_map>
#include <algorithm>    // std::find, min
#include <sys/types.h>

#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace bd_mines {

namespace {

// Property bit flags
enum ElementProperties {
  kNone = 0,
  kConsumable = 1 << 0,
  kCanExplode = 1 << 1,
  kRounded = 1 << 2,
};

// All possible elements
const Element kElRockford = {
  HiddenCellType::kRockford, VisibleCellType::kRockford,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, '@'
};
const Element kElRockfordInExit = {
  HiddenCellType::kRockfordInExit, VisibleCellType::kRockfordInExit,
  ElementProperties::kNone, '!'
};
const Element kElExitOpen = {
  HiddenCellType::kExitOpen, VisibleCellType::kExitOpen,
  ElementProperties::kNone, '#'
};
const Element kElExitClosed = {
  HiddenCellType::kExitClosed, VisibleCellType::kExitClosed,
  ElementProperties::kNone, 'C'
};
const Element kElEmpty = {
  HiddenCellType::kEmpty, VisibleCellType::kEmpty,
  ElementProperties::kConsumable, ' '
};
const Element kElDirt = {
  HiddenCellType::kDirt, VisibleCellType::kDirt,
  ElementProperties::kConsumable, '.'
};
const Element kElBoulder = {
  HiddenCellType::kBoulder, VisibleCellType::kBoulder,
  ElementProperties::kConsumable | ElementProperties::kRounded, 'o'
};
const Element kElBoulderFalling = {
  HiddenCellType::kBoulderFalling, VisibleCellType::kBoulder,
  ElementProperties::kConsumable, 'o'
};
const Element kElDiamond = {
  HiddenCellType::KDiamond, VisibleCellType::KDiamond,
  ElementProperties::kConsumable | ElementProperties::kRounded, '*'
};
const Element kElDiamondFalling = {
  HiddenCellType::kDiamondFalling, VisibleCellType::KDiamond,
  ElementProperties::kConsumable, '*'
};
const Element kElFireflyUp = {
  HiddenCellType::kFireflyUp, VisibleCellType::kFirefly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'
};
const Element kElFireflyLeft = {
  HiddenCellType::kFireflyLeft, VisibleCellType::kFirefly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'
};
const Element kElFireflyDown = {
  HiddenCellType::kFireflyDown, VisibleCellType::kFirefly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'
};
const Element kElFireflyRight = {
  HiddenCellType::kFireflyRight, VisibleCellType::kFirefly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'F'
};
const Element kElButterflyUp = {
  HiddenCellType::kButterflyUp, VisibleCellType::kButterfly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'
};
const Element kElButterflyLeft = {
  HiddenCellType::kButterflyLeft, VisibleCellType::kButterfly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'
};
const Element kElButterflyDown = {
  HiddenCellType::kButterflyDown, VisibleCellType::kButterfly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'
};
const Element kElButterflyRight = {
  HiddenCellType::kButterflyRight, VisibleCellType::kButterfly,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'U'
};
const Element kElAmoeba = {
  HiddenCellType::kAmoeba, VisibleCellType::kAmoeba,
  ElementProperties::kConsumable, 'A'
};
const Element kElWallBrick = {
  HiddenCellType::kWallBrick, VisibleCellType::kWallBrick,
  ElementProperties::kConsumable | ElementProperties::kRounded, 'H'
};
const Element kElWallSteel = {
  HiddenCellType::kWallSteel, VisibleCellType::kWallSteel,
  ElementProperties::kNone, 'S'
};
const Element kElWallMagicOn = {
  HiddenCellType::kWallMagicOn, VisibleCellType::kWallMagicOn,
  ElementProperties::kConsumable, 'M'
};
const Element kElWallMagicDormant = {
  HiddenCellType::kWallMagicDormant, VisibleCellType::kWallMagicOff,
  ElementProperties::kConsumable, 'Q'
};
const Element kElWallMagicExpired = {
  HiddenCellType::kWallMagicExpired, VisibleCellType::kWallMagicOff,
  ElementProperties::kConsumable, 'Q'
};
const Element kElExplosionDiamond = {
  HiddenCellType::kExplosionDiamond, VisibleCellType::kExplosion,
  ElementProperties::kNone, 'E'
};
const Element kElExplosionBoulder = {
  HiddenCellType::kExplosionBoulder, VisibleCellType::kExplosion,
  ElementProperties::kNone, 'E'
};
const Element kElExplosionEmpty = {
  HiddenCellType::kExplosionEmpty, VisibleCellType::kExplosion,
  ElementProperties::kNone, 'E'
};
const Element kElGateRedClosed = {
  HiddenCellType::kGateRedClosed, VisibleCellType::kGateRedClosed,
  ElementProperties::kNone, 'r'
};
const Element kElGateRedOpen = {
  HiddenCellType::kGateRedOpen, VisibleCellType::kGateRedOpen,
  ElementProperties::kNone, 'R'
};
const Element kElKeyRed = {
  HiddenCellType::kKeyRed, VisibleCellType::kKeyRed,
  ElementProperties::kNone, '1'
};
const Element kElGateBlueClosed = {
  HiddenCellType::kGateBlueClosed, VisibleCellType::kGateBlueClosed,
  ElementProperties::kNone, 'b'
};
const Element kElGateBlueOpen = {
  HiddenCellType::kGateBlueOpen, VisibleCellType::kGateBlueOpen,
  ElementProperties::kNone, 'B'
};
const Element kElKeyBlue = {
  HiddenCellType::kKeyBlue, VisibleCellType::kKeyBlue,
  ElementProperties::kNone, '2'
};
const Element kElGateGreenClosed = {
  HiddenCellType::kGateGreenClosed, VisibleCellType::kGateGreenClosed,
  ElementProperties::kNone, 'g'
};
const Element kElGateGreenOpen = {
  HiddenCellType::kGateGreenOpen, VisibleCellType::kGateGreenOpen,
  ElementProperties::kNone, 'G'
};
const Element kElKeyGreen = {
  HiddenCellType::kKeyGreen, VisibleCellType::kKeyGreen,
  ElementProperties::kNone, '3'
};
const Element kElGateYellowClosed = {
  HiddenCellType::kGateYellowClosed, VisibleCellType::kGateYellowClosed,
  ElementProperties::kNone, 'y'
};
const Element kElGateYellowOpen = {
  HiddenCellType::kGateYellowOpen, VisibleCellType::kGateYellowOpen,
  ElementProperties::kNone, 'Y'
};
const Element kElKeyYellow = {
  HiddenCellType::kKeyYellow, VisibleCellType::kKeyYellow,
  ElementProperties::kNone, '4'
};
const Element kElNut = {
  HiddenCellType::kNut, VisibleCellType::kNut,
  ElementProperties::kRounded | ElementProperties::kConsumable, '+'
};
const Element kElNutFalling = {
  HiddenCellType::kNutFalling, VisibleCellType::kNut,
  ElementProperties::kRounded | ElementProperties::kConsumable, '+'
};
const Element kElBomb = {
  HiddenCellType::kBomb, VisibleCellType::kBomb,
  ElementProperties::kRounded | ElementProperties::kConsumable | ElementProperties::kCanExplode, '^'
};
const Element kElBombFalling = {
  HiddenCellType::kBombFalling, VisibleCellType::kBomb,
  ElementProperties::kRounded | ElementProperties::kConsumable | ElementProperties::kCanExplode, '^'
};

// Hash for Element, so we can use as a map key
struct ElementHash {
  std::size_t operator()(const Element& e) const {
    return static_cast<int>(e.cell_type) - static_cast<int>(HiddenCellType::kNull);
  }
};

// ----- Conversion maps -----
// Swap map for DeserializeState
const std::unordered_map<int, Element> kCellTypeToElement {
  {static_cast<int>(HiddenCellType::kNull), kNullElement}, 
  {static_cast<int>(HiddenCellType::kRockford), kElRockford}, 
  {static_cast<int>(HiddenCellType::kEmpty), kElEmpty}, 
  {static_cast<int>(HiddenCellType::kDirt), kElDirt}, 
  {static_cast<int>(HiddenCellType::kBoulder), kElBoulder}, 
  {static_cast<int>(HiddenCellType::kBoulderFalling), kElBoulderFalling}, 
  {static_cast<int>(HiddenCellType::KDiamond), kElDiamond}, 
  {static_cast<int>(HiddenCellType::kDiamondFalling), kElDiamondFalling}, 
  {static_cast<int>(HiddenCellType::kExitClosed), kElExitClosed}, 
  {static_cast<int>(HiddenCellType::kExitOpen), kElExitOpen}, 
  {static_cast<int>(HiddenCellType::kRockfordInExit), kElRockfordInExit}, 
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
  {static_cast<int>(HiddenCellType::kAmoeba), kElAmoeba}, 
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
};

// Rotate actions right
const std::unordered_map<int, int> kRotateRight {
  {Directions::kUp, Directions::kRight}, {Directions::kRight, Directions::kDown}, 
  {Directions::kDown, Directions::kLeft}, {Directions::kLeft, Directions::kUp},
  {Directions::kNone, Directions::kNone}, 
};

// Rotate actions left
const std::unordered_map<int, int> kRotateLeft {
  {Directions::kUp, Directions::kLeft}, {Directions::kLeft, Directions::kDown}, 
  {Directions::kDown, Directions::kRight}, {Directions::kRight, Directions::kUp},
  {Directions::kNone, Directions::kNone}, 
};

// actions to strings
const std::unordered_map<int, std::string> kActionsToString {
  {Directions::kUp, "up"}, {Directions::kLeft, "left"}, {Directions::kDown, "down"}, 
  {Directions::kRight, "right"}, {Directions::kNone, "none"},
};

// directions to offsets (col, row) 
const std::unordered_map<int, std::pair<int, int>> kDirectionOffsets {
  {Directions::kUp, {0, -1}}, {Directions::kUpLeft, {-1, -1}}, {Directions::kLeft, {-1, 0}}, 
  {Directions::kDownLeft, {-1, 1}}, {Directions::kDown, {0, 1}}, {Directions::kDownRight, {1, 1}}, 
  {Directions::kRight, {1, 0}}, {Directions::kUpRight, {1, -1}}, {Directions::kNone, {0, 0}},
};

// Directions to fireflys
const std::unordered_map<int, Element> kDirectionToFirefly {
  {Directions::kUp, kElFireflyUp}, {Directions::kLeft, kElFireflyLeft}, 
  {Directions::kDown, kElFireflyDown}, {Directions::kRight, kElFireflyRight},
};

// Firefly to directions
const std::unordered_map<Element, int, ElementHash> kFireflyToDirection {
  {kElFireflyUp, Directions::kUp}, {kElFireflyLeft, Directions::kLeft}, 
  {kElFireflyDown, Directions::kDown}, {kElFireflyRight, Directions::kRight},
};

// Butterfly to directions
const std::unordered_map<int, Element> kDirectionToButterfly {
  {Directions::kUp, kElButterflyUp}, {Directions::kLeft, kElButterflyLeft}, 
  {Directions::kDown, kElButterflyDown}, {Directions::kRight, kElButterflyRight},
};

// Directions to butterflys
const std::unordered_map<Element, int, ElementHash> kButterflyToDirection {
  {kElButterflyUp, Directions::kUp}, {kElButterflyLeft, Directions::kLeft}, 
  {kElButterflyDown, Directions::kDown}, {kElButterflyRight, Directions::kRight},
};

// Element explosion maps
const std::unordered_map<Element, Element, ElementHash> kElementToExplosion {
  {kElFireflyUp, kElExplosionEmpty}, {kElFireflyLeft, kElExplosionEmpty}, 
  {kElFireflyDown, kElExplosionEmpty}, {kElFireflyRight, kElExplosionEmpty},
  {kElButterflyUp, kElExplosionDiamond}, {kElButterflyLeft, kElExplosionDiamond}, 
  {kElButterflyDown, kElExplosionDiamond}, {kElButterflyRight, kElExplosionDiamond},
  {kElRockford, kElExplosionEmpty}, {kElBomb, kElExplosionEmpty},
  {kElBombFalling, kElExplosionEmpty}
};

// Explosions back to elements
const std::unordered_map<Element, Element, ElementHash> kExplosionToElement {
  {kElExplosionDiamond, kElDiamond}, {kElExplosionBoulder, kElBoulder},
  {kElExplosionEmpty, kElEmpty},
};

// Magic wall conversion map
const std::unordered_map<Element, Element, ElementHash> kMagicWallConversion {
  {kElBoulderFalling, kElDiamondFalling}, {kElDiamondFalling, kElBoulderFalling},
};

// Gem point maps
const std::unordered_map<Element, int, ElementHash> kGemPoints {
  {kElDiamond, 10}, {kElDiamondFalling, 10},
};

// Gate open conversion map
const std::unordered_map<Element, Element, ElementHash> kGateOpenMap {
  {kElGateRedClosed, kElGateRedOpen}, {kElGateBlueClosed, kElGateBlueOpen},
  {kElGateGreenClosed, kElGateGreenOpen}, {kElGateYellowClosed, kElGateYellowOpen},
};
// Gate key map
const std::unordered_map<Element, Element, ElementHash> kKeyToGate {
  {kElKeyRed, kElGateRedClosed}, {kElKeyBlue, kElGateBlueClosed},
  {kElKeyGreen, kElGateGreenClosed}, {kElKeyYellow, kElGateYellowClosed},
};

// Stationary to falling
const std::unordered_map<Element, Element, ElementHash> kElToFalling {
  {kElDiamond, kElDiamondFalling}, {kElBoulder, kElBoulderFalling},
  {kElNut, kElNutFalling}, {kElBomb, kElBombFalling},
};

// Default parameters.
constexpr int kDefaultMagicWallSteps = 140; // Number of steps before magic walls expire
constexpr int kDefaultAmoebaChance = 20;    // Chance to spawn another amoeba (out of 256)
constexpr double kDefaultAmoebaMaxPercentage = 0.16;   // Maximum number of amoeba before they collapse (percentage of map size)

// Facts about the game
const GameType kGameType{
    /*short_name=*/"bd_mines",
    /*long_name=*/"Boulder Dash Mines",
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
    {
        {"magic_wall_steps", GameParameter(kDefaultMagicWallSteps)},
        {"amoeba_chance", GameParameter(kDefaultAmoebaChance)},
        {"amoeba_max_percentage", GameParameter(kDefaultAmoebaMaxPercentage)},
        {"rng_seed", GameParameter(0)},
        {"grid", GameParameter(std::string(kDefaultGrid))}
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BDMinesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string BDMinesState::ActionToString(Player player, Action move_id) const {
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

bool BDMinesState::IsTerminal() const {
  // Time complete or rockford exploded
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElRockford);
  return steps_remaining_ <= 0 || it == grid_.elements.end();
}

std::vector<double> BDMinesState::Returns() const {
  // Sum of rewards, and should agree with Rewards()
  return std::vector<double>{(double)sum_reward_};
}

std::vector<double> BDMinesState::Rewards() const {
  // reward for most recent state transition
  return std::vector<double>{(double)current_reward_};
}

std::string BDMinesState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsChanceNode()) {
    return "ChanceNode -- no observation";
  }
  return ToString();
}

void BDMinesState::ObservationTensor(Player player,
                                 std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 3-d tensor.
  TensorView<3> view(values, {kNumVisibleCellType, grid_.num_rows, grid_.num_cols}, true);

  // No observations at chance nodes.
  if (IsChanceNode()) {
    std::fill(values->begin(), values->end(), 0);
    return;
  }

  int i = 0;
  for (int row = 0; row < grid_.num_rows; ++row) {
    for (int col = 0; col < grid_.num_cols; ++col) {
      int channel = static_cast<int>(grid_.elements[i].visible_type);
      view[{channel, row, col}] = 1.0;
      ++i;
    }
  }
}

int BDMinesState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

// element helper functions
namespace {

bool IsActionHorz(int action) {
  return action == Directions::kLeft || action == Directions::kRight;
}

bool IsActionVert(int action) {
  return action == Directions::kUp || action == Directions::kDown;
}

bool IsFirefly(const Element &element) {
  return element == kElFireflyUp || element == kElFireflyLeft ||
    element == kElFireflyDown || element == kElFireflyRight;
}

bool IsButterfly(const Element &element) {
  return element == kElButterflyUp || element == kElButterflyLeft ||
    element == kElButterflyDown || element == kElButterflyRight;
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

} // namespace


// ---------- Game dynamic function ----------

// Given an index and action, get the new flat index
int BDMinesState::IndexFromAction(int index, int action) const {
  int col = index % grid_.num_cols;
  int row = (index - col) / grid_.num_cols;
  std::pair<int, int> offsets = kDirectionOffsets.at(action);
  col += offsets.first;
  row += offsets.second;
  return (grid_.num_cols * row) + col;
}

// Check if the index with a given action step will remain in bounds
bool BDMinesState::InBounds(int index, int action) const {
  int col = index % grid_.num_cols;
  int row = (index - col) / grid_.num_cols;
  std::pair<int, int> offsets = kDirectionOffsets.at(action);
  col += offsets.first;
  row += offsets.second;
  return col >= 0 && col < grid_.num_cols && row >= 0 && row < grid_.num_rows;
}

// Check if the index after applying action contains the given element
bool BDMinesState::IsType(int index, Element element, int action) const {
  int new_index = IndexFromAction(index, action);
  return InBounds(index, action) && grid_.elements[new_index] == element;
}

// Check if the index after applying action has an element with the given property
bool BDMinesState::HasProperty(int index, int property, int action) const {
  int new_index = IndexFromAction(index, action);
  return InBounds(index, action) && ((grid_.elements[new_index].properties & property) > 0);
}

// Move the element given the action, and set the old index to empty
void BDMinesState::MoveItem(int index, int action) {
  int new_index = IndexFromAction(index, action);
  grid_.elements[new_index] = grid_.elements[index];
  grid_.elements[new_index].has_updated = true;
  grid_.elements[index] = kElEmpty;
}

// Set the new index to the given element
void BDMinesState::SetItem(int index, Element element, int action) {
  int new_index = IndexFromAction(index, action);
  grid_.elements[new_index] = element;
  grid_.elements[new_index].has_updated = true;
}

// Get the item after applying the action to the index
Element BDMinesState::GetItem(int index, int action) const {
  return grid_.elements[IndexFromAction(index, action)];
}

// Check if the element is adjacent to and cell around the given index
bool BDMinesState::IsTypeAdjacent(int index, Element element) const {
  return IsType(index, element, Directions::kUp) || IsType(index, element, Directions::kLeft) ||
         IsType(index, element, Directions::kDown) || IsType(index, element, Directions::kRight);
}

// Can roll left if sitting on rounded element, left and bottom left clear
bool BDMinesState::CanRollLeft(int index) const {
  return HasProperty(index, ElementProperties::kRounded, Directions::kDown) &&
         IsType(index, kElEmpty, Directions::kLeft) && IsType(index, kElEmpty, Directions::kDownLeft);
}

// Can roll right if sitting on rounded element, right and bottom right clear
bool BDMinesState::CanRollRight(int index) const {
  return HasProperty(index, ElementProperties::kRounded, Directions::kDown) &&
         IsType(index, kElEmpty, Directions::kRight) && IsType(index, kElEmpty, Directions::kDownRight);
}

// Roll the item to the left
void BDMinesState::RollLeft(int index, Element element) {
  SetItem(index, element);
  MoveItem(index, Directions::kLeft);
}

// Roll the item to the right
void BDMinesState::RollRight(int index, Element element) {
  SetItem(index, element);
  MoveItem(index, Directions::kRight);
}

// Push the item
void BDMinesState::Push(int index, Element stationary, Element falling, int action) {
  int new_index = IndexFromAction(index, action);
  // Check if same direction past element is empty so that theres room to push
  if (IsType(new_index, kElEmpty, action)) {
    SetItem(index, kElEmpty);
    SetItem(new_index, kElRockford);
    // Check if the element will become stationary or falling
    int next_index = IndexFromAction(new_index, action);
    bool is_empty = IsType(next_index, kElEmpty, Directions::kDown);
    SetItem(new_index, is_empty ? falling : stationary, action);
  }
}

// Move the item through the magic wall
void BDMinesState::MoveThroughMagic(int index, Element element) {
  // Check if magic wall is still active
  if (magic_wall_steps_ <= 0) {return;}
  magic_active_ = true;
  int index_below = IndexFromAction(index, Directions::kDown);
  // Ned to ensure cell below magic wall is empty (so item can pass through)
  if (IsType(index_below, kElEmpty, Directions::kDown)) {
    SetItem(index, kElEmpty);
    SetItem(index_below, element, Directions::kDown);
  }
}

// Explode the item
void BDMinesState::Explode(int index, Element element, int action) {
  int new_index = IndexFromAction(index, action);
  auto it = kElementToExplosion.find(GetItem(new_index));
  Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
  SetItem(new_index, element);
  // Recursively check all directions for chain explosions
  for (int dir = 0; dir < kNumDirections; ++dir) {
    if (dir == Directions::kNone || !InBounds(new_index, dir)) {continue;}
    if (HasProperty(new_index, ElementProperties::kCanExplode, dir)) {
      Explode(new_index, ex, dir);
    } else if (HasProperty(new_index, ElementProperties::kConsumable, dir)) {
      SetItem(new_index, ex, dir);
    }
  }
}

void BDMinesState::OpenGate(Element element) {
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), element);
  if (it != grid_.elements.end()) {
    int index = std::distance(grid_.elements.begin(), it);
    SetItem(index, kGateOpenMap.at(GetItem(index)));
  }
}

void BDMinesState::UpdateBoulder(int index) {
  // Boulder falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElBoulderFalling);
    UpdateBoulderFalling(index);
  } else if (CanRollLeft(index)) {    // Roll left/right if possible
    RollLeft(index, kElBoulderFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBoulderFalling);
  }
}

void BDMinesState::UpdateBoulderFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (HasProperty(index, ElementProperties::kCanExplode, Directions::kDown)) {
    // Falling boulders can cause elements to explode
    auto it = kElementToExplosion.find(GetItem(index, Directions::kDown));
    Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex, Directions::kDown);
  } else if (IsType(index, kElWallMagicOn, Directions::kDown) || 
             IsType(index, kElWallMagicDormant, Directions::kDown)) {
    MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
  } else if (IsType(index, kElNut, Directions::kDown)) {
    // Falling on a nut, crack it open to reveal a diamond!
    SetItem(index, kElDiamond, Directions::kDown);
  } else if (IsType(index, kElNut, Directions::kDown)) {
    // Falling on a bomb, explode!
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElBoulderFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBoulderFalling);
  } else {
    // Default options is for falling boulder to become stationary
    SetItem(index, kElBoulder);
  }
}

void BDMinesState::UpdateDiamond(int index) {
  // Diamond falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElDiamondFalling);
    UpdateDiamondFalling(index);
  } else if (CanRollLeft(index)) {    // Roll left/right if possible
    RollLeft(index, kElDiamondFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElDiamondFalling);
  }
}

void BDMinesState::UpdateDiamondFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (HasProperty(index, ElementProperties::kCanExplode, Directions::kDown) &&
             !IsType(index, kElBomb, Directions::kDown) &&
             !IsType(index, kElBombFalling, Directions::kDown)) {
    // Falling diamonds can cause elements to explode (but not bombs)
    auto it = kElementToExplosion.find(GetItem(index, Directions::kDown));
    Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex, Directions::kDown);
  } else if (IsType(index, kElWallMagicOn, Directions::kDown) || 
             IsType(index, kElWallMagicDormant, Directions::kDown)) {
    MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElDiamondFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElDiamondFalling);
  } else {
    // Default options is for falling diamond to become stationary
    SetItem(index, kElDiamond);
  }
}

void BDMinesState::UpdateNut(int index) {
  // Nut falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElNutFalling);
    UpdateNutFalling(index);
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElNutFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElNutFalling);
  }
}

void BDMinesState::UpdateNutFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElNutFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElNutFalling);
  } else {
    // Default options is for falling nut to become stationary
    SetItem(index, kElNut);
  }
}

void BDMinesState::UpdateBomb(int index) {
  // Nut falls if empty below
  if (IsType(index, kElEmpty, Directions::kDown)) {
    SetItem(index, kElBombFalling);
    UpdateBombFalling(index);
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElBomb);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBomb);
  }
}

void BDMinesState::UpdateBombFalling(int index) {
  // Continue to fall as normal
  if (IsType(index, kElEmpty, Directions::kDown)) {
    MoveItem(index, Directions::kDown);
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElBombFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBombFalling);
  } else {
    // Default options is for bomb to explode if stopped falling
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  }
}

void BDMinesState::UpdateExit(int index) {
  // Open exit if enough gems collected
  if (gems_collected_ >= gems_required_) {
    SetItem(index, kElExitOpen);
  }
}

void BDMinesState::UpdateRockford(int index, int action) {
  if (IsType(index, kElEmpty, action) || IsType(index, kElDirt, action)) {
    // Move if empty/dirt
    MoveItem(index, action);
  } else if (IsType(index, kElDiamond, action) || IsType(index, kElDiamondFalling, action)) {
    // Collect gems
    ++gems_collected_;
    current_reward_ += kGemPoints.at(GetItem(index, action));
    sum_reward_ += kGemPoints.at(GetItem(index, action));
    MoveItem(index, action);
  } else if (IsActionHorz(action) && (IsType(index, kElBoulder, action) ||
                                    IsType(index, kElNut, action) ||
                                    IsType(index, kElBomb, action))) {
    // Push boulder, nut, or bomb if action is horizontal
    Push(index, GetItem(index, action), kElToFalling.at(GetItem(index, action)), action);                                 
  } else if (IsKey(GetItem(index, action))) {
    // Collecting key, set gate open
    OpenGate(kKeyToGate.at(GetItem(index, action)));
    MoveItem(index, action);
  } else if (IsOpenGate(GetItem(index, action))) {
    // Walking through an open gate
    int index_gate = IndexFromAction(index, action);
      if (IsType(index_gate, kElEmpty, action)) {
      SetItem(index, kElEmpty);
      SetItem(index_gate, kElRockford, action);
    }
  } else if (IsType(index, kElExitOpen, action)) {
    // Walking into exit after collecting enough gems
    MoveItem(index, action);
    SetItem(index, kElRockfordInExit, action);
    current_reward_ += steps_remaining_;
    sum_reward_ += steps_remaining_;
  }
}

void BDMinesState::UpdateFirefly(int index, int action) {
  int new_dir = kRotateLeft.at(action);
  if (IsTypeAdjacent(index, kElRockford) || IsTypeAdjacent(index, kElAmoeba)) {
    // Explode if touching rockford/amoeba
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else if (IsType(index, kElEmpty, new_dir)) {
    // Fireflies always try to rotate left, otherwise continue forward
    SetItem(index, kDirectionToFirefly.at(new_dir));
    MoveItem(index, new_dir);
  } else if (IsType(index, kElEmpty, action)) {
    SetItem(index, kDirectionToFirefly.at(action));
    MoveItem(index, action);
  } else {
    // No other options, rotate right
    SetItem(index, kDirectionToFirefly.at(kRotateRight.at(action)));
  }
}

void BDMinesState::UpdateButterfly(int index, int action) {
  int new_dir = kRotateRight.at(action);
  if (IsTypeAdjacent(index, kElRockford) || IsTypeAdjacent(index, kElAmoeba)) {
    // Explode if touching rockford/amoeba
    auto it = kElementToExplosion.find(GetItem(index));
    Element ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    Explode(index, ex);
  } else if (IsType(index, kElEmpty, new_dir)) {
    // Butterflies always try to rotate right, otherwise continue forward
    SetItem(index, kDirectionToButterfly.at(new_dir));
    MoveItem(index, new_dir);
  } else if (IsType(index, kElEmpty, action)) {
    SetItem(index, kDirectionToButterfly.at(action));
    MoveItem(index, action);
  } else {
    // No other options, rotate right
    SetItem(index, kDirectionToButterfly.at(kRotateLeft.at(action)));
  }
}

void BDMinesState::UpdateMagicWall(int index) {
  // Dorminant, active, then expired once time runs out
  if (magic_active_) {
    SetItem(index, kElWallMagicOn);
  } else if (magic_wall_steps_ > 0) {
    SetItem(index, kElWallMagicDormant);
  } else {
    SetItem(index, kElWallMagicExpired);
  }
}

void BDMinesState::UpdateAmoeba(int index) {
  // Replace amoebas if swap element set
  if (amoeba_swap_ != kNullElement) {
    SetItem(index, amoeba_swap_);
    return;
  }
  ++amoeba_size_;
  // Check if at least one tile amoeba can grow to
  if (IsTypeAdjacent(index, kElEmpty) || IsTypeAdjacent(index, kElDirt)) {
    amoeba_enclosed_ = false;
  }
  // Roll if to grow and direction
  bool will_grow = (rng_() % 256) < amoeba_chance_;
  int grow_dir = rng_() % kNumActions;
  if (will_grow && (IsType(index, kElEmpty, grow_dir) || IsType(index, kElDirt, grow_dir))) {
    SetItem(index, kElAmoeba, grow_dir);
  }
}

void BDMinesState::UpdateExplosions(int index) {
  SetItem(index, kExplosionToElement.at(GetItem(index)));
}

void BDMinesState::StartScan() {
  // Update global flags
  --steps_remaining_;
  current_reward_ = 0;
  amoeba_size_ = 0;
  amoeba_enclosed_ = true;
  // Reset element flags
  for (auto & e : grid_.elements) {
    e.has_updated = false;
  }
}

void BDMinesState::EndScan() {
  // Check if amoeba dead/closed/size
  if (amoeba_swap_ == kNullElement) {
    if (amoeba_enclosed_) {
      // amoebas become diamonds if enclosed
      amoeba_swap_ = kElDiamond;
    } else if (amoeba_size_ > amoeba_max_size_) {
      // amoebas become boulders is they grow too large
      amoeba_swap_ = kElBoulder;
    }
  }
  // Reduce magic wall steps if active
  if (magic_active_) {
    magic_wall_steps_ = std::max(magic_wall_steps_ - 1, 0);
  }
  // Check if still active
  magic_active_ = (magic_active_ && magic_wall_steps_ > 0);
}

void BDMinesState::DoApplyAction(Action move) {
  if (cur_player_ == kChancePlayerId) {
    // Check each cell and apply respective dynamics function
    for (int index = 0; index < grid_.num_cols *  grid_.num_rows; ++index) {
      Element &e = grid_.elements[index];
      if (e.has_updated) {
        continue;
      } else if (e == kElBoulder) {
        UpdateBoulder(index);
      } else if (e == kElBoulderFalling) {
        UpdateBoulderFalling(index);
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
      } else if (IsMagicWall(e)) {
        UpdateMagicWall(index);
      } else if (e == kElAmoeba) {
        UpdateAmoeba(index);
      } else if (IsExplosion(e)) {
        UpdateExplosions(index);
      }
    }
    EndScan();
    cur_player_ = 0;
  } else {
    StartScan();
    // Find where rockford is, and update his position
    auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElRockford);
    int index = std::distance(grid_.elements.begin(), it);
    UpdateRockford(index, move);
    cur_player_ = kChancePlayerId;
  }
}

std::vector<Action> BDMinesState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();;
  } else if (IsTerminal()) {
    return {};
  } else {
    return {Directions::kNone, Directions::kUp, Directions::kRight, Directions::kDown, Directions::kLeft};
  }
}

std::vector<std::pair<Action, double>> BDMinesState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes = {std::make_pair(0, 1.0)};
  return outcomes;
}

std::string BDMinesState::ToString() const {
  if (IsChanceNode()) {
    return "chance node";
  }
  std::string out_str;
  int col_counter = 0;
  for (const auto el : grid_.elements) {
    ++col_counter;
    out_str += el.id;
    if (col_counter == grid_.num_cols) {
      out_str += "\n";
      col_counter = 0;
    }
  }
  out_str += "time left: " + std::to_string(steps_remaining_) + ", ";
  out_str += "gems required: " + std::to_string(gems_required_)  + ", ";
  out_str += "gems collectred: " + std::to_string(gems_collected_);
  return out_str;
}

std::string BDMinesState::Serialize() const {
  std::string out_str;
  // grid properties
  absl::StrAppend(&out_str, grid_.num_cols, ",");
  absl::StrAppend(&out_str, grid_.num_rows, ",");
  absl::StrAppend(&out_str, steps_remaining_, ",");
  absl::StrAppend(&out_str, magic_wall_steps_, ",");
  absl::StrAppend(&out_str, magic_active_, ",");
  absl::StrAppend(&out_str, amoeba_max_size_, ",");
  absl::StrAppend(&out_str, amoeba_size_, ",");
  absl::StrAppend(&out_str, amoeba_chance_, ",");
  absl::StrAppend(&out_str, std::to_string(static_cast<int>(amoeba_swap_.cell_type)) + ",");
  absl::StrAppend(&out_str, amoeba_enclosed_, ",");
  absl::StrAppend(&out_str, gems_required_, ",");
  absl::StrAppend(&out_str, gems_collected_, ",");
  absl::StrAppend(&out_str, current_reward_, ",");
  absl::StrAppend(&out_str, sum_reward_, ",");
  absl::StrAppend(&out_str, cur_player_, "\n");
  // grid contents
  int col_counter = 0;
  for (const auto el : grid_.elements) {
    ++col_counter;
    absl::StrAppend(&out_str, static_cast<int>(el.cell_type), ",");
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

std::unique_ptr<State> BDMinesState::Clone() const {
  return std::unique_ptr<State>(new BDMinesState(*this));
}

// ------ game -------

std::unique_ptr<State> BDMinesGame::DeserializeState(const std::string& str) const {
  // empty string
  if (str.empty()) {return NewInitialState();}
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  if (lines.size() < 2) {
    SpielFatalError("Empty map string passed.");
  }
  // Read grid properties
  std::vector<std::string> property_line = absl::StrSplit(lines[0], ',');
  Grid grid;
  int steps_remaining, magic_wall_steps, amoeba_max_size, amoeba_size, amoeba_chance,
      gems_required, gems_collected, current_reward, sum_reward, cur_player;
  bool magic_active, amoeba_enclosed;
  Element amoeba_swap;
  try {
    grid.num_cols = std::stoi(property_line[0]);
    grid.num_rows = std::stoi(property_line[1]);
    steps_remaining = std::stoi(property_line[2]);
    magic_wall_steps = std::stoi(property_line[3]);
    magic_active = std::stoi(property_line[4]);
    amoeba_max_size = std::stoi(property_line[5]);
    amoeba_size = std::stoi(property_line[6]);
    amoeba_chance = std::stoi(property_line[7]);
    amoeba_swap = kCellTypeToElement.at(std::stoi(property_line[8]));
    amoeba_enclosed = std::stoi(property_line[9]);
    gems_required = std::stoi(property_line[10]);
    gems_collected = std::stoi(property_line[11]);
    current_reward = std::stoi(property_line[12]);
    sum_reward = std::stoi(property_line[13]);
    cur_player = std::stoi(property_line[14]);
  } catch (...) {
    SpielFatalError("Invalid grid properties2 ");
  }
  // Set grid elements
  for (std::size_t i = 1; i < lines.size(); ++i) {
    std::vector<std::string> grid_line = absl::StrSplit(lines[i], ',');
    // Check for proper number of columns
    if (grid_line.size() != grid.num_cols) {
      SpielFatalError("Grid line " + std::to_string(i-1) + "doesn't have correct number of elements.");
    }
    // Check each element in row
    for (const auto & type : grid_line) {
      auto it = kCellTypeToElement.find(std::stoi(type));
      if (it != kCellTypeToElement.end()) {
        grid.elements.push_back(it->second);
      } else {
        SpielFatalError("Unknown element id: " + type);
      }
    }
  }
  // Ensure we read proper number of rows
  if (lines.size() - 1 != grid.num_rows) {
    SpielFatalError("Incorrect number of rows, got " + std::to_string(lines.size() - 1) + 
                    " but need " + std::to_string(grid.num_rows));
  }
  // Ensure rockford exists in the map
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElRockford);
  if (it == grid_.elements.end()) {
    SpielFatalError("Grid string doesn't contain agent rockford.");
  }

  return std::unique_ptr<State>(
      new BDMinesState(shared_from_this(), steps_remaining, magic_wall_steps,
               magic_active, amoeba_max_size, amoeba_size, amoeba_chance, amoeba_swap,
               amoeba_enclosed, gems_required, gems_collected, current_reward,
               sum_reward, grid, ++rng_seed_, cur_player));
}

int BDMinesGame::NumDistinctActions() const { 
  return kNumActions;
}

// There is arbitrarily chosen number to ensure the game is finite.
int BDMinesGame::MaxGameLength() const {
  return max_steps_;
}

int BDMinesGame::NumPlayers() const {
  return 1; 
}

double BDMinesGame::MinUtility() const {
  return 0; 
}

double BDMinesGame::MaxUtility() const {
  return max_steps_ + 500;
}

std::vector<int> BDMinesGame::ObservationTensorShape() const {
  return {kNumVisibleCellType, grid_.num_rows, grid_.num_cols};
}

Grid BDMinesGame::ParseGrid(const std::string& grid_string, double amoeba_max_percentage) {
  Grid grid;
  std::vector<std::string> lines = absl::StrSplit(grid_string, '\n');
  if (lines.size() < 2) {
    SpielFatalError("Empty map string passed.");
  }
  // Parse first line which contains level properties
  std::vector<std::string> property_line = absl::StrSplit(lines[0], ',');
  try {
    grid.num_cols = std::stoi(property_line[0]);
    grid.num_rows = std::stoi(property_line[1]);
    max_steps_ = std::stoi(property_line[2]);
    gems_required_ = std::stoi(property_line[3]);
  } catch (...) {
    SpielFatalError("Missing width, height, maximum steps, and/or gems required on first line");
  }
  // Parse grid contents
  for (std::size_t i = 1; i < lines.size(); ++i) {
    // Check for proper number of columns
    std::vector<std::string> grid_line = absl::StrSplit(lines[i], ',');
    if (grid_line.size() != grid.num_cols) {
      SpielFatalError("Grid line " + std::to_string(i-1) + " doesn't have correct number of elements." +
        " Received " + std::to_string(grid_line.size()) + ", expected " + std::to_string(grid.num_cols));
    }
    // Check each element in row
    for (const auto & type : grid_line) {
      auto it = kCellTypeToElement.find(std::stoi(type));
      if (it != kCellTypeToElement.end()) {
        grid.elements.push_back(it->second);
      } else {
        SpielFatalError("Unknown element id: " + type);
      }
    }
  }
  // Ensure we read proper number of rows
  if (lines.size() - 1 != grid.num_rows) {
    SpielFatalError("Incorrect number of rows, received " + std::to_string(lines.size() - 1) + 
                    ", expected " + std::to_string(grid.num_rows));
  }
  // Ensure rockford exists in the map
  auto it = std::find(grid_.elements.begin(), grid_.elements.end(), kElRockford);
  if (it == grid_.elements.end()) {
    SpielFatalError("Grid string doesn't contain agent rockford.");
  }
  amoeba_max_size_ = (int)(grid_.num_cols * grid_.num_rows * amoeba_max_percentage);

  return grid;
}

BDMinesGame::BDMinesGame(const GameParameters& params)
    : Game(kGameType, params),
      magic_wall_steps_(ParameterValue<int>("magic_wall_steps")),
      amoeba_chance_(ParameterValue<int>("amoeba_chance")),
      rng_seed_(ParameterValue<int>("rng_seed")),
      grid_(ParseGrid(ParameterValue<std::string>("grid"), 
            ParameterValue<double>("amoeba_max_percentage"))) {}

}  // namespace bd_mines
}  // namespace open_spiel
