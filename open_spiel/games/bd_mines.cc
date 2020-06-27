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
#include <sstream>
#include <sys/types.h>

#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace bd_mines {

namespace {
// // Cell types supported from Boulderdash/Emerald Mines
// enum class HiddenCellType {
//   kRockford = 0,
//   kEmpty = 1,
//   kDirt = 2,
//   kBoulder = 3,
//   kBoulderFalling = 4,
//   KDiamond = 5,
//   kDiamondFalling = 6,
//   kExitClosed = 7,
//   kExitOpen = 8,
//   kRockfordInExit = 9,
//   kFireflyUp = 10,
//   kFireflyLeft = 11,
//   kFireflyDown = 12,
//   kFireflyRight = 13,
//   kButterflyUp = 14,
//   kButterflyLeft = 15,
//   kButterflyDown = 16,
//   kButterflyRight = 17,
//   kWallBrick = 18,
//   kWallSteel = 19,
//   kWallMagicDormant = 20,
//   kWallMagicOn = 21,
//   kWallMagicExpired = 22,
//   kAmoeba = 23,
//   kExplosionDiamond = 24,
//   kExplosionBoulder = 25,
//   kExplosionEmpty = 26
// };

// // Cell types which are observable
// enum class VisibleCellType {
//   kRockford = 0,
//   kEmpty = 1,
//   kDirt = 2,
//   kBoulder = 3,
//   KDiamond = 4,
//   kExitClosed = 5,
//   kExitOpen = 6,
//   kRockfordInExit = 7,
//   kFirefly = 8,
//   kButterfly = 9,
//   kWallBrick = 10,
//   kWallSteel = 11,
//   kWallMagicOff = 12,
//   kWallMagicOn = 13,
//   kAmoeba = 14,
//   kExplosion = 15,
// };

constexpr int kNumVisibleCellType = 16;

// Property bit flags
enum ElementProperties {
  kNone = 0,
  kConsumable = 1 << 0,
  kCanExplode = 1 << 1,
  kRounded = 1 << 2,
};

// All possible elements
const Element kNullElement = {HiddenCellType::kNull, VisibleCellType::kNull, -1, 0};

const Element kElRockford = {
  HiddenCellType::kRockford, VisibleCellType::kRockford,
  ElementProperties::kConsumable | ElementProperties::kCanExplode, 'R'
};
const Element kElRockfordInExit = {
  HiddenCellType::kRockfordInExit, VisibleCellType::kRockfordInExit,
  ElementProperties::kNone, '!'
};
const Element kElExitOpen = {
  HiddenCellType::kExitOpen, VisibleCellType::kExitOpen,
  ElementProperties::kNone, 'O'
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
  ElementProperties::kConsumable | ElementProperties::kRounded, 'D'
};
const Element kElDiamondFalling = {
  HiddenCellType::kDiamondFalling, VisibleCellType::KDiamond,
  ElementProperties::kConsumable, 'D'
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
  ElementProperties::kConsumable | ElementProperties::kRounded, 'B'
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

// // Directions the interactions take place
// enum Directions {
//   kNone = 0, kUp = 1, kRight = 2, kDown = 3, kLeft = 4, kUpRight = 5, 
//   kDownRight = 6, kDownLeft = 7, kUpLeft = 8
// };

// // Actions which the agent can perform
// // enum RockfordActions {
// //   kNone = Directions::kNone, kUp = Directions::kUp, kLeft = Directions::kLeft, 
// //   kDown = Directions::kDown, kRight = Directions::kRight
// // };

// constexpr int kNumDirections = 9;
// constexpr int kNumActions = 5;

struct ElementHash {
  std::size_t operator()(const Element& e) const {
    return static_cast<int>(e.cell_type) - static_cast<int>(HiddenCellType::kNull);
  }
};

// ----- Conversion maps -----
// Input chars to cell types
const std::unordered_map<char, Element> kCharElementMap {
  {'R', kElRockford},
  {' ', kElEmpty},
  {'.', kElDirt},
  {'o', kElBoulder},
  {'D', kElDiamond},
  {'C', kElExitClosed},
  {'F', kElFireflyLeft},
  {'U', kElButterflyDown},
  {'B', kElWallBrick},
  {'S', kElWallSteel},
  {'Q', kElWallMagicDormant},
  {'A', kElAmoeba},
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
  {kElFireflyUp, kElExplosionDiamond}, {kElFireflyLeft, kElExplosionDiamond}, 
  {kElFireflyDown, kElExplosionDiamond}, {kElFireflyRight, kElExplosionDiamond},
  {kElButterflyUp, kElExplosionEmpty}, {kElButterflyLeft, kElExplosionEmpty}, 
  {kElButterflyDown, kElExplosionEmpty}, {kElButterflyRight, kElExplosionEmpty},
  {kElRockford, kElExplosionEmpty},
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


// Default parameters.
constexpr int kDefaultMaxSteps = 1000;      // Maximum number of steps before time expires
constexpr int kDefaultMagicWallSteps = 20;  // Number of steps before magic walls expire
constexpr int kDefaultAmoebaMaxSize = 20;   // Maximum number of amoeba before they collapse

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
        {"max_steps", GameParameter(kDefaultMaxSteps)},
        {"magic_wall_steps", GameParameter(kDefaultMagicWallSteps)},
        {"amoeba_max_size", GameParameter(kDefaultAmoebaMaxSize)},
        {"rng_seed", GameParameter(0)},
        {"grid", GameParameter(std::string(kDefaultGrid))}
    }};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BDMinesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string BDMinesState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome:", move_id);
  } else {
    SPIEL_CHECK_GE(move_id, 0);
    SPIEL_CHECK_LT(move_id, kNumActions);
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
  return std::vector<double>{sum_reward_};
}

std::vector<double> BDMinesState::Rewards() const {
  // reward for most recent state transition
  return std::vector<double>{current_reward_};
}

std::string BDMinesState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void BDMinesState::ObservationTensor(Player player,
                                 std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kNumVisibleCellType, grid_.num_rows * grid_.num_cols}, true);

  for (int i = 0; i < grid_.num_rows * grid_.num_cols; ++i) {
    int channel = static_cast<int>(grid_.elements[i].visible_type);
    view[{channel, i}] = 1;
  }
}

BDMinesState::BDMinesState(std::shared_ptr<const Game> game, int max_steps, int magic_wall_steps,
               int amoeba_max_size, int gems_required, Grid grid, int rng_seed)
    : State(game),
      steps_remaining_(max_steps),
      magic_wall_steps_(magic_wall_steps),
      amoeba_max_size_(amoeba_max_size),
      gems_required_(gems_required),
      grid_(grid),
      rng_(rng_seed) {
  gems_collected_ = 0;
  magic_active_ = false;
  amoeba_size_ = 0;
  amoeba_swap_ = kNullElement;
  amoeba_enclosed_ = true;
  sum_reward_ = 0.0;
  current_reward_ = 0.0;
  cur_player_ = 0;
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
} // namespace


// Game dynamic function
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
  return InBounds(index, action) && grid_.elements[new_index].properties & property > 0;
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
void BDMinesState::Push(int index, int action) {
  int new_index = IndexFromAction(index, action);
  // Check if same direction past boulder is empty so that theres room to push
  if (IsType(new_index, kElEmpty, action)) {
    SetItem(index, kElEmpty);
    SetItem(new_index, kElRockford);
    // Check if the boulder will become stationary or falling
    int rock_index = IndexFromAction(new_index, action);
    bool is_empty = IsType(rock_index, kElEmpty, Directions::kDown);
    SetItem(new_index, is_empty ? kElBoulderFalling : kElBoulder, action);
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
void BDMinesState::Explode(int index, int action) {
  int new_index = IndexFromAction(index, action);
  SetItem(new_index, kElementToExplosion.at(GetItem(new_index)));
  // Recursively check all directions for chain explosions
  for (int dir = 0; dir < kNumDirections; ++dir) {
    if (dir == Directions::kNone || !InBounds(new_index, dir)) {continue;}
    if (HasProperty(new_index, ElementProperties::kCanExplode, dir)) {
      Explode(new_index, dir);
    } else if (HasProperty(new_index, ElementProperties::kConsumable, dir)) {
      SetItem(new_index, kElExplosionEmpty, dir);
    }
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
    Explode(index, Directions::kDown);
  } else if (IsType(index, kElWallMagicOn, Directions::kDown) || 
             IsType(index, kElWallMagicOn, Directions::kDown)) {
    MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElBoulderFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElBoulderFalling);
  } else {    // Default options is for falling boulder to become stationary
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
  } else if (HasProperty(index, ElementProperties::kCanExplode, Directions::kDown)) {
    Explode(index, Directions::kDown);
  } else if (IsType(index, kElWallMagicOn, Directions::kDown) || 
             IsType(index, kElWallMagicOn, Directions::kDown)) {
    MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
  } else if (CanRollLeft(index)) {    // Roll left/right
    RollLeft(index, kElDiamondFalling);
  } else if (CanRollRight(index)) {
    RollRight(index, kElDiamondFalling);
  } else {    // Default options is for falling diamond to become stationary
    SetItem(index, kElDiamond);
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
    current_reward_ += kGemPoints.at(GetItem(index));
    sum_reward_ += kGemPoints.at(GetItem(index));
    MoveItem(index, action);
  } else if (IsActionHorz(action) && IsType(index, kElBoulder, action)) {
    // Push boulder only if action is horizontal
    Push(index, action);
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
    Explode(index);
  } else if (IsType(index, kElEmpty, new_dir)) {
    // Fireflys always try to rotate left, otherwise continue forward
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
    Explode(index);
  } else if (IsType(index, kElEmpty, new_dir)) {
    // Fireflys always try to rotate right, otherwise continue forward
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
  bool will_grow = (rng_() % 128) > 32;
  int grow_dir = rng_() % kNumActions;
  if (will_grow && IsType(index, kElEmpty, grow_dir) || IsType(index, kElDirt, grow_dir)) {
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
  for (auto e : grid_.elements) {
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
    magic_wall_steps_ = std::min(magic_wall_steps_ - 1, 0);
  }
  // Check if still active
  magic_active_ = (magic_active_ && magic_wall_steps_ > 0);
}

void BDMinesState::DoApplyAction(Action move) {
  StartScan();

  EndScan();
}

std::vector<Action> BDMinesState::LegalActions() const {
  if (IsChanceNode()) {
    // <TODO> Does this use None, or is this a dummy??
    return {0};
  } else if (IsTerminal()) {
    return {};
  } else {
    return {Directions::kNone, Directions::kUp, Directions::kLeft, Directions::kDown, Directions::kRight};
  }
}

std::vector<std::pair<Action, double>> BDMinesState::ChanceOutcomes() const {
  return {{0, 1.0}};
}

std::string BDMinesState::ToString() const {
  std::string out_str;
  int col_counter = 0;
  for (const auto el : grid_.elements) {
    ++col_counter;
    out_str += el.id;
    if (col_counter == grid_.num_cols) {
      out_str += '\n';
    }
  }
  out_str += "time left: " + std::to_string(steps_remaining_) + ", ";
  out_str += "gems required: " + std::to_string(gems_required_)  + ", ";
  out_str += "gems collectred: " + std::to_string(gems_collected_);
  return out_str;
}

std::unique_ptr<State> BDMinesState::Clone() const {
  return std::unique_ptr<State>(new BDMinesState(*this));
}

// ------ game -------

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
  return -max_steps_; 
}

double BDMinesGame::MaxUtility() const {
  return max_steps_;
}

std::vector<int> BDMinesGame::ObservationTensorShape() const {
  return {kNumVisibleCellType, grid_.num_rows, grid_.num_cols};
}

Grid BDMinesGame::ParseGrid(const std::string& grid_string) {
  std::istringstream ss(grid_string);
  std::string curr_line;
  Grid grid;

  // Level properties are on first line
  if (!(ss >> grid_.num_cols >> grid_.num_rows >> max_steps_ >> gems_required_)) {
    SpielFatalError("Missing width, height, maximum steps, and/or gems required on first line");
  }
  std::getline(ss, curr_line, '\n'); // Clear remaining new line

  // Grid content follows
  int row_counter = 0;
  while (std::getline(ss, curr_line, '\n')) {
    ++row_counter;
    // Check for proper number of columns
    if (curr_line.size() != grid_.num_cols) {
      SpielFatalError("Grid line" + std::to_string(row_counter) + "doesn't have correct number of elements.");
    }
    // Check each element in row
    for (const auto c : curr_line) {
      auto it = kCharElementMap.find(c);
      if (it != kCharElementMap.end()) {
        grid_.elements.push_back(it->second);
      } else {
        SpielFatalError("Unknown element char: " + c);
      }
    }
  }

  // Ensure we read proper number of rows
  if (row_counter != grid_.num_rows) {
    SpielFatalError("Incorrect number of rows, got " + std::to_string(row_counter) + " but need " + std::to_string(grid_.num_rows));
  }

  return grid;
}

BDMinesGame::BDMinesGame(const GameParameters& params)
    : Game(kGameType, params),
      max_steps_(ParameterValue<int>("max_steps")),
      magic_wall_steps_(ParameterValue<int>("magic_wall_steps")),
      amoeba_max_size_(ParameterValue<int>("amoeba_max_size")),
      rng_seed_(ParameterValue<int>("rng_seed")),
      grid_(ParseGrid(ParameterValue<std::string>("grid"))) {}

}  // namespace bd_mines
}  // namespace open_spiel
