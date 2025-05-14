// Copyright 2025 DeepMind Technologies Limited
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

#include "open_spiel/games/hive/hive.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/hive/hive_board.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace hive {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"hive",
                         /*long_name=*/"Hive",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                             // the radius of the underlying hexagonal grid.
                             // customisable to reduce computational complexity
                             // where needed. Max size of 14.
                             {"board_size", GameParameter(kDefaultBoardRadius)},
                             // expansion pieces, included by default
                             {"uses_mosquito", GameParameter(true)},
                             {"uses_ladybug", GameParameter(true)},
                             {"uses_pillbug", GameParameter(true)},
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HiveGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

HiveState::HiveState(std::shared_ptr<const Game> game, int board_size,
                     ExpansionInfo expansions, int num_bug_types,
                     bool ansi_color_output)
    : State(game),
      board_(std::min(board_size, kMaxBoardRadius), expansions),
      expansions_(expansions),
      num_bug_types_(num_bug_types),
      ansi_color_output_(ansi_color_output),
      force_terminal_(false) {}

std::string HiveState::ActionToString(Player player, Action action_id) const {
  return ActionToMove(action_id).ToUHP();
}

std::string HiveState::ToString() const {
  if (!ansi_color_output_) {
    return Serialize();
  }

  static std::string white = "\033[38;5;223m";  // white-beige-ish
  static std::string black = "\033[1;31m";      // using red to represent black
  static std::string reset = "\033[1;39m";
  static float indent_size = 2.5f;

  std::string string = "\n";
  string.reserve(Board().SquareDimensions() * Board().SquareDimensions() * 5);
  std::vector<HiveTile> top_tiles;

  // loop over valid Q, R, to generate a hexagon
  int radius = Board().Radius();
  for (int r = -radius; r <= radius; ++r) {
    // indent based on which row we are on (r). Intentionally taking the floor
    // to offset odd numbered rows
    int num_spaces = std::abs(r) * indent_size;
    for (int i = 0; i < num_spaces; ++i) {
      absl::StrAppend(&string, " ");
    }

    // print each tile on row r by iterating valid q indices
    for (int q = std::max(-radius, -r - radius);
         q <= std::min(radius, -r + radius); ++q) {
      HiveTile tile = Board().GetTopTileAt(
          {static_cast<int8_t>(q), static_cast<int8_t>(r)});

      std::ostringstream oss;
      if (tile.HasValue()) {
        if (tile.GetColour() == Colour::kWhite) {
          oss << white;
        } else {
          oss << black;
        }

        std::string uhp = tile.ToUHP();
        if (Board().GetPositionOf(tile).H() > 0) {
          uhp = absl::StrCat("^", uhp);
          top_tiles.push_back(tile);
        }

        // print the tile's UHP representation, or "-" otherwise, centered
        // around a padded 5 char long string
        int left_padding = (5 - uhp.size()) / 2;
        int right_padding = (5 - uhp.size()) - left_padding;
        for (int i = 0; i < left_padding; ++i) {
          oss << ' ';
        }

        oss << uhp;

        // use an asterisk to indicate this bug was most recently moved
        if (tile == Board().LastMovedTile()) {
          oss << "*";
          --right_padding;
        }

        for (int i = 0; i < right_padding; ++i) {
          oss << ' ';
        }

        oss << reset;
      } else {
        // use an asterisk to indicate the location of the last moved tile
        if (Board().LastMovedTile().HasValue() &&
            Board().LastMovedFrom() == HivePosition(q, r)) {
          if (Board().LastMovedTile().GetColour() == Colour::kWhite) {
            oss << white;
          } else {
            oss << black;
          }

          oss << "  *  " << reset;
        } else {
          oss << "  -  " << reset;
        }
      }
      absl::StrAppend(&string, oss.str());
    }
    absl::StrAppend(&string, "\n\n");
  }

  // print bug stacks
  for (auto tile : top_tiles) {
    HivePosition pos = Board().GetPositionOf(tile);
    absl::StrAppend(&string, tile.ToUHP());

    HiveTile below = Board().GetTileBelow(pos);
    while (below.HasValue()) {
      absl::StrAppend(&string, " > ", below.ToUHP());

      pos += {0, 0, -1};
      if (pos.H() <= 0) {
        break;
      }

      below = Board().GetTileBelow(pos);
    }

    absl::StrAppend(&string, "\n");
  }

  return string;
}

// e.g. the string "wA2 /bQ" translates to: "Move White's 2nd Ant to the
// south-west of Black's Queen"
Action HiveState::StringToAction(Player player,
                                 const std::string& move_str) const {
  // pass move?
  if (move_str == "pass") {
    return PassAction();
  }

  Move move;
  move.direction = Direction::kNumAllDirections;
  std::vector<std::string> bugs = absl::StrSplit(move_str, ' ');
  SPIEL_DCHECK_GT(bugs.size(), 0);
  SPIEL_DCHECK_LE(bugs.size(), 2);

  // first bug should always be valid
  move.from = HiveTile::UHPToTile(bugs[0]);
  if (!move.from.HasValue()) {
    SpielFatalError("HiveState::StringToAction() - invalid move string: " +
                    move_str);
  }

  // special case: if only one bug is provided, it is a 1st turn move
  if (bugs.size() == 1) {
    return MoveToAction(move);
  }

  // get second bug and its relative direction
  char c = bugs[1].front();
  if (c == '\\') {
    move.direction = Direction::kNW;
  } else if (c == '-') {
    move.direction = Direction::kW;
  } else if (c == '/') {
    move.direction = Direction::kSW;
  }

  // check last char if we haven't found a direction
  if (move.direction == Direction::kNumAllDirections) {
    c = bugs[1].back();
    if (c == '\\') {
      move.direction = Direction::kSE;
    } else if (c == '-') {
      move.direction = Direction::kE;
    } else if (c == '/') {
      move.direction = Direction::kNE;
    }
  }

  // if still no direction, it must be above
  if (move.direction == Direction::kNumAllDirections) {
    move.direction = Direction::kAbove;
  }

  // now extract just the bug + ordinal from string
  size_t start_index = bugs[1].find_first_not_of("\\-/");
  size_t end_index = bugs[1].find_last_not_of("\\-/");
  move.to = HiveTile::UHPToTile(
      bugs[1].substr(start_index, end_index - start_index + 1));

  return MoveToAction(move);
}

std::vector<double> HiveState::Returns() const {
  bool white_winner = WinConditionMet(kPlayerWhite);
  bool black_winner = WinConditionMet(kPlayerBlack);

  if (white_winner ^ black_winner) {
    return {white_winner ? 1.f : -1.f, black_winner ? 1.f : -1.f};
  }

  return {0, 0};
}

std::string HiveState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string HiveState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void HiveState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // starting indices for each 2D feature plane, variable based on expansions
  int articulation_idx = num_bug_types_ * num_players_;
  int placeable_idx = articulation_idx + 2;
  int covered_idx = placeable_idx + 2;
  int turn_idx = covered_idx + 2;

  // Treat values as a 3d-tensor, where each feature plane has square dimensions
  // (radius * 2 + 1) x (radius * 2 + 1), and contains one-hot encodings of the
  // current board state
  TensorView<3> view(values,
                     {game_->ObservationTensorShape()[0],
                      Board().SquareDimensions(), Board().SquareDimensions()},
                     true);

  int plane_idx = 0;
  Colour my_colour = PlayerToColour(player);
  Colour opposing_colour = OtherColour(my_colour);

  // populate all planes that reference a tile in play
  for (auto tile : Board().GetPlayedTiles()) {
    HivePosition pos = Board().GetPositionOf(tile);
    std::array<int, 2> indices = AxialToTensorIndex(pos);
    bool is_opposing = tile.GetColour() == opposing_colour;

    // bug type planes
    plane_idx = BugTypeToTensorIndex(tile.GetBugType()) +
                (is_opposing ? num_bug_types_ : 0);
    view[{plane_idx, indices[0], indices[1]}] = 1.0f;

    // pinned plane
    plane_idx = articulation_idx + (is_opposing ? 1 : 0);
    if (Board().IsPinned(pos)) {
      view[{plane_idx, indices[0], indices[1]}] = 1.0f;
    }

    // covered plane
    plane_idx = covered_idx + (is_opposing ? 1 : 0);
    if (Board().IsCovered(tile)) {
      view[{plane_idx, indices[0], indices[1]}] = 1.0f;
    }
  }

  // populate all planes that reference a specific position
  int radius = Board().Radius();
  for (int r = -radius; r <= radius; ++r) {
    for (int q = -radius; q <= radius; ++q) {
      HivePosition pos = {static_cast<int8_t>(q), static_cast<int8_t>(r), 0};
      std::array<int, 2> indices = AxialToTensorIndex(pos);

      // current player's turn
      view[{turn_idx, indices[0], indices[1]}] =
          static_cast<float>(current_player_);

      // player and opponent's placeable positions
      if (Board().IsPlaceable(my_colour, pos)) {
        view[{placeable_idx, indices[0], indices[1]}] = 1.0f;
      } else if (Board().IsPlaceable(opposing_colour, pos)) {
        view[{placeable_idx + 1, indices[0], indices[1]}] = 1.0f;
      }
    }
  }
}

std::unique_ptr<State> HiveState::Clone() const {
  return std::unique_ptr<State>(new HiveState(*this));
}

std::vector<Action> HiveState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }

  std::vector<Move> moves;
  std::set<Action> unique_actions;

  Board().GenerateAllMoves(&moves, PlayerToColour(current_player_),
                           move_number_);
  std::transform(moves.begin(), moves.end(),
                 std::inserter(unique_actions, unique_actions.end()),
                 [this](Move& m) { return MoveToAction(m); });

  std::vector<Action> actions(unique_actions.begin(), unique_actions.end());

  // if a player has no legal actions, then they must pass
  if (actions.empty()) {
    actions.push_back(PassAction());
  }

  return actions;
}

std::string HiveState::Serialize() const {
  return absl::StrJoin(
      {UHPGameString(), UHPProgressString(), UHPTurnString(), UHPMovesString()},
      ";", [](std::string* out, const absl::string_view& t) {
        if (!t.empty()) {
          absl::StrAppend(out, t);
        }
      });
}

Move HiveState::ActionToMove(Action action) const {
  // pass action
  if (action == PassAction()) {
    return Move{HiveTile::kNoneTile, HiveTile::kNoneTile,
                Direction::kNumAllDirections};
  }

  int64_t direction = action % Direction::kNumAllDirections;
  int64_t to = (action / Direction::kNumAllDirections) % kMaxTileCount;
  int64_t from = action / (kMaxTileCount * Direction::kNumAllDirections);

  // special case: for the first turn actions, they are encoded as playing a
  // tile on top of itself. In this case, we want "to" to be kNoneTile
  if (from == to && direction == Direction::kAbove) {
    to = HiveTile::kNoneTile;
  }

  return Move{HiveTile(from), HiveTile(to), static_cast<Direction>(direction)};
}

Action HiveState::MoveToAction(Move move) const {
  // pass move encoded as "moving no tile"
  if (move.IsPass()) {
    return PassAction();
  }

  // if there is no second bug "to", then we have a special case for first turn
  if (!move.to.HasValue()) {
    return (move.from * (kMaxTileCount * Direction::kNumAllDirections)) +
           (move.from * Direction::kNumAllDirections) + Direction::kAbove;
  }

  // as if indexing into a 3d array with indices [from][to][direction]
  return (move.from * (kMaxTileCount * Direction::kNumAllDirections)) +
         (move.to * Direction::kNumAllDirections) + move.direction;
}

std::string HiveState::UHPGameString() const {
  return absl::StrFormat("Base%s%s%s%s", expansions_.HasAny() ? "+" : "",
                         expansions_.uses_mosquito ? "M" : "",
                         expansions_.uses_ladybug ? "L" : "",
                         expansions_.uses_pillbug ? "P" : "");
}

std::string HiveState::UHPProgressString() const {
  if (move_number_ == 0) {
    return kUHPNotStarted;
  }

  if (move_number_ > game_->MaxGameLength()) {
    return kUHPDraw;
  }

  if (IsTerminal()) {
    auto returns = Returns();
    if (returns[kPlayerWhite] > returns[kPlayerBlack]) {
      return kUHPWhiteWins;
    } else if (returns[kPlayerWhite] < returns[kPlayerBlack]) {
      return kUHPBlackWins;
    } else {
      return kUHPDraw;
    }
  }

  return kUHPInProgress;
}

std::string HiveState::UHPTurnString() const {
  return absl::StrFormat("%s[%d]",
                         current_player_ == kPlayerWhite ? "White" : "Black",
                         (move_number_ + 2) / 2);
}

std::string HiveState::UHPMovesString() const {
  return absl::StrJoin(ActionsToStrings(*this, History()), ";");
}

size_t HiveState::BugTypeToTensorIndex(BugType type) const {
  size_t index = 0;
  for (uint8_t i = 0; i < static_cast<int>(BugType::kNumBugTypes); ++i) {
    if (expansions_.IsBugTypeEnabled(static_cast<BugType>(i))) {
      if (type == static_cast<BugType>(i)) {
        return index;
      }

      ++index;
    }
  }

  return -1;
}

// we assume the move is valid at this point and simply apply it
void HiveState::DoApplyAction(Action action) {
  if (action == PassAction()) {
    Board().Pass();
  } else {
    bool success = Board().MoveTile(ActionToMove(action));

    // if something has gone wrong, force end the game as a draw
    // (should only happen with with reduced board_sizes that go out of bounds)
    if (!success) {
      force_terminal_ = true;
    }
  }

  current_player_ = (++current_player_) % kNumPlayers;
}

HiveGame::HiveGame(const GameParameters& params)
    : Game(kGameType, params),
      board_radius_(ParameterValue<int>("board_size")),
      num_bug_types_(kNumBaseBugTypes),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")),
      expansions_({ParameterValue<bool>("uses_mosquito"),
                   ParameterValue<bool>("uses_ladybug"),
                   ParameterValue<bool>("uses_pillbug")}) {
  if (expansions_.uses_mosquito) {
    ++num_bug_types_;
  }

  if (expansions_.uses_ladybug) {
    ++num_bug_types_;
  }

  if (expansions_.uses_pillbug) {
    ++num_bug_types_;
  }
}

std::unique_ptr<State> HiveGame::DeserializeState(
    const std::string& uhp_string) const {
  std::vector<absl::string_view> tokens = absl::StrSplit(uhp_string, ';');
  SPIEL_DCHECK_GE(tokens.size(), 3);

  // first substring is the game string (e.g. "Base+MLP" for all expansions).
  // since we are already inside a const game object, verify that the UHP game
  // string matches what we expect it to be at this point
  SPIEL_DCHECK_TRUE(absl::StrContains(tokens[0], "Base"));
  if (expansions_.uses_mosquito) {
    SPIEL_DCHECK_TRUE(absl::StrContains(tokens[0], "M"));
  }
  if (expansions_.uses_ladybug) {
    SPIEL_DCHECK_TRUE(absl::StrContains(tokens[0], "L"));
  }
  if (expansions_.uses_pillbug) {
    SPIEL_DCHECK_TRUE(absl::StrContains(tokens[0], "P"));
  }

  std::unique_ptr<State> state = NewInitialState();
  if (tokens[1] == kUHPNotStarted) {
    return state;
  }

  // skip tokens[2] (turn string) as it is implicitly derived from the actions
  for (int i = 3; i < tokens.size(); ++i) {
    state->ApplyAction(state->StringToAction(std::string(tokens[i])));
  }

  // now verify state string (tokens[1])
  if (state->IsTerminal()) {
    if (state->Returns()[kPlayerWhite] > 0) {
      SPIEL_DCHECK_TRUE(tokens[1] == kUHPWhiteWins);
    } else if (state->Returns()[kPlayerBlack] > 0) {
      SPIEL_DCHECK_TRUE(tokens[1] == kUHPBlackWins);
    } else {
      SPIEL_DCHECK_TRUE(tokens[1] == kUHPDraw);
    }
  } else {
    SPIEL_DCHECK_TRUE(tokens[1] == kUHPInProgress);
  }

  return state;
}

std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
DeserializeUHPGameAndState(const std::string& uhp_string) {
  auto pos = uhp_string.find(';');
  auto game_str = uhp_string.substr(0, pos);
  SPIEL_DCHECK_TRUE(absl::StrContains(game_str, "Base"));

  GameParameters params{};
  params["name"] = GameParameter(kGameType.short_name);
  params["uses_mosquito"] = GameParameter(absl::StrContains(game_str, "M"));
  params["uses_ladybug"] = GameParameter(absl::StrContains(game_str, "L"));
  params["uses_pillbug"] = GameParameter(absl::StrContains(game_str, "P"));

  auto game = LoadGame(params);
  return {game, game->DeserializeState(uhp_string)};
}

}  // namespace hive
}  // namespace open_spiel
