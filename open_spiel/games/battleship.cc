// Copyright 2020 DeepMind Technologies Ltd. All rights reserved.
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
// limitations under the License.#include "open_spiel/games/battleship.h"

#include "open_spiel/games/battleship.h"

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/strings/strip.h"

namespace open_spiel {
namespace battleship {

const double kFloatTolerance = 1e-9;

BattleshipState::BattleshipState(
    const std::shared_ptr<const BattleshipGame> bs_game)
    : State(bs_game), bs_game_(bs_game) {}

Player BattleshipState::CurrentPlayer() const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  // The players place the ships on the board in turns, starting from Player 1.
  //
  // NOTE: It is important whether or not the players place all their ships at
  // once or not for correlated equilibria purposes. This is because in
  // correlated equilibria, the recommender can stop issuing recommendations
  // after a player deviates from a recommended *action*.
  if (!AllShipsPlaced()) {
    // In this case, if an even number (possibly 0) of ships have been placed,
    // then it is Player 1's turn to act next. Else, it is Player 2's.
    if (NumShipsPlaced() % 2 == 0) {
      return Player{0};
    } else {
      return Player{1};
    }
  } else {
    // In this case, all ships have been placed. The players can take turns for
    // their next moves, starting from Player 1.

    // First, we check whether the game is over.
    //
    // The game is over only in two cases:
    // * Both players have taken `conf.num_shots` shots; or
    // * At least one player has lost all of their ships.
    if (moves_.size() == 2 * conf.ships.size() + 2 * conf.num_shots) {
      return kTerminalPlayerId;
    } else if (AllPlayersShipsSank(Player{0}) ||
               AllPlayersShipsSank(Player{1})) {
      return kTerminalPlayerId;
    }

    // If we are here, the game is not over yet.
    if (moves_.size() % 2 == 0) {
      return Player{0};
    } else {
      return Player{1};
    }
  }
}

std::vector<Action> BattleshipState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  } else {
    const Player player = CurrentPlayer();
    const BattleshipConfiguration& conf = bs_game_->configuration;

    std::vector<Action> actions;
    actions.reserve(NumDistinctActions());

    if (!AllShipsPlaced()) {
      // If we are here, we still have some ships to place on the board.
      //
      // First, we find the first ship that hasn't been placed on the board yet.
      const Ship next_ship = NextShipToPlace(player);

      // Horizontal placement.
      if (next_ship.length <= conf.board_width) {
        for (int row = 0; row < conf.board_height; ++row) {
          for (int col = 0; col < conf.board_width - next_ship.length + 1;
               ++col) {
            const ShipPlacement placement(ShipPlacement::Direction::Horizontal,
                                          /* ship = */ next_ship,
                                          /* tl_corner = */ Cell{row, col});
            if (PlacementDoesNotOverlap(placement, player)) {
              actions.push_back(SerializeShipPlacementAction(placement));
            }
          }
        }
      }

      // Vertical placement.
      //
      // NOTE: vertical placement is defined only for ships with length more
      //     than one. This avoids duplicating placement actions for 1x1 ships.
      if (next_ship.length > 1 && next_ship.length <= conf.board_height) {
        for (int row = 0; row < conf.board_height - next_ship.length + 1;
             ++row) {
          for (int col = 0; col < conf.board_width - next_ship.length; ++col) {
            const ShipPlacement placement(ShipPlacement::Direction::Vertical,
                                          /* ship = */ next_ship,
                                          /* tl_corner = */ Cell{row, col});
            if (PlacementDoesNotOverlap(placement, player)) {
              actions.push_back(SerializeShipPlacementAction(placement));
            }
          }
        }
      }

      // FIXME(gfarina): It would be better to have a check of this time at game
      //     construction time.
      if (actions.empty()) {
        SpielFatalError(
            "Battleship: it is NOT possible to fit all the ships on the "
            "board!");
      }
    } else {
      // In this case, the only thing the player can do is to shoot on a cell
      //
      // Depending on whether repeated shots are allowed or not, we might filter
      // out some cells.
      for (int row = 0; row < conf.board_height; ++row) {
        for (int col = 0; col < conf.board_width; ++col) {
          if (!conf.allow_repeated_shots &&
              AlreadyShot(Cell{row, col}, CurrentPlayer())) {
            // We do not duplicate the shot, so nothing to do here...
          } else {
            actions.push_back(SerializeShotAction(Shot{row, col}));
          }
        }
      }

      // SAFETY: The assert below can never fail, because when
      //     allow_repeated_shot is false, we check at game construction time
      //     that the number of shots per player is <= the number of cells in
      //     the board.
      SPIEL_DCHECK_FALSE(actions.empty());
    }

    return actions;
  }
}

std::string BattleshipState::ActionToString(Player player,
                                            Action action) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  // FIXME(gfarina): It was not clear to me from the documentation whether
  //     the next condition is always guaranteed.
  //
  //     Action ids are reused at different states, so the meaning of the same
  //     action id is contingent on the current state.
  SPIEL_CHECK_EQ(player, CurrentPlayer());

  if (!AllShipsPlaced()) {
    // If we are here, we still have some ships to place on the board.
    //
    // First, we find the first ship that hasn't been placed on the board yet.
    const ShipPlacement ship_placement = DeserializeShipPlacementAction(action);
    return ship_placement.ToString();
  } else {
    // In this case, the only thing the player can do is to shoot on a cell
    const Shot shot = DeserializeShotAction(action);
    return shot.ToString();
  }
}

std::string BattleshipState::ToString() const {
  std::string state_str;
  for (const auto& move : moves_) {
    if (move.player == Player{0}) {
      absl::StrAppend(&state_str, "/0:");
    } else {
      absl::StrAppend(&state_str, "/1:");
    }
    if (absl::holds_alternative<ShipPlacement>(move.action)) {
      absl::StrAppend(&state_str,
                      absl::get<ShipPlacement>(move.action).ToString());
    } else {
      SPIEL_DCHECK_TRUE(absl::holds_alternative<Shot>(move.action));
      absl::StrAppend(&state_str, absl::get<Shot>(move.action).ToString());
    }
  }
  return state_str;
}

std::string BattleshipState::ToPrettyString() const {
  std::string state_str;

  absl::StrAppend(&state_str, "Player 0's board:\n");
  absl::StrAppend(&state_str, OwnBoardString(Player{0}));
  absl::StrAppend(&state_str, "Player 1's board:\n");
  absl::StrAppend(&state_str, OwnBoardString(Player{1}));
  // NOTE: It is not necessary to also include the boards of the players' shots,
  //     since those can be inferred from the two ship boards.

  return state_str;
}

bool BattleshipState::IsTerminal() const {
  return CurrentPlayer() == kTerminalPlayerId;
}

std::vector<double> BattleshipState::Returns() const {
  SPIEL_CHECK_TRUE(IsTerminal());
  const BattleshipConfiguration& conf = bs_game_->configuration;

  // The description of the game in the header file contains more details
  // about how the payoffs for the players are computed at the end of the
  // game, as well as the meaning of the `loss_multiplier`.
  const double loss_multiplier = conf.loss_multiplier;

  double damage_pl1 = 0.0;
  double damage_pl2 = 0.0;
  for (const Ship& ship : conf.ships) {
    if (DidShipSink(ship, Player{0})) damage_pl1 += ship.value;
    if (DidShipSink(ship, Player{1})) damage_pl2 += ship.value;
  }

  return {damage_pl2 - loss_multiplier * damage_pl1,
          damage_pl1 - loss_multiplier * damage_pl2};
}

std::unique_ptr<State> BattleshipState::Clone() const {
  return std::make_unique<BattleshipState>(*this);
}

std::string BattleshipState::InformationStateString(Player player) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < NumPlayers());

  const BattleshipConfiguration& conf = bs_game_->configuration;
  const Player opponent = (player == Player{0}) ? Player{1} : Player{0};

  // We will need to figure out whether each of the player's shots (i) hit the
  // water, (ii) damaged but did not sink yet one of the opponent's ships, or
  // (iii) damaged and sank one of the opponent's ships.
  //
  // To be able to figure that out, we will keep track of the damage that each
  // of the opponent's ship has received so far. The vector `ship_damage`
  // contains and updates this information as each player's shot is processed
  // in order. Position i corresponds to the damage that the opponent's ship
  // in position i of bs_game->configuration.ships has suffered.
  std::vector<int> ship_damage(conf.ships.size(), 0);
  // Since in general we might have repeated shots, we cannot simply increase
  // the ship damage every time a shot hits a ship. For that, we keep track of
  // whether a cell was already hit in the past. We reuse the
  // serialization/deserialization routines for shots to map from (r, c) to
  // cell index r * board_width + c.
  std::vector<bool> cell_hit(conf.board_width * conf.board_height, false);

  std::string information_state;
  for (const auto& move : moves_) {
    if (absl::holds_alternative<ShipPlacement>(move.action)) {
      // The player observed *their own* ship placements.
      if (move.player == player) {
        absl::StrAppend(&information_state, "/");
        absl::StrAppend(&information_state,
                        absl::get<ShipPlacement>(move.action).ToString());
      }
    } else {
      const Shot& shot = absl::get<Shot>(move.action);

      if (move.player != player) {
        // If the shot came from the opponent, the player has seen it.
        absl::StrAppend(&information_state, "/oppshot_", shot.ToString());
      } else {
        const int cell_index = SerializeShotAction(shot);

        char shot_outcome = 'W';  // For 'water'.
        for (int ship_index = 0; ship_index < conf.ships.size(); ++ship_index) {
          const Ship& ship = conf.ships.at(ship_index);

          // SAFETY: the call to FindShipPlacement_ is safe, because if we are
          // here it means that all ships have been placed.
          const ShipPlacement ship_placement =
              FindShipPlacement(ship, opponent);

          if (ship_placement.CoversCell(shot)) {
            if (!cell_hit[cell_index]) {
              // This is a new hit: we have to increas the ship damage and
              // mark the cell as already hit.
              ++ship_damage.at(ship_index);
              cell_hit.at(cell_index) = true;
            }
            if (ship_damage.at(ship_index) == ship.length) {
              shot_outcome = 'S';  // For 'sunk'.
            } else {
              shot_outcome = 'H';  // For 'hit' (but not sunk).
            }
          }
        }

        // Otherwise, the player knows they shot, but also knows whether the
        // shot hit the water, hit a ship (but did not sink it), or sank a
        // ship.
        absl::StrAppend(&information_state, "/shot_", shot.ToString(), ":");
        information_state.push_back(shot_outcome);
      }
    }
  }

  return information_state;
}

std::string BattleshipState::OwnBoardString(const Player player) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < NumPlayers());

  const Player opponent = (player == Player{0}) ? Player{1} : Player{0};
  const BattleshipConfiguration& conf = bs_game_->configuration;

  // We keep the board in memory as vectors of strings. Initially, all strings
  // only contain whitespace.
  std::vector<std::string> player_board(conf.board_height,
                                        std::string(conf.board_width, ' '));

  // We start by drawing the ships on the player's board. For now, we do not
  // include any information about where the opponent shot.
  char ship_id = 'a';
  for (const auto& move : moves_) {
    if (move.player == player &&
        absl::holds_alternative<ShipPlacement>(move.action)) {
      const ShipPlacement& placement = absl::get<ShipPlacement>(move.action);

      // We now iterate over all the cells that the ship covers on the board
      // and fill in the `player_board` string representation.
      Cell cell = placement.TopLeftCorner();
      for (int i = 0; i < placement.ship.length; ++i) {
        SPIEL_DCHECK_TRUE(cell.row >= 0 && cell.row < conf.board_height);
        SPIEL_DCHECK_TRUE(cell.col >= 0 && cell.col < conf.board_width);

        // The ships do not overlap.
        SPIEL_DCHECK_EQ(player_board[cell.row][cell.col], ' ');
        player_board[cell.row][cell.col] = ship_id;

        if (placement.direction == ShipPlacement::Direction::Horizontal) {
          ++cell.col;
        } else {
          SPIEL_DCHECK_TRUE(placement.direction ==
                            ShipPlacement::Direction::Vertical);
          ++cell.row;
        }
      }

      ++ship_id;
    }
  }
  // It is impossible that the player placed more ships than they own.
  SPIEL_DCHECK_LE(ship_id, 'a' + conf.ships.size());

  // We now include the opponent's shots on the player's board.
  for (const auto& move : moves_) {
    if (move.player == opponent && absl::holds_alternative<Shot>(move.action)) {
      const Shot& shot = absl::get<Shot>(move.action);

      if (player_board[shot.row][shot.col] == ' ') {
        player_board[shot.row][shot.col] = '*';
      } else {
        SPIEL_DCHECK_TRUE(std::isalpha(player_board[shot.row][shot.col]));

        // The shot hit one of the ships. In this case, we use the uppercase
        // letter corresponding to the ship.
        player_board[shot.row][shot.col] =
            std::toupper(player_board[shot.row][shot.col]);
      }
    }
  }

  std::string output;
  absl::StrAppend(&output, "+", std::string(conf.board_width, '-'), "+\n");
  for (const auto& row : player_board) {
    absl::StrAppend(&output, "|", row, "|\n");
  }
  absl::StrAppend(&output, "+", std::string(conf.board_width, '-'), "+\n");
  return output;
}

std::string BattleshipState::ShotsBoardString(const Player player) const {
  SPIEL_CHECK_TRUE(player >= 0 && player < NumPlayers());

  const Player opponent = (player == Player{0}) ? Player{1} : Player{0};
  const BattleshipConfiguration& conf = bs_game_->configuration;

  // We keep the board in memory as vectors of strings. Initially, all strings
  // only contain whitespace.
  std::vector<std::string> shots_board(conf.board_height,
                                       std::string(conf.board_width, ' '));

  // We fill in the board that represents the outcome of the player's
  // shots.
  //
  // We start by adding a '@' to all the positions where the player shot.
  // That corresponds to marking all shots as 'misses'. We will promote them
  // to ship-hit marks '#' in a shortly.
  for (const auto& move : moves_) {
    if (move.player == player && absl::holds_alternative<Shot>(move.action)) {
      const Shot& shot = absl::get<Shot>(move.action);

      SPIEL_DCHECK_EQ(shots_board[shot.row][shot.col], ' ');
      shots_board[shot.row][shot.col] = '@';
    }
  }

  // Now, we iterate through the ship placements of the opponent. If a ship
  // has been hit, then we will promote '@' to '#'.
  for (const auto& move : moves_) {
    if (move.player == opponent &&
        absl::holds_alternative<ShipPlacement>(move.action)) {
      const ShipPlacement& placement = absl::get<ShipPlacement>(move.action);

      // We now iterate over all the cells that the ship covers on the board
      // and fill in the `player_board` string representation.
      Cell cell = placement.TopLeftCorner();
      for (int i = 0; i < placement.ship.length; ++i) {
        SPIEL_DCHECK_TRUE(cell.row >= 0 && cell.row < conf.board_height);
        SPIEL_DCHECK_TRUE(cell.col >= 0 && cell.col < conf.board_width);

        if (shots_board[cell.row][cell.col] == '@') {
          shots_board[cell.row][cell.col] = '#';
        } else {
          // Ships cannot intersect, so it's impossible that we would go over
          // a '#'.
          SPIEL_DCHECK_EQ(shots_board[cell.row][cell.col], ' ');

          std::cout << "(" << cell.row << ", " << cell.col << ") -> '"
                    << shots_board[cell.row][cell.col] << "'\n";
        }

        if (placement.direction == ShipPlacement::Direction::Horizontal) {
          ++cell.col;
        } else {
          SPIEL_DCHECK_TRUE(placement.direction ==
                            ShipPlacement::Direction::Vertical);
          ++cell.row;
        }
      }
    }
  }

  std::string output;
  absl::StrAppend(&output, "+", std::string(conf.board_width, '-'), "+\n");
  for (const auto& row : shots_board) {
    absl::StrAppend(&output, "|", row, "|\n");
  }
  absl::StrAppend(&output, "+", std::string(conf.board_width, '-'), "+\n");
  return output;
}

std::string BattleshipState::ObservationString(Player player) const {
  std::string output = "State of player's ships:\n";
  absl::StrAppend(&output, OwnBoardString(player));
  absl::StrAppend(&output, "\nPlayer's shot outcomes:\n");
  absl::StrAppend(&output, ShotsBoardString(player));
  return output;
}

void BattleshipState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_GT(moves_.size(), 0);
  // XXX(gfarina): It looks like SPIEL_CHECK_EQ wants to print a PlayerAction
  //     on failure, but std::cout was not overloaded. For now I moved to a
  //     SPIEL_CHECK_TRUE.
  SPIEL_CHECK_TRUE((history_.back() == PlayerAction{player, action}));

  history_.pop_back();
  moves_.pop_back();
  --move_number_;
}

void BattleshipState::DoApplyAction(Action action) {
  SPIEL_CHECK_FALSE(IsTerminal());

  const auto legal_actions = LegalActions();

  // Instead of validating the input action, we simply check that it is one
  // of the legal actions. This effectively moves all the burden of validation
  // onto `LegalActions`.
  SPIEL_CHECK_EQ(std::count(legal_actions.begin(), legal_actions.end(), action),
                 1);
  moves_.emplace_back(DeserializeGameMove(action));
}

int BattleshipState::NumShipsPlaced() const {
  return static_cast<int>(
      std::count_if(moves_.begin(), moves_.end(), [](const GameMove& move) {
        return absl::holds_alternative<ShipPlacement>(move.action);
      }));
}

bool BattleshipState::AllShipsPlaced() const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  return NumShipsPlaced() == 2 * conf.ships.size();
}

bool BattleshipState::IsShipPlaced(const Ship& ship,
                                   const Player player) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  for (const auto& move : moves_) {
    if (move.player == player &&
        absl::holds_alternative<ShipPlacement>(move.action) &&
        absl::get<ShipPlacement>(move.action).ship.id == ship.id) {
      return true;
    }
  }
  return false;
}

Ship BattleshipState::NextShipToPlace(const Player player) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  const BattleshipConfiguration& conf = bs_game_->configuration;
  const auto next_ship = std::find_if_not(
      conf.ships.begin(), conf.ships.end(), [this, player](const Ship& ship) {
        return this->IsShipPlaced(ship, player);
      });

  SPIEL_DCHECK_TRUE(next_ship != conf.ships.end());
  return *next_ship;
}

ShipPlacement BattleshipState::FindShipPlacement(const Ship& ship,
                                                 const Player player) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  // NOTE: for now, this function is intended to be called only after all the
  //     ships have been placed.
  SPIEL_DCHECK_TRUE(AllShipsPlaced());

  // We iterate through the moves of the player, filtering those that belong
  // to the requested one. We match ships based on their unique id.
  for (const auto& move : moves_) {
    if (move.player == player &&
        absl::holds_alternative<ShipPlacement>(move.action)) {
      const ShipPlacement& placement = absl::get<ShipPlacement>(move.action);
      if (placement.ship.id == ship.id) {
        return placement;
      }
    }
  }
  SPIEL_DCHECK_TRUE(false);  // Unreachable!
}

bool BattleshipState::PlacementDoesNotOverlap(const ShipPlacement& proposed,
                                              const Player player) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(proposed.TopLeftCorner().row, 0);
  SPIEL_DCHECK_LT(proposed.TopLeftCorner().row, conf.board_height);
  SPIEL_DCHECK_GE(proposed.TopLeftCorner().col, 0);
  SPIEL_DCHECK_LT(proposed.TopLeftCorner().col, conf.board_width);

  SPIEL_DCHECK_GE(proposed.BottomRightCorner().row, 0);
  SPIEL_DCHECK_LT(proposed.BottomRightCorner().row, conf.board_height);
  SPIEL_DCHECK_GE(proposed.BottomRightCorner().col, 0);
  SPIEL_DCHECK_LT(proposed.BottomRightCorner().col, conf.board_width);

  for (const auto& move : moves_) {
    if (move.player == player &&
        absl::holds_alternative<ShipPlacement>(move.action)) {
      const ShipPlacement& prior_placement =
          absl::get<ShipPlacement>(move.action);

      if (proposed.OverlapsWith(prior_placement)) {
        return false;
      }
    }
  }
  return true;
}

bool BattleshipState::DidShipSink(const Ship& ship, const Player player) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  // NOTE: for now, this function is intended to be called only after all the
  //     ships have been placed.
  SPIEL_DCHECK_TRUE(AllShipsPlaced());

  const BattleshipConfiguration& conf = bs_game_->configuration;

  // We go through the history of shots by the opponent, and filter those that
  // intersect with the ship.
  std::vector<Cell> hits;
  const ShipPlacement placement = FindShipPlacement(ship, player);
  for (const auto& move : moves_) {
    if (absl::holds_alternative<Shot>(move.action)) {
      const Shot& shot = absl::get<Shot>(move.action);
      if (move.player != player && placement.CoversCell(shot)) {
        hits.push_back(shot);
      }
    }
  }

  // We need to account for the possibility that the opponent hit the same
  // cell more than once, when `allow_repeated_shots = true`. For this, we
  // de-duplicate the vector of hits.
  std::sort(hits.begin(), hits.end());
  const auto new_end = std::unique(hits.begin(), hits.end());
  SPIEL_DCHECK_TRUE(new_end == hits.end() || conf.allow_repeated_shots);

  const size_t num_unique_shots = std::distance(hits.begin(), new_end);
  SPIEL_DCHECK_LE(num_unique_shots, ship.length);

  return num_unique_shots == ship.length;
}

bool BattleshipState::AllPlayersShipsSank(const Player player) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  const BattleshipConfiguration& conf = bs_game_->configuration;

  for (const Ship& ship : conf.ships) {
    if (!DidShipSink(ship, player)) return false;
  }
  return true;
}

bool BattleshipState::AlreadyShot(const Shot& shot, const Player player) const {
  SPIEL_DCHECK_TRUE(player == Player{0} || player == Player{1});

  return std::find_if(moves_.begin(), moves_.end(),
                      [player, shot](const GameMove& move) {
                        return move.player == player &&
                               absl::holds_alternative<Shot>(move.action) &&
                               absl::get<Shot>(move.action) == shot;
                      }) != moves_.end();
}

Action BattleshipState::SerializeShipPlacementAction(
    const ShipPlacement& ship_placement) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  Action shift = 0;
  if (ship_placement.direction == ShipPlacement::Direction::Vertical) {
    shift = conf.board_width * conf.board_height;
  }

  return shift + SerializeShotAction(ship_placement.TopLeftCorner());
}

Action BattleshipState::SerializeShotAction(const Shot& shot) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(shot.row, 0);
  SPIEL_DCHECK_GE(shot.col, 0);
  SPIEL_DCHECK_LT(shot.row, conf.board_height);
  SPIEL_DCHECK_LT(shot.col, conf.board_width);
  return shot.row * conf.board_width + shot.col;
}

ShipPlacement BattleshipState::DeserializeShipPlacementAction(
    const Action action) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(action, 0);
  SPIEL_DCHECK_LT(action, 2 * conf.board_width * conf.board_height);

  const Player player = CurrentPlayer();
  const Ship ship = NextShipToPlace(player);

  // FIXME(gfarina): Here we are exploiting the detail that Shot == Cell as
  //     a type. Perhaps it would be better to avoid this trick.
  ShipPlacement::Direction direction;
  Cell tl_corner;
  if (action >= conf.board_width * conf.board_height) {
    direction = ShipPlacement::Direction::Vertical;
    tl_corner =
        DeserializeShotAction(action - conf.board_width * conf.board_height);
  } else {
    direction = ShipPlacement::Direction::Horizontal;
    tl_corner = DeserializeShotAction(action);
  }

  return ShipPlacement(/* direction */ direction, /* ship = */ ship,
                       /* tl_corner = */ tl_corner);
}

Shot BattleshipState::DeserializeShotAction(const Action action) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(action, 0);
  SPIEL_DCHECK_LT(action, conf.board_width * conf.board_height);
  return Shot{/* row = */ static_cast<int>(action / conf.board_width),
              /* col = */ static_cast<int>(action % conf.board_width)};
}

GameMove BattleshipState::DeserializeGameMove(const Action action) const {
  if (!AllShipsPlaced()) {
    // If we are here, the action represents a `ShipPlacement`.
    return GameMove{CurrentPlayer(), DeserializeShipPlacementAction(action)};
  } else {
    // Otherwise, the action is a `Shot`.
    return GameMove{CurrentPlayer(), DeserializeShotAction(action)};
  }
}

// Facts about the game
//
// NOTE: The utility type is overridden in the game constructor and set to
//     `kZeroSum` when the loss multiplier is 1.0.
const GameType kGameType{
    /* short_name = */ "battleship",
    /* long_name = */ "Battleship",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /* max_num_players = */ 2,
    /* min_num_players = */ 2,
    /* provides_information_state_string = */ true,
    /* provides_information_state_tensor = */ false,
    /* provides_observation_string = */ true,
    /* provides_observation_tensor = */ false,
    /* parameter_specification = */
    {{"board_width", GameParameter(kDefaultBoardWidth)},
     {"board_height", GameParameter(kDefaultBoardHeight)},
     {"ship_sizes", GameParameter(kDefaultShipSizes)},
     {"ship_values", GameParameter(kDefaultShipValues)},
     {"num_shots", GameParameter(kDefaultNumShots)},
     {"allow_repeated_shots", GameParameter(kDefaultAllowRepeatedShots)},
     {"loss_multiplier", GameParameter(kDefaultLossMultiplier)}}};

constexpr int kMaxDimension = 10;

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<const BattleshipGame>(params);
}
REGISTER_SPIEL_GAME(kGameType, Factory);

BattleshipGame::BattleshipGame(const GameParameters& params)
    : Game(kGameType, params) {
  configuration.board_width = ParameterValue<int>("board_width");
  SPIEL_CHECK_GE(configuration.board_width, 0);
  SPIEL_CHECK_LE(configuration.board_width, kMaxDimension);

  configuration.board_height = ParameterValue<int>("board_height");
  SPIEL_CHECK_GE(configuration.board_height, 0);
  SPIEL_CHECK_LE(configuration.board_height, kMaxDimension);

  // NOTE: It is *very* important to clone ship_sizes and ship_values onto the
  //     stack, otherwise we would run into undefined behavior without
  //     warning, because ParameterValue() returns a temporary without
  //     storage, and absl::string_view would amount to a fat pointer to a
  //     temporary.
  const std::string ship_sizes_param_str =
      ParameterValue<std::string>("ship_sizes");
  const std::string ship_values_param_str =
      ParameterValue<std::string>("ship_values");

  // First, we check that the list starts with '[' and ends with ']'.
  absl::string_view ship_sizes_param =
      absl::StripAsciiWhitespace(ship_sizes_param_str);
  SPIEL_CHECK_TRUE(absl::ConsumePrefix(&ship_sizes_param, "["));
  SPIEL_CHECK_TRUE(absl::ConsumeSuffix(&ship_sizes_param, "]"));

  absl::string_view ship_values_param =
      absl::StripAsciiWhitespace(ship_values_param_str);
  SPIEL_CHECK_TRUE(absl::ConsumePrefix(&ship_values_param, "["));
  SPIEL_CHECK_TRUE(absl::ConsumeSuffix(&ship_values_param, "]"));

  const std::vector<absl::string_view> ship_sizes =
      absl::StrSplit(ship_sizes_param, ';');
  const std::vector<absl::string_view> ship_values =
      absl::StrSplit(ship_values_param, ';');
  SPIEL_CHECK_EQ(ship_sizes.size(), ship_values.size());

  for (size_t ship_index = 0; ship_index < ship_sizes.size(); ++ship_index) {
    Ship ship;
    ship.id = ship_index;

    SPIEL_CHECK_TRUE(absl::SimpleAtoi(ship_sizes.at(ship_index), &ship.length));
    SPIEL_CHECK_TRUE(absl::SimpleAtod(ship_values.at(ship_index), &ship.value));

    SPIEL_CHECK_TRUE(ship.length < configuration.board_width ||
                     ship.length <= configuration.board_height);
    SPIEL_CHECK_GE(ship.value, 0.0);

    configuration.ships.push_back(ship);
  }
  SPIEL_CHECK_GT(configuration.ships.size(), 0);

  // XXX(gfarina): The next restriction is not really intrinsic in the game,
  //     but we need it to pretty print the board status in
  //     `ObservationString`, since we use ASCII letters (a-z) to identify the
  //     ships.
  SPIEL_CHECK_LE(configuration.ships.size(), 26);

  configuration.num_shots = ParameterValue<int>("num_shots");
  SPIEL_CHECK_GT(configuration.num_shots, 0);

  configuration.allow_repeated_shots =
      ParameterValue<bool>("allow_repeated_shots");
  if (!configuration.allow_repeated_shots) {
    SPIEL_CHECK_LE(configuration.num_shots,
                   configuration.board_width * configuration.board_height);
  }

  configuration.loss_multiplier = ParameterValue<double>("loss_multiplier");

  if (std::abs(configuration.loss_multiplier - 1.0) < kFloatTolerance) {
    game_type_.utility = GameType::Utility::kZeroSum;
  }
}

int BattleshipGame::NumDistinctActions() const {
  // See comment about (de)serialization of actions in `BattleshipState`.
  return 2 * configuration.board_width * configuration.board_height;
}

std::unique_ptr<State> BattleshipGame::NewInitialState() const {
  const auto ptr =
      std::dynamic_pointer_cast<const BattleshipGame>(shared_from_this());
  return std::make_unique<BattleshipState>(ptr);
}

double BattleshipGame::MinUtility() const {
  // The final payoff is a sum of values of ships we destroyed, minus sum of
  // our own destroyed ships multiplied by the loss multiplier.
  //
  // So, here we take the worst possible case: we destroy no ship and all of
  // our ships are destroyed.
  //
  // Note: the implementation below is only correct if the ship values are >=
  // 0. That condition is checked at game construction time. However, we allow
  // for a negative loss_multiplier.
  double min_utility = 0.0;
  if (configuration.loss_multiplier > 0.0) {
    for (const Ship& ship : configuration.ships) {
      SPIEL_DCHECK_GE(ship.value, 0.0);
      min_utility -= configuration.loss_multiplier * ship.value;
    }
  }

  return min_utility;
}

double BattleshipGame::MaxUtility() const {
  // The final payoff is a sum of values of ships we destroyed, minus sum of
  // our own destroyed ships multiplied by the loss multiplier.
  //
  // So, here we take the best possible case: we destroy all of the opponent's
  // ship and have none of ours sunk.
  //
  // Note: the implementation below is only correct if the ship values are >=
  // 0. That condition is checked at game construction time. However, we allow
  // for a negative loss_multiplier.
  double max_utility = 0.0;
  for (const Ship& ship : configuration.ships) {
    SPIEL_DCHECK_GE(ship.value, 0.0);
    max_utility += ship.value;
  }

  if (configuration.loss_multiplier < 0.0) {
    max_utility *= (1.0 - configuration.loss_multiplier);
  }

  return max_utility;
}

double BattleshipGame::UtilitySum() const {
  if (std::abs(configuration.loss_multiplier - 1.0) < kFloatTolerance) {
    return 0.0;
  } else {
    SpielFatalError(
        "Called `UtilitySum()` on a general sum Battleship game: set "
        "loss_multiplier = 1.0 for a zero-sum game.");
  }
}

int BattleshipGame::MaxGameLength() const {
  // Each player has to place their ships, plus potentially as many turns as
  // the number of shots
  return 2 * (configuration.ships.size() + configuration.num_shots);
}
}  // namespace battleship
}  // namespace open_spiel
