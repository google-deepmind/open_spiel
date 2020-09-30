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
// limitations under the License.

// Parametric two-player Battleship game, as introduced in [1]. It is inspired
// by the famous [board game][wikipedia].
//
//
// Game dynamics
// =============
//
// > The following description is loosely taken from the
// > [Wikipedia page][wikipedia] for the Battlship game.
//
// The game is played on two grids, one for each player. The grids have equal
// size, specifically `board_height` rows and `board_width` columns, where
// `board_height` and `board_width` are parameters passed to the generator.
//
// [wikipedia]: https://en.wikipedia.org/wiki/Battleship_(game)
//
// The game has two phases: ship placement and war:
//
// Ship placement phase
// --------------------
//
// During the ship placement phase, each player
// secretly arranges their ships on their grid. For the purposes of this
// generator, we assume that the players place one ship at a time, without
// revealing their position.
//
// Each ship occupies a number of consecutive squares on the grid, arranged
// either horizontally or vertically. The number of squares for each ship is
// determined by the type of the ship. The ships cannot overlap (i.e., only one
// ship can occupy any given square in the grid). The lengths and values of
// ships are the same for each player.
//
// The number of ships, as well as their lengths and values are parameters that
// can be specified at game generation time.
//
// War phase
// ---------
//
// After the ships have been positioned, the game proceeds with `num_rounds`
// rounds. In each round, each player (starting from Player 1) takes a turn to
// announce a target square in the opponent's grid which is to be shot at.
// Depending on the flags passed in to the generator, a player might or might
// not be able to shoot in a previously-selected position (default: no).
//
// The opponent announces whether or not the square is occupied by a ship. When
// all of the squares of a ship have been hit, the ship's owner announce the
// sinking of the ship.
//
// If all of a player's ships have been sunk, the game is over.
//
// Payoff computation
// ------------------
//
// The game payoffs are computed based on the following two quantities:
//
// - `damage_pl1`: this is the sum of values of Player 1's ships that have been
// sunk by Player 2.
// - `damage_pl2`: this is the sum of values of Player 2's ships that have been
// sunk by Player 1.
//
// The payoff for Player 1 is computed as `damage_pl2 - loss_multiplier *
// damage_pl1`, while the payoff for Player 2 is symmetrically computed as
// `damage_pl1 - loss_multiplier * damage_pl2`, where `loss_multiplier` is a
// parameter to the generator.
//
// When `loss_multiplier = 1`, the game is zero-sum.
// Note that currently no partial credit is awarded to a player that hit but did
// not sink a ship.
//
//
// Parameters
//     "board_width"           int     Number of columns of the game board for
//                                     each player            (default = 2)
//     "board_height"          int     Number of rows of the game board for
//                                     each player            (default = 2)
//     "ship_sizes"          [int]     Length of the ships each player has
//                                                            (default = [2])
//     "ship_values"      [double]     Value of the ships each player has
//                                                            (default = [1.0])
//     "num_shots"             int     Number of shots available to each
//                                     player                 (default = 2)
//     "allow_repeated_shots" bool     If false, the players will be prevented
//                                     from shooting multiple times at the same
//                                     cell of the board      (default = false)
//     "loss_multiplier"    double     Loss multiplier (see above). The game is
//                                     zero-sum iff the loss multiplier is 1.0
//                                                            (default = 2.0)
//
// NOTE: The list parameters must be supplied as a string of semicolon-separated
//       values, wrapped in square brackets. For example: "[1;2]" is a list with
//       elements `1` and `2`. "[1]" is a list with only one element.
//
// References
// ==========
//
// [1]:
// https://papers.nips.cc/paper/9122-correlation-in-extensive-form-games-saddle-point-formulation-and-benchmarks.pdf
//
// If you want to reference the paper that introduced the benchmark game, here
// is a Bibtex citation:
//
// ```
// @inproceedings{Farina19:Correlation,
//    title=    {Correlation in Extensive-Form Games: Saddle-Point Formulation
//               and Benchmarks},
//    author=   {Farina, Gabriele and Ling, Chun Kai and Fang, Fei and
//               Sandholm, Tuomas},
//    booktitle={Conference on Neural Information Processing Systems
//               (NeurIPS)},
//    year={2019}
// }
// ```

#ifndef OPEN_SPIEL_GAMES_BATTLESHIP_H_
#define OPEN_SPIEL_GAMES_BATTLESHIP_H_

#include <memory>
#include <vector>

#include "open_spiel/games/battleship_types.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace battleship {

constexpr int kDefaultBoardWidth = 2;
constexpr int kDefaultBoardHeight = 2;
constexpr const char* kDefaultShipSizes = "[2]";
constexpr const char* kDefaultShipValues = "[1.0]";
constexpr int kDefaultNumShots = 2;
constexpr bool kDefaultAllowRepeatedShots = false;
constexpr double kDefaultLossMultiplier = 2.0;

class BattleshipGame;
class BattleshipState final : public State {
 public:
  explicit BattleshipState(const std::shared_ptr<const BattleshipGame> bs_game);
  ~BattleshipState() = default;

  // Virtual functions inherited by OpenSpiel's `State` interface
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void UndoAction(Player player, Action action) override;

  // Draws the board of a player.
  //
  // The board is drawn as a rectangular grid of characters, with the following
  // conventions:
  // - Ships are identified with letters, starting from 'a'. For each ship, we
  //   mark all of the cells it occupies with the same letter.
  //   - Lowercase letters denote that the cell was never hit by the opponent.
  //   - Uppercase letters denote that the cell was hit by the opponent.
  // - Cells marked with '*' denote shots by the opponent, that hit water.
  // - All other cells are empty, that is, filled with a space ' ' character.
  //
  //
  // Example
  // -------
  //
  // This is what a typical 3x6 board string might looks like after 4 shots by
  // the opponent.
  //
  // ```
  // +------+
  // |*a    |
  // | A*   |
  // |   bbB|
  // +------+
  // ```
  std::string OwnBoardString(const Player player) const;

  // Draws the state of the player's shots for far.
  // This corresponds to the incremental board that the player builds over time
  // by shooting at the opponent.
  //
  // The board is drawn as a rectangular grid of characters, with the following
  // conventions:
  // - Shots that hit a ship are marked with '#'.
  // - Shots that hit the water are marked with '@'.
  // - All other cells are empty, that is, filled with a space ' ' character.
  //
  //
  // Example
  // -------
  //
  // This is what the opponent player to the example provided in
  // `OwnBoardString` will see:
  //
  // ```
  // +------+
  // |@     |
  // | #@   |
  // |     #|
  // +------+
  // ```
  std::string ShotsBoardString(const Player player) const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // Ship placement methods
  // ======================

  // Returns how many ships have been placed collectively by the two player.
  int NumShipsPlaced() const;

  // Checks whether both players have placed all of their ships.
  bool AllShipsPlaced() const;

  // Checks whether the given ship has already been placed on the board by the
  // given player.
  bool IsShipPlaced(const Ship& ship, const Player player) const;

  // Returns the ship that the given player should place on the board next.
  // Ships are placed in the order they are defined in the
  // `BattleshipConfiguration` object.
  //
  // NOTE: assumes (and checks in debug mode) that the player has not yet placed
  // all of their ships on the board.
  Ship NextShipToPlace(const Player player) const;

  // Returns the placement associated with the given ship of the given player.
  //
  // NOTE: assumes (and checks in debug mode) that the player has already placed
  // the ship on the board.
  ShipPlacement FindShipPlacement(const Ship& ship, const Player player) const;

  // Checks whether the proposed ship placement would overlap with the ships
  // that the player has placed so far.
  bool PlacementDoesNotOverlap(const ShipPlacement& proposed,
                               const Player player) const;

  // Sunken ship methods
  // ===================

  // Checks whether the given ship of the given player has been sunk by the
  // opponent.
  //
  // NOTE: assumes (and checks in debug mode) that *all* ships (of both players)
  // have already been placed.
  bool DidShipSink(const Ship& ship, const Player player) const;

  // Checks whether all of the given player's ships have been sunk by the
  // opponent.
  //
  // NOTE: assumes (and checks in debug mode) that *all* ships (of both players)
  // have already been placed.
  bool AllPlayersShipsSank(const Player player) const;

  // Shot methods
  // ============

  // Checks whether the given player has already shot the given cell.
  bool AlreadyShot(const Shot& shot, const Player player) const;

  // Action (de)serialization routines
  // =================================
  //
  // A cell (r, c) is serialized to action r * board_width + c.
  // A ship placement with top-left corner (r, c) and direction d is serialized
  // as follows:
  //   * If d is horizontal, then we serialize cell (r, c)
  //   * If d is vertical, then we serialize cell (r, c) and add shift
  //       board_width * board_height.
  //
  // This means that the highes possible action number is
  //   2 * board_width * board_height

  // Converts a `ShipPlacement` action into a unique action number, as required
  // by OpenSpiel's interface.
  //
  // See above for details about our serialization scheme.
  Action SerializeShipPlacementAction(
      const ShipPlacement& ship_placement) const;

  // Converts a `Shot` action into a unique action number, as required
  // by OpenSpiel's interface.
  Action SerializeShotAction(const Shot& shot) const;

  // Converts an action number to a `ShipPlacement`
  //
  // See above for details about our serialization scheme.
  ShipPlacement DeserializeShipPlacementAction(const Action action) const;

  // Converts an action number to a `ShipPlacement`
  //
  // See above for details about our serialization scheme.
  Shot DeserializeShotAction(const Action action) const;

  // Converts an action number to the unique move it represents at this state of
  // the game.
  GameMove DeserializeGameMove(const Action action) const;

  // Members
  // =======

  // In addition to OpenSpiel's `game` pointer, which is of type `Game`, we also
  // store a more specialized `bs_game_` back-pointer to the Battleship game
  // that generated the state.
  //
  // This is useful to avoid having to dynamic cast game_ to retrieve the
  // `BattleshipConfiguration` object.
  std::shared_ptr<const BattleshipGame> bs_game_;

  // In addition to OpenSpiel's `history_` protected member defined in `State`,
  // which is a vector of serialized action numbers, we store a friendlier
  // representation of moves that happened so far.
  //
  // The two representations will always be in sync.
  std::vector<GameMove> moves_;
};

class BattleshipGame final : public Game {
 public:
  explicit BattleshipGame(const GameParameters& params);

  // Virtual functions inherited by OpenSpiel's `Game` interface

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 0; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override;
  int MaxGameLength() const override;

  BattleshipConfiguration configuration;
};

}  // namespace battleship
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BATTLESHIP_H_
