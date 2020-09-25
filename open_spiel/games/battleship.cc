#include "open_spiel/games/battleship.h"

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"

namespace open_spiel {
namespace battleship {

const Player Player1 = Player{0};
const Player Player2 = Player{1};

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
  if (!AllShipsPlaced_()) {
    // In this case, if an even number (possibly 0) of ships have been placed,
    // then it is Player 1's turn to act next. Else, it is Player 2's.
    if (NumShipsPlaced_() % 2 == 0) {
      return Player1;
    } else {
      return Player2;
    }
  } else {
    // In this case, all ships have been placed. The players can take turns for
    // their next moves, starting from Player 1.

    // First, we check whether the game is over.
    //
    // The game is over only in two cases:
    // * Both players have taken `conf.num_shots` shots; or
    // * All ships for all players have already been sunk.
    if (moves_.size() == 2 * conf.ships.size() + 2 * conf.num_shots) {
      return kTerminalPlayerId;
    } else if (AllPlayersShipsSank_(Player1) && AllPlayersShipsSank_(Player2)) {
      return kTerminalPlayerId;
    }

    // If we are here, the game is not over yet.
    if (moves_.size() % 2 == 0) {
      return Player1;
    } else {
      return Player2;
    }
  }
}

std::vector<Action> BattleshipState::LegalActions() const {
  SPIEL_CHECK_FALSE(IsTerminal());
  const Player player = CurrentPlayer();
  const BattleshipConfiguration& conf = bs_game_->configuration;

  std::vector<Action> actions;
  actions.reserve(NumDistinctActions());

  if (!AllShipsPlaced_()) {
    // If we are here, we still have some ships to place on the board.
    //
    // First, we find the first ship that hasn't been placed on the board yet.
    const Ship next_ship = NextShipToPlace_(player);

    // Horizontal placement.
    if (next_ship.length <= conf.board_width) {
      for (int row = 0; row < conf.board_height; ++row) {
        for (int col = 0; col < conf.board_width - next_ship.length + 1;
             ++col) {
          const ShipPlacement placement(ShipPlacement::Direction::Horizontal,
                                        /* ship = */ next_ship,
                                        /* tl_corner = */ Cell{row, col});
          actions.push_back(SerializeShipPlacementAction_(placement));
        }
      }
    }

    // Vertical placement.
    //
    // NOTE: vertical placement is defined only for ships with length more than
    // one. This avoids duplicating placement actions for 1x1 ships.
    if (next_ship.length > 1 && next_ship.length <= conf.board_height) {
      for (int row = 0; row < conf.board_height; ++row) {
        for (int col = 0; col < conf.board_width - next_ship.length + 1;
             ++col) {
          const ShipPlacement placement(ShipPlacement::Direction::Vertical,
                                        /* ship = */ next_ship,
                                        /* tl_corner = */ Cell{row, col});
          actions.push_back(SerializeShipPlacementAction_(placement));
        }
      }
    }

    // FIXME(gfarina): It would be better to have a check of this time at game
    // construction time.
    if (actions.empty()) {
      SpielFatalError(
          "Battleship: it is NOT possible to fit all the ships on the board!");
    }
  } else {
    // In this case, the only thing the player can do is to shoot on a cell
    //
    // Depending on whether repeated shots are allowed or not, we might filter
    // out some cells.
    for (int row = 0; row < conf.board_height; ++row) {
      for (int col = 0; col < conf.board_width; ++col) {
        if (!conf.allow_repeated_shots &&
            AlreadyShot_(Cell{row, col}, CurrentPlayer())) {
          // We do not duplicate the shot, so nothing to do here...
        } else {
          actions.push_back(SerializeShotAction_(Shot{row, col}));
        }
      }
    }

    // SAFETY: The assert below can never fail, because when allow_repeated_shot
    // is false, we check at game construction time that the number of shots per
    // player is <= the number of cells in the board.
    SPIEL_DCHECK_FALSE(actions.empty());
  }
}

std::string BattleshipState::ActionToString(Player player,
                                            Action action) const {
  // FIXME(gfarina): It was not clear to me from the documentation whether
  // the next condition is always guaranteed.
  //
  // Action ids are reused at different states, so the meaning of the same
  // action id is contingent on the current state.
  SPIEL_CHECK_EQ(player, CurrentPlayer());

  if (!AllShipsPlaced_()) {
    // If we are here, we still have some ships to place on the board.
    //
    // First, we find the first ship that hasn't been placed on the board yet.
    const ShipPlacement ship_placement =
        DeserializeShipPlacementAction_(action);
    return ship_placement.ToString();
  } else {
    // In this case, the only thing the player can do is to shoot on a cell
    const Shot shot = DeserializeShotAction_(action);
    return shot.ToString();
  }
}

std::string BattleshipState::ToString() const {
  std::string state_str;
  for (const auto& move : moves_) {
    if (move.player == Player1) {
      absl::StrAppend(&state_str, "/1:");
    } else {
      absl::StrAppend(&state_str, "/2:");
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
    if (DidShipSink_(ship, Player1)) damage_pl1 += ship.value;
    if (DidShipSink_(ship, Player2)) damage_pl2 += ship.value;
  }

  return {damage_pl2 - loss_multiplier * damage_pl1,
          damage_pl1 - loss_multiplier * damage_pl2};
}

std::unique_ptr<State> BattleshipState::Clone() const {
  return std::make_unique<BattleshipState>(*this);
}

std::string BattleshipState::InformationStateString(Player player) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;
  const Player opponent = (player == Player1) ? Player2 : Player1;

  // We will need to figure out whether each of the player's shots
  // (i) hit the water, (ii) damaged but did not sink yet one of the opponent's
  // ships, or (iii) damaged and sank one of the opponent's ships.
  //
  // To be able to figure that out, we will keep track of the damage that each
  // of the opponent's ship has received so far. The vector `ship_damage`
  // contains and updates this information as each player's shot is processed in
  // order. Position i corresponds to the damage that the opponent's ship in
  // position i of bs_game->configuration.ships has suffered.
  std::vector<int> ship_damage(conf.ships.size(), 0);
  // Since in general we might have repeated shots, we cannot simply increase
  // the ship damage every time a shot hits a ship. For that, we keep track of
  // whether a cell was already hit in the past. We reuse the
  // serialization/deserialization routines for shots to map from (r, c) to cell
  // index r * board_width + c.
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
        const int cell_index = SerializeShotAction_(shot);

        char shot_outcome = 'W';  // For 'water'.
        for (int ship_index = 0; ship_index < conf.ships.size(); ++ship_index) {
          const Ship& ship = conf.ships.at(ship_index);

          // SAFETY: the call to FindShipPlacement_ is safe, because if we are
          // here it means that all ships have been placed.
          const ShipPlacement ship_placement =
              FindShipPlacement_(ship, opponent);

          if (ship_placement.CoversCell(shot)) {
            if (!cell_hit[cell_index]) {
              // This is a new hit: we have to increas the ship damage and mark
              // the cell as already hit.
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

        // XXX(gfarina): Apparently appending chars to a string is not a thing
        // in Abseil?
        information_state += shot_outcome;
      }
    }
  }

  return information_state;
}

std::string BattleshipState::ObservationString(Player player) const {
  return InformationStateString(player);
}

void BattleshipState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_GT(moves_.size(), 0);
  // XXX(gfarina): It looks like SPIEL_CHECK_EQ wants to print a PlayerAction
  // on failure, but std::cout was not overloaded. For now I moved to a
  // SPIEL_CHECK_TRUE.
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
  moves_.emplace_back(DeserializeGameMove_(action));
}

int BattleshipState::NumShipsPlaced_() const {
  return static_cast<int>(
      std::count_if(moves_.begin(), moves_.end(), [](const GameMove& move) {
        return absl::holds_alternative<ShipPlacement>(move.action);
      }));
}

bool BattleshipState::AllShipsPlaced_() const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  return NumShipsPlaced_() == 2 * conf.ships.size();
}

bool BattleshipState::IsShipPlaced_(const Ship& ship,
                                    const Player player) const {
  for (const auto& move : moves_) {
    if (move.player == player &&
        absl::holds_alternative<ShipPlacement>(move.action) &&
        absl::get<ShipPlacement>(move.action).ship.id == ship.id) {
      return true;
    }
  }
  return false;
}

Ship BattleshipState::NextShipToPlace_(const Player player) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;
  const auto next_ship = std::find_if_not(
      conf.ships.begin(), conf.ships.end(), [this, player](const Ship& ship) {
        return this->IsShipPlaced_(ship, player);
      });

  SPIEL_DCHECK_TRUE(next_ship != conf.ships.end());
  return *next_ship;
}

ShipPlacement BattleshipState::FindShipPlacement_(const Ship& ship,
                                                  const Player player) const {
  // NOTE: for now, this function is indented to be called only after all the
  // ships have been placed.
  SPIEL_DCHECK_TRUE(AllShipsPlaced_());

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

bool BattleshipState::DidShipSink_(const Ship& ship,
                                   const Player player) const {
  // NOTE: for now, this function is indented to be called only after all the
  // ships have been placed.
  SPIEL_DCHECK_TRUE(AllShipsPlaced_());

  const BattleshipConfiguration& conf = bs_game_->configuration;

  // We go through the history of shots by the opponent, and filter those that
  // intersect with the ship.
  std::vector<Cell> hits;
  const ShipPlacement placement = FindShipPlacement_(ship, player);
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

bool BattleshipState::AllPlayersShipsSank_(const Player player) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  for (const Ship& ship : conf.ships) {
    if (!DidShipSink_(ship, player)) return false;
  }
  return true;
}

bool BattleshipState::AlreadyShot_(const Shot& shot,
                                   const Player player) const {
  return std::find_if(moves_.begin(), moves_.end(),
                      [player, shot](const GameMove& move) {
                        return move.player == player &&
                               absl::holds_alternative<Shot>(move.action) &&
                               absl::get<Shot>(move.action) == shot;
                      }) != moves_.end();
}

Action BattleshipState::SerializeShipPlacementAction_(
    const ShipPlacement& ship_placement) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  Action shift = 0;
  if (ship_placement.direction == ShipPlacement::Direction::Vertical) {
    shift = conf.board_width * conf.board_height;
  }

  return shift + SerializeShotAction_(ship_placement.TopLeftCorner());
}

Action BattleshipState::SerializeShotAction_(const Shot& shot) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(shot.row, 0);
  SPIEL_DCHECK_GE(shot.col, 0);
  SPIEL_DCHECK_LT(shot.row, conf.board_height);
  SPIEL_DCHECK_LT(shot.col, conf.board_width);
  return shot.row * conf.board_width + shot.col;
}

ShipPlacement BattleshipState::DeserializeShipPlacementAction_(
    const Action action) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(action, 0);
  SPIEL_DCHECK_LT(action, 2 * conf.board_width * conf.board_height);

  const Player player = CurrentPlayer();
  const Ship ship = NextShipToPlace_(player);

  // FIXME(gfarina): Here we are exploiting the detail that Shot == Cell as
  // a type. Perhaps it would be better to avoid this trick.
  ShipPlacement::Direction direction;
  Cell tl_corner;
  if (action >= conf.board_width * conf.board_height) {
    direction = ShipPlacement::Direction::Vertical;
    tl_corner =
        DeserializeShotAction_(action - conf.board_width * conf.board_height);
  } else {
    direction = ShipPlacement::Direction::Horizontal;
    tl_corner = DeserializeShotAction_(action);
  }

  return ShipPlacement(/* direction */ direction, /* ship = */ ship,
                       /* tl_corner = */ tl_corner);
}

Shot BattleshipState::DeserializeShotAction_(const Action action) const {
  const BattleshipConfiguration& conf = bs_game_->configuration;

  SPIEL_DCHECK_GE(action, 0);
  SPIEL_DCHECK_LT(action, conf.board_width * conf.board_height);
  return Shot{action / conf.board_width, action % conf.board_width};
}

GameMove BattleshipState::DeserializeGameMove_(const Action action) const {
  if (!AllShipsPlaced_()) {
    // If we are here, the action represents a `ShipPlacement`.
    return GameMove{CurrentPlayer(), DeserializeShipPlacementAction_(action)};
  } else {
    // Otherwise, the action is a `Shot`.
    return GameMove{CurrentPlayer(), DeserializeShotAction_(action)};
  }
}

// Facts about the game
//
// FIXME(gfarina): Is there a way to tell OpenSpiel that the game would be
// general sum or zero sum depending on the parameters? Does it matter?
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
  return std::shared_ptr<const Game>(new BattleshipGame(params));
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

  configuration.num_shots = ParameterValue<int>("num_shots");
  SPIEL_CHECK_GE(configuration.num_shots, 0);

  const std::vector<absl::string_view> ship_sizes =
      absl::StrSplit(ParameterValue<std::string>("ship_sizes"), ',');
  const std::vector<absl::string_view> ship_values =
      absl::StrSplit(ParameterValue<std::string>("ship_values"), ',');
  SPIEL_CHECK_EQ(ship_sizes.size(), ship_values.size());

  for (size_t ship_index = 0; ship_index < ship_sizes.size(); ++ship_index) {
    Ship ship;
    ship.id = ship_index;

    SPIEL_CHECK_TRUE(absl::SimpleAtoi(ship_sizes.at(ship_index), &ship.length));
    SPIEL_CHECK_TRUE(absl::SimpleAtod(ship_values.at(ship_index), &ship.value));

    SPIEL_CHECK_TRUE(ship.length < configuration.board_width ||
                     ship.length <= configuration.board_height);
    SPIEL_CHECK_GE(ship.value, 0.0);
  }

  configuration.allow_repeated_shots =
      ParameterValue<bool>("allow_repeated_shots");
  if (!configuration.allow_repeated_shots) {
    SPIEL_CHECK_LE(configuration.num_shots,
                   configuration.board_width * configuration.board_height);
  }
  configuration.loss_multiplier = ParameterValue<double>("loss_multiplier");
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
  // Note: the implementation below is only correct if the ship values are >= 0.
  // That condition is checked at game construction time. However, we allow for
  // a negative loss_multiplier.
  double min_utility = 0.0;
  for (const Ship& ship : configuration.ships) {
    min_utility -= configuration.loss_multiplier * ship.value;
  }
  return std::min(0.0, min_utility);
}

double BattleshipGame::MaxUtility() const {
  // The final payoff is a sum of values of ships we destroyed, minus sum of
  // our own destroyed ships multiplied by the loss multiplier.
  //
  // So, here we take the best possible case: we destroy all of the opponent's
  // ship and have none of ours sunk.
  //
  // Note: the implementation below is only correct if the ship values are >= 0.
  // That condition is checked at game construction time. However, we allow for
  // a negative loss_multiplier.
  double max_utility = 0.0;
  for (const Ship& ship : configuration.ships) {
    max_utility += ship.value;
  }

  if (configuration.loss_multiplier < 0.0) {
    max_utility *= (1.0 - configuration.loss_multiplier);
  }

  return max_utility;
}

double BattleshipGame::UtilitySum() const {
  if (configuration.loss_multiplier) {
    return 0.0;
  } else {
    SpielFatalError(
        "Called `UtilitySum()` on a general sum Battleship game: set "
        "loss_multiplier = 1.0 for a zero-sum game.");
  }
}

int BattleshipGame::MaxGameLength() const {
  // Each player has to place their ships, plus potentially as many turns as the
  // number of shots
  return 2 * (configuration.ships.size() + configuration.num_shots);
}
}  // namespace battleship
}  // namespace open_spiel
