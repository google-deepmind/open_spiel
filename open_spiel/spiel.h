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

#ifndef THIRD_PARTY_OPEN_SPIEL_SPIEL_H_
#define THIRD_PARTY_OPEN_SPIEL_SPIEL_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Player ids are 0, 1, 2, ...
// Negative numbers are used for various special values.
enum PlayerId {
  // The fixed player id for chance/nature.
  kChancePlayerId = -1,
  // What is returned as a player id when the game is simultaneous.
  kSimultaneousPlayerId = -2,
  // Invalid player.
  kInvalidPlayer = -3,
  // What is returned as the player id on terminal nodes.
  kTerminalPlayerId = -4
};

// Constant representing an invalid action.
inline constexpr Action kInvalidAction = -1;

// Static information for a game. This will determine what algorithms are
// applicable. For example, minimax search is only applicable to two-player,
// zero-sum games with perfect information. (Though can be made applicable to
// games that are constant-sum.)
//
// The number of players is not considered part of this static game type,
// because this depends on the parameterization. See Game::NumPlayers.
struct GameType {
  // A short name with no spaces that uniquely identifies the game, e.g.
  // "msoccer". This is the key used to distinguish games.
  std::string short_name;

  // A long human-readable name, e.g. "Markov Soccer".
  std::string long_name;

  // Is the game one-player-at-a-time or do players act simultaneously?
  enum class Dynamics {
    kSimultaneous,  // Every player acts at each stage.
    kSequential,    // Turn-based games.
  };
  Dynamics dynamics;

  // Are there any chance nodes? If so, how is chance treated?
  // Either all possible chance outcomes are explicitly returned as
  // ChanceOutcomes(), and the result of ApplyAction() is deterministic. Or
  // just one ChanceOutcome is returned, and the result of ApplyAction() is
  // stochastic.
  enum class ChanceMode {
    kDeterministic,       // No chance nodes
    kExplicitStochastic,  // Has at least one chance node, all with
                          // deterministic ApplyAction()
    kSampledStochastic,   // At least one chance node with non-deterministic
                          // ApplyAction()
  };
  ChanceMode chance_mode;

  // The information type of the game.
  enum class Information {
    kOneShot,               // aka Normal-form games (single simultaneous turn).
    kPerfectInformation,    // All players know the state of the game.
    kImperfectInformation,  // Some information is hidden from some players.
  };
  Information information;

  // Whether the game has any constraints on the player utilities.
  enum class Utility {
    kZeroSum,      // Utilities of all players sum to 0
    kConstantSum,  // Utilities of all players sum to a constant
    kGeneralSum,   // Total utility of all players differs in different outcomes
    kIdentical,    // Every player gets an identical value (cooperative game).
  };
  Utility utility;

  // When are rewards handed out? Note that even if the game only specifies
  // utilities at terminal states, the default implementation of State::Rewards
  // should work for RL uses (giving 0 everywhere except terminal states).
  enum class RewardModel {
    kRewards,   // RL-style func r(s, a, s') via State::Rewards() call at s'.
    kTerminal,  // Games-style, only at terminals. Call (State::Returns()).
  };
  RewardModel reward_model;

  // How many players can play the game. If the number can vary, the actual
  // instantiation of the game should specify how many players there are.
  int max_num_players;
  int min_num_players;

  // Which type of information state representations are supported?
  // The information state is a perfect-recall state-of-the-game from the
  // perspective of one player.
  bool provides_information_state;
  bool provides_information_state_as_normalized_vector;

  // Which type of observation representations are supported?
  // The observation is some subset of the information state with the property
  // that remembering all the player's observations and actions is sufficient
  // to reconstruct the information state.
  bool provides_observation;
  bool provides_observation_as_normalized_vector;

  std::map<std::string, GameParameter> parameter_specification;
  bool ContainsRequiredParameters() const;
};

enum class StateType {
  kTerminal,  // If the state is terminal.
  kChance,    // If the player to act equals kChanceId.
  kDecision,  // If a player other than kChanceId is acting.
};
std::ostream& operator<<(std::ostream& os, const StateType& type);

std::ostream& operator<<(std::ostream& stream, GameType::Dynamics value);
std::ostream& operator<<(std::ostream& stream, GameType::ChanceMode value);
std::ostream& operator<<(std::ostream& stream, GameType::Information value);
std::ostream& operator<<(std::ostream& stream, GameType::Utility value);

// The probability of taking each possible action in a particular info state.
using ActionsAndProbs = std::vector<std::pair<Action, double>>;

// Forward declaration needed for the backpointer within State.
class Game;

// An abstract class that represents a state of the game.
class State {
 public:
  virtual ~State() = default;

  // Derived classes must call one of these constructors. Note that a state must
  // be passed a pointer to the game which created it. Some methods in some
  // games rely on this and so it must correspond to a valid game object.
  // The easiest way to ensure this is to use Game::NewInitialState to create
  // new states, which will pass a pointer to the parent game object. Also,
  // since this shared pointer to the parent is required, Game objects cannot
  // be used as value types and should always be created via a shared pointer.
  // See the documentation of the Game object for further details.
  State(std::shared_ptr<const Game> game);
  State(const State&) = default;

  // Returns current player. Player numbers start from 0.
  // Negative numbers are for chance (-1) or simultaneous (-2).
  // kTerminalState should be returned on a TerminalNode().
  virtual Player CurrentPlayer() const = 0;

  // Change the state of the game by applying the specified action in turn-based
  // games. This function encodes the logic of the game rules. Returns true
  // on success. In simultaneous games, returns false (ApplyActions should be
  // used in that case.)
  //
  // In the case of chance nodes, the behavior of this function depends on
  // GameType::chance_mode. If kExplicit, then the outcome should be
  // directly applied. If kSampled, then a dummy outcome is passed and the
  // sampling of and outcome should be done in this function and then applied.
  //
  // Games should implement DoApplyAction.
  virtual void ApplyAction(Action action_id) {
    // history_ needs to be modified *after* DoApplyAction which could
    // be using it.
    DoApplyAction(action_id);
    history_.push_back(action_id);
  }

  // `LegalActions(Player player)` is valid for all nodes in all games,
  // returning an empty list for players who don't act at this state. The
  // actions should be returned in ascending order.
  //
  // This default implementation is fine for turn-based games, but should
  // be overridden by simultaneous-move games.
  //
  // Since games mostly override LegalActions(), this method will not be visible
  // in derived classes unless a using directive is added.
  virtual std::vector<Action> LegalActions(Player player) const {
    if (!IsTerminal() && player == CurrentPlayer()) {
      return IsChanceNode() ? LegalChanceOutcomes() : LegalActions();
    } else {
      return {};
    }
  }

  // `LegalActions()` returns the actions for the current player (including at
  // chance nodes). All games should implement this function.
  // For any action `a`, it must hold that 0 <= `a` < NumDistinctActions().
  // The actions should be returned in ascending order.
  // If the state is non-terminal, there must be at least one legal action.
  //
  // In simultaneous-move games, the abstract base class implements it in
  // terms of LegalActions(player) and LegalChanceOutcomes(), and so derived
  // classes only need to implement `LegalActions(Player player)`.
  // This will result in LegalActions() being hidden unless a using directive
  // is added.
  virtual std::vector<Action> LegalActions() const = 0;

  // Returns a vector of length `game.NumDistinctActions()` containing 1 for
  // legal actions and 0 for illegal actions.
  std::vector<int> LegalActionsMask(Player player) const {
    std::vector<int> mask(num_distinct_actions_, 0);
    std::vector<Action> legal_actions = LegalActions(player);

    for (auto const& value : legal_actions) {
      mask[value] = 1;
    }
    return mask;
  }

  // Convenience function for turn-based games.
  std::vector<int> LegalActionsMask() const {
    return LegalActionsMask(CurrentPlayer());
  }

  // Returns a string representation of the specified action for the player.
  // The representation may depend on the current state of the game, e.g.
  // for chess the string "Nf3" would correspond to different starting squares
  // in different states (and hence probably different action ids).
  // This method will format chance outcomes if player == kChancePlayer
  virtual std::string ActionToString(Player player, Action action_id) const = 0;

  // Returns a string representation of the state. This has no particular
  // semantics and is targeting debugging code.
  virtual std::string ToString() const = 0;

  // Is this a terminal state? (i.e. has the game ended?)
  virtual bool IsTerminal() const = 0;

  // Returns reward from the most recent state transition (s, a, s') for all
  // players. This is provided so that RL-style games with intermediate rewards
  // (along the episode, rather than just one value at the end) can be properly
  // implemented. The default is to return 0 except at terminal states, where
  // the terminal returns are returned.
  //
  // Note 1: should not be called at chance nodes (undefined and crashes).
  // Note 2: This must agree with Returns(). That is, for any state S_t,
  //         Returns(St) = Sum(Rewards(S_0), Rewards(S_1)... Rewards(S_t)).
  //         The default implementation is only correct for games that only
  //         have a final reward. Games with intermediate rewards must override
  //         both this method and Returns().
  virtual std::vector<double> Rewards() const {
    if (IsTerminal()) {
      return Returns();
    } else {
      SPIEL_CHECK_FALSE(IsChanceNode());
      return std::vector<double>(num_players_, 0.0);
    }
  }

  // Returns sums of all rewards for each player up to the current state.
  // For games that only have a final reward, it should be 0 for all
  // non-terminal states, and the terminal utility for the final state.
  virtual std::vector<double> Returns() const = 0;

  // Returns Reward for one player (see above for definition). If Rewards for
  // multiple players are required it is more efficient to use Rewards() above.
  virtual double PlayerReward(Player player) const {
    auto rewards = Rewards();
    SPIEL_CHECK_LT(player, rewards.size());
    return rewards[player];
  }

  // Returns Return for one player (see above for definition). If Returns for
  // multiple players are required it is more efficient to use Returns() above.
  virtual double PlayerReturn(Player player) const {
    auto returns = Returns();
    SPIEL_CHECK_LT(player, returns.size());
    return returns[player];
  }

  // Is this state a chance node? Chance nodes are "states" whose actions
  // represent stochastic outcomes. "Chance" or "Nature" is thought of as a
  // player with a fixed (randomized) policy.
  bool IsChanceNode() const { return CurrentPlayer() == kChancePlayerId; }

  // Is this state a node that requires simultaneous action choices from more
  // than one player? If this is ever true, then the game should be marked as
  // a simultaneous game.
  bool IsSimultaneousNode() const {
    return CurrentPlayer() == kSimultaneousPlayerId;
  }

  // A string representation for the history. There should be a one to one
  // mapping between an history (i.e. a sequence of actions for all players,
  // including chance) and the `State` objects.
  virtual std::vector<Action> History() const { return history_; }

  virtual std::string HistoryString() const {
    return absl::StrJoin(history_, " ");
  }

  // For imperfect information games. Returns an identifier for the current
  // information state for the specified player.
  // Different ground states can yield the same information state for a player
  // when the only part of the state that differs is not observable by that
  // player (e.g. opponents' cards in Poker.)

  // Games that do not have imperfect information do not need to implement
  // these methods, but most algorithms intended for imperfect information
  // games will work on perfect information games provided the InformationState
  // is returned in a form they support. For example, InformationState()
  // could simply return the history for a perfect information game.

  // The InformationState must be returned at terminal states, since this is
  // required in some applications (e.g. final observation in an RL
  // environment).

  // The information state should be perfect-recall, i.e. if two states
  // have a different InformationState, then all successors of one must have
  // a different InformationState to all successors of the other.
  // For example, in tic-tac-toe, the current state of the board would not be
  // a perfect-recall representation, but the sequence of moves played would
  // be.

  // There are currently no use-case for calling this function with
  // 'kChancePlayerId'. Thus, games are expected to raise an error in that case.
  virtual std::string InformationState(Player player) const {
    SpielFatalError("InformationState is not implemented.");
  }

  // This function should raise an error on Terminal nodes, since the
  // CurrentPlayer() should be kTerminalPlayerId.
  virtual std::string InformationState() const {
    return InformationState(CurrentPlayer());
  }

  // Vector form, useful for neural-net function approximation approaches.
  // The size of the vector must match Game::InformationStateShape()
  // with values in lexicographic order. E.g. for 2x4x3, order would be:
  // (0,0,0), (0,0,1), (0,0,2), (0,1,0), ... , (1,3,2).
  // This function should resize the supplied vector if required.

  // There are currently no use-case for calling this function with
  // 'kChancePlayerId'. Thus, games are expected to raise an error in that case.
  virtual void InformationStateAsNormalizedVector(
      Player player, std::vector<double>* values) const {
    SpielFatalError("InformationStateAsNormalizedVector unimplemented!");
  }

  virtual std::vector<double> InformationStateAsNormalizedVector(
      Player player) const {
    std::vector<double> normalized_info_state;
    InformationStateAsNormalizedVector(player, &normalized_info_state);
    return normalized_info_state;
  }

  virtual std::vector<double> InformationStateAsNormalizedVector() const {
    std::vector<double> normalized_info_state;
    InformationStateAsNormalizedVector(CurrentPlayer(), &normalized_info_state);
    return normalized_info_state;
  }

  // We have functions for observations which are parallel to those for
  // information states. An observation should have the following properties:
  //  - It has at most the same information content as the information state
  //  - The complete history of observations and our actions over the
  //    course of the game is sufficient to reconstruct the information
  //    state.
  //
  // For example, the cards revealed and bets made since our previous move in
  // poker, or the current state of the board in chess.
  // Note that neither of these are valid information states, since the same
  // observation may arise from two different observation histories (i.e. they
  // are not perfect recall).
  virtual std::string Observation(Player player) const {
    SpielFatalError("Observation is not implemented.");
  }

  virtual std::string Observation() const {
    return Observation(CurrentPlayer());
  }

  virtual void ObservationAsNormalizedVector(
      Player player, std::vector<double>* values) const {
    SpielFatalError("ObservationAsNormalizedVector unimplemented!");
  }

  virtual std::vector<double> ObservationAsNormalizedVector(
      Player player) const {
    std::vector<double> normalized_observation;
    ObservationAsNormalizedVector(player, &normalized_observation);
    return normalized_observation;
  }

  virtual std::vector<double> ObservationAsNormalizedVector() const {
    std::vector<double> normalized_observation;
    ObservationAsNormalizedVector(CurrentPlayer(), &normalized_observation);
    return normalized_observation;
  }

  // Return a copy of this state.
  virtual std::unique_ptr<State> Clone() const = 0;

  // Creates the child from State corresponding to action.
  std::unique_ptr<State> Child(Action action) const {
    std::unique_ptr<State> child = Clone();
    child->ApplyAction(action);
    return child;
  }

  // Undoes the last action, which must be supplied. This is a fast method to
  // undo an action. It is only necessary for algorithms that need a fast undo
  // (e.g. minimax search).
  // One must call history_.pop_back() in the implementations.
  virtual void UndoAction(Player player, Action action) {
    SpielFatalError("UndoAction function is not overridden; not undoing.");
  }

  // Change the state of the game by applying the specified actions, one per
  // player, for simultaneous action games. This function encodes the logic of
  // the game rules. Element i of the vector is the action for player i.
  // Every player must submit a action; if one of the players has no actions at
  // this node, then kInvalidAction should be passed instead.
  //
  // Simulatenous games should implement DoApplyActions.
  void ApplyActions(const std::vector<Action>& actions) {
    // history_ needs to be modified *after* DoApplyActions which could
    // be using it.
    DoApplyActions(actions);
    history_.reserve(history_.size() + actions.size());
    history_.insert(history_.end(), actions.begin(), actions.end());
  }

  // The size of the action space. See `Game` for a full description.
  int NumDistinctActions() const { return num_distinct_actions_; }

  // Returns the number of players in this game.
  int NumPlayers() const { return num_players_; }

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetGame() const { return game_; }

  // Get the chance outcomes and their probabilities.
  //
  // Chance actions do not have a separate UID space from regular actions.
  //
  // Note: what is returned here depending on the game's chance_mode (in
  // its GameType):
  //   - Option 1. kExplicit. All chance node outcomes are returned along with
  //     their respective probabilities. Then State::ApplyAction(...) is
  //     deterministic.
  //   - Option 2. kSampled. Return a dummy single action here with probability
  //     1, and then State::ApplyAction(...) does the real sampling. In this
  //     case, the game has to maintain its own RNG.
  virtual ActionsAndProbs ChanceOutcomes() const {
    SpielFatalError("ChanceOutcomes unimplemented!");
  }

  // Lists the valid chance outcomes at the current state.
  // Derived classes may substitute this with a more efficient implementation.
  virtual std::vector<Action> LegalChanceOutcomes() const {
    ActionsAndProbs outcomes_with_probs = ChanceOutcomes();
    std::vector<Action> outcome_list;
    outcome_list.reserve(outcomes_with_probs.size());
    for (auto& pair : outcomes_with_probs) {
      outcome_list.push_back(pair.first);
    }
    return outcome_list;
  }

  // Returns the type of the state. Either Chance, Terminal, or Decision. See
  // StateType definition for definitions of the different types.
  StateType GetType() const;

 protected:
  // See ApplyAction.
  virtual void DoApplyAction(Action action_id) {
    SpielFatalError("DoApplyAction is not implemented.");
  }
  // See ApplyActions.
  virtual void DoApplyActions(const std::vector<Action>& actions) {
    SpielFatalError("DoApplyActions is not implemented.");
  }

  // Fields common to every game state.
  int num_distinct_actions_;
  int num_players_;
  std::vector<Action> history_;  // The list of actions leading to the state.

  // A pointer to the game that created this state.
  std::shared_ptr<const Game> game_;
};

// A class that refers to a particular game instantiation, for example
// Breakthrough(8x8).
//
// Important note: Game objects cannot be instantiated on the stack or via
// unique_ptr, because shared pointers to the game object must be sent down to
// the states that created them. So, they *must* be created via
// shared_ptr<const Game> or via the LoadGame methods.
class Game : public std::enable_shared_from_this<Game> {
 public:
  virtual ~Game() = default;

  // Maximum number of distinct actions in the game for any one player. This is
  // not the same as max number of legal actions in any state as distinct
  // actions are independent of the context (state), and often independent of
  // the player as well. So, for instance in Tic-Tac-Toe this value is 9, one
  // for each square. In games where pieces move, like e.g. Breakthrough, then
  // it would be 64*6*2, since from an 8x8 board a single piece could only ever
  // move to at most 6 places, and it can be a regular move or a capture move.
  // Note: chance node outcomes are not included in this count.
  // For example, this would correspond to the size of the policy net head
  // learning which move to play.
  virtual int NumDistinctActions() const = 0;

  // Returns a newly allocated initial state.
  virtual std::unique_ptr<State> NewInitialState() const = 0;

  // Maximum number of chance outcomes for each chance node.
  virtual int MaxChanceOutcomes() const { return 0; }

  // If the game is parametrizable, returns an object with the current parameter
  // values, including defaulted values. Returns empty parameters otherwise.
  GameParameters GetParameters() const {
    GameParameters params = game_parameters_;
    params.insert(defaulted_parameters_.begin(), defaulted_parameters_.end());
    return params;
  }

  // The number of players in this instantiation of the game.
  // Does not include the chance-player.
  virtual int NumPlayers() const = 0;

  // Utility range. These functions define the lower and upper bounds on the
  // values returned by State::PlayerReturn(Player player) over all valid player
  // numbers. This range should be as tight as possible; the intention is to
  // give some information to algorithms that require it, and so their
  // performance may suffer if the range is not tight. Loss/win/draw outcomes
  // are common among games and should use the standard values of {-1,0,1}.
  virtual double MinUtility() const = 0;
  virtual double MaxUtility() const = 0;

  // Return a clone of this game.
  virtual std::shared_ptr<const Game> Clone() const = 0;

  // Static information on the game type. This should match the information
  // provided when registering the game.
  const GameType& GetType() const { return game_type_; }

  // The total utility for all players, if this is a constant-sum-utility game.
  // Should return 0. if the game is zero-sum.
  virtual double UtilitySum() const {
    SpielFatalError("UtilitySum unimplemented.");
    return 0.;
  }

  // Describes the structure of the information state representation in a
  // tensor-like format. This is especially useful for experiments involving
  // reinforcement learning and neural networks. Note: the actual information is
  // returned in a 1-D vector by State::InformationStateAsNormalizedVector -
  // see the documentation of that function for details of the data layout.
  virtual std::vector<int> InformationStateNormalizedVectorShape() const {
    SpielFatalError("InformationStateNormalizedVectorShape unimplemented.");
  }

  // The size of (flat) vector needed for the information state tensor-like
  // format.
  int InformationStateNormalizedVectorSize() const {
    std::vector<int> shape = InformationStateNormalizedVectorShape();
    return shape.empty() ? 0
                         : std::accumulate(shape.begin(), shape.end(), 1,
                                           std::multiplies<double>());
  }

  // Describes the structure of the observation representation in a
  // tensor-like format. This is especially useful for experiments involving
  // reinforcement learning and neural networks. Note: the actual observation is
  // returned in a 1-D vector by State::ObservationAsNormalizedVector -
  // see the documentation of that function for details of the data layout.
  virtual std::vector<int> ObservationNormalizedVectorShape() const {
    SpielFatalError("ObservationNormalizedVectorShape unimplemented.");
  }

  // The size of (flat) vector needed for the observation tensor-like
  // format.
  int ObservationNormalizedVectorSize() const {
    std::vector<int> shape = ObservationNormalizedVectorShape();
    return shape.empty() ? 0
                         : std::accumulate(shape.begin(), shape.end(), 1,
                                           std::multiplies<double>());
  }

  // Serializes a state into a string.
  //
  // The default implementation writes out a sequence of actions, one per line,
  // taken from the initial state. Note: this default serialization scheme will
  // not work games whose chance mode is kSampledStochastic, as there is
  // currently no general way to set the state's seed to ensure that it samples
  // the same chance event at chance nodes.
  //
  // If overridden, this must be the inverse of Game::DeserializeState.
  virtual std::string SerializeState(const State& state) const;

  // Returns a newly allocated state built from a string. Caller takes ownership
  // of the state.

  // Build a state from a string.
  //
  // The default implementation assumes a sequence of actions, one per line,
  // that is taken from the initial state.
  //
  // If this method is overridden, then it should be inverse of
  // Game::SerializeState (i.e. it should also be overridden).
  virtual std::unique_ptr<State> DeserializeState(const std::string& str) const;

  // Maximum length of any one game (in terms of number of decision nodes
  // visited in the game tree). For a simultaneous action game, this is the
  // maximum number of joint decisions. In a turn-based game, this is the
  // maximum number of individual decisions summed over all players. Outcomes
  // of chance nodes are not included in this length.
  virtual int MaxGameLength() const = 0;

  // A string representation of the game, which can be passed to LoadGame.
  std::string ToString() const;

 protected:
  Game(GameType game_type, GameParameters game_parameters)
      : game_type_(game_type), game_parameters_(game_parameters) {}

  // Access to game parameters. Returns the value provided by the user. If not:
  // - Defaults to the value stored as the default in
  // game_type.parameter_specification if the `default_value` is std::nullopt
  // - Returns `default_value` if provided.
  template <typename T>
  T ParameterValue(const std::string& key,
                   std::optional<T> default_value = std::nullopt) const;

  // The game type.
  GameType game_type_;

  // Any parameters supplied when constructing the game.
  GameParameters game_parameters_;

  // Track the parameters for which a default value has been used. This
  // enables us to report the actual value used for every parameter.
  mutable GameParameters defaulted_parameters_;
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_SPIEL_GAME(info, factory) \
  GameRegisterer CONCAT(game, __COUNTER__)(info, factory);

class GameRegisterer {
 public:
  using CreateFunc =
      std::function<std::shared_ptr<const Game>(const GameParameters& params)>;

  GameRegisterer(const GameType& game_type, CreateFunc creator);

  static std::shared_ptr<const Game> CreateByName(const std::string& short_name,
                                                  const GameParameters& params);

  static std::vector<std::string> RegisteredNames();
  static std::vector<GameType> RegisteredGames();
  static bool IsValidName(const std::string& short_name);
  static void RegisterGame(const GameType& game_type, CreateFunc creator);

 private:
  // Returns a "global" map of registrations (i.e. an object that lives from
  // initialization to the end of the program). Note that we do not just use
  // a static data member, as we want the map to be initialized before first
  // use.
  static std::map<std::string, std::pair<GameType, CreateFunc>>& factories() {
    static std::map<std::string, std::pair<GameType, CreateFunc>> impl;
    return impl;
  }
};

// Returns true if the game is registered, false otherwise.
bool IsGameRegistered(const std::string& short_name);

// Returns a list of registered games' short names.
std::vector<std::string> RegisteredGames();

// Returns a list of registered game types.
std::vector<GameType> RegisteredGameTypes();

// Returns a new game object from the specified string, which is the short
// name plus optional parameters, e.g. "go(komi=4.5,board_size=19)"
std::shared_ptr<const Game> LoadGame(const std::string& game_string);

// Returns a new game object with the specified parameters.
std::shared_ptr<const Game> LoadGame(const std::string& short_name,
                                     const GameParameters& params);

// Returns a new game object with the specified parameters; reads the name
// of the game from the 'name' parameter (which is not passed to the game
// implementation).
std::shared_ptr<const Game> LoadGame(GameParameters params);

// Used to sample a policy. Can also sample from chance outcomes.
// Probabilities of the actions must sum to 1.
// The parameter z should be a sample from a uniform distribution on the range
// [0, 1).
Action SampleChanceOutcome(const ActionsAndProbs& outcomes, double z);

// Serialize the game and the state into one self-contained string that can
// be reloaded via open_spiel::DeserializeGameAndState.
//
// The format of the string is the following (contains three sections,
// marked by single-line headers in square brackets with specific keywords),
// see below. The meta section contains general info. The game string is
// parsed using LoadGame(string) and the state section is parsed using
// Game::DeserializeState.
//
// Example file contents:
//
//   # Comments are ok, but hash '#' must be first chatacter in the line.
//   # Blank lines and lines that start with hash '#' are ignored
//   [Meta]
//   Version: <version>
//
//   [Game]
//   <serialized game string; may take up several lines>
//
//   [State]
//   <serialized state; may take up several lines>
std::string SerializeGameAndState(const Game& game, const State& state);

// A general deserialization which reconstructs both the game and the state,
// which have been saved using the default simple implementation in
// SerializeGameAndState. The game must be registered so that it is loadable via
// LoadGame.
//
// The state string must have a specific format. See
// Game::SerializeGameAndState for a description of the saved format.
//
// Note: This serialization scheme will not work for games whose chance mode is
// kSampledStochastic, as there is currently no general way to set the state's
// seed.
std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
DeserializeGameAndState(const std::string& serialized_state);

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_SPIEL_H_
