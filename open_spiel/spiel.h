// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_SPIEL_H_
#define OPEN_SPIEL_SPIEL_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/bit_gen_ref.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

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
    kSimultaneous,           // In some or all nodes every player acts.
    kSequential,             // Turn-based games.
    // Mean field game. In particular, this adds mean field nodes. Support for
    // mean field games is experimental. See details in games/mfg/README.md.
    kMeanField,              // Is a Mean Field Game
  };
  Dynamics dynamics;

  // Are there any chance nodes? If so, how is chance treated?
  // Either all possible chance outcomes are explicitly returned as
  // ChanceOutcomes(), and the result of ApplyAction() is deterministic. Or
  // just one ChanceOutcome is returned, and the result of ApplyAction() is
  // stochastic. If in doubt, it is better to implement stochastic games with
  // kExplicitStochastic, as this makes more information available to any
  // learning algorithms you choose to use (i.e. the whole chance outcome
  // distribution is visible to the algorithm, rather than just the sampled
  // outcome). For more discussion of this field, see the github issue:
  // https://github.com/deepmind/open_spiel/issues/792.
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
  bool provides_information_state_string;
  bool provides_information_state_tensor;

  // Which type of observation representations are supported?
  // The observation is some subset of the information state with the property
  // that remembering all the player's observations and actions is sufficient
  // to reconstruct the information state.
  bool provides_observation_string;
  bool provides_observation_tensor;

  GameParameters parameter_specification;
  bool ContainsRequiredParameters() const;

  // A number of optional values that have defaults, whose values can be
  // overridden in each game.

  // Can the game be loaded with no parameters? It is strongly recommended that
  // games be loadable with default arguments.
  bool default_loadable = true;

  // Can we factorize observations into public and private parts?
  // This is similar to observation fields before, but adds additional
  // distinction between public and private observations.
  bool provides_factored_observation_string = false;

  bool provides_information_state() const {
    return provides_information_state_tensor
        || provides_information_state_string;
  }
  bool provides_observation() const {
    return provides_observation_tensor
        || provides_observation_string;
  }
};

// Information about a concrete Game instantiation.
// This information may depend on the game parameters, and hence cannot
// be part of `GameType`.
struct GameInfo {
  // The size of the action space. See `Game` for a full description.
  int num_distinct_actions;

  // Maximum number of distinct chance outcomes for chance nodes in the game.
  int max_chance_outcomes;

  // The number of players in this instantiation of the game.
  // Does not include the chance-player.
  int num_players;

  // Utility range. These functions define the lower and upper bounds on the
  // values returned by State::PlayerReturn(Player player) over all valid player
  // numbers. This range should be as tight as possible; the intention is to
  // give some information to algorithms that require it, and so their
  // performance may suffer if the range is not tight. Loss/draw/win outcomes
  // are common among games and should use the standard values of {-1,0,1}.
  double min_utility;
  double max_utility;

  // The total utility for all players, if this is a constant-sum-utility game.
  // Should be zero if the game is zero-sum.
  double utility_sum;

  // The maximum number of player decisions in a game. Does not include chance
  // events. For a simultaneous action game, this is the maximum number of joint
  // decisions. In a turn-based game, this is the maximum number of individual
  // decisions summed over all players.
  int max_game_length;
};

std::ostream& operator<<(std::ostream& os, const StateType& type);

std::ostream& operator<<(std::ostream& stream, GameType::Dynamics value);
std::ostream& operator<<(std::ostream& stream, GameType::ChanceMode value);
std::ostream& operator<<(std::ostream& stream, GameType::Information value);
std::ostream& operator<<(std::ostream& stream, GameType::Utility value);
std::ostream& operator<<(std::ostream& stream, GameType::RewardModel value);

// The probability of taking each possible action in a particular info state.
using ActionsAndProbs = std::vector<std::pair<Action, double>>;

// We alias this here as we can't import state_distribution.h or we'd have a
// circular dependency.
using HistoryDistribution =
    std::pair<std::vector<std::unique_ptr<State>>, std::vector<double>>;

// Forward declarations.
class Game;
class Observer;

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
  // kTerminalPlayerId should be returned on a TerminalNode().
  virtual Player CurrentPlayer() const = 0;

  // Change the state of the game by applying the specified action in turn-based
  // games or in non-simultaneous nodes of simultaneous move games.
  // This function encodes the logic of the game rules.
  //
  // In the case of chance nodes, the behavior of this function depends on
  // GameType::chance_mode. If kExplicit, then the outcome should be
  // directly applied. If kSampled, then a dummy outcome is passed and the
  // sampling of and outcome should be done in this function and then applied.
  //
  // Games should implement DoApplyAction.
  virtual void ApplyAction(Action action_id);

  // Helper versions of ApplyAction that first does a legality check.
  virtual void ApplyActionWithLegalityCheck(Action action_id);

  // `LegalActions(Player player)` is valid for all nodes in all games,
  // returning an empty list for players who don't act at this state. The
  // actions should be returned in ascending order.
  //
  // This default implementation is fine for turn-based games, but should
  // be overridden by simultaneous-move games. At least one player should have a
  // legal action or the game should be terminal.
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
  // At a player node, all returned actions should be in
  // [0, NumDistinctActions()). For a chance node, they should all be in
  // [0, MaxChanceOutcomes()).
  // The actions should be returned in ascending order.
  // If the state is non-terminal (and not a mean field node), there must be at
  // least one legal action.
  //
  // In simultaneous-move games, the abstract base class implements it in
  // terms of LegalActions(player) and LegalChanceOutcomes(), and so derived
  // classes only need to implement `LegalActions(Player player)`.
  // This will result in LegalActions() being hidden unless a using directive
  // is added.
  virtual std::vector<Action> LegalActions() const = 0;

  // Returns a vector containing 1 for legal actions and 0 for illegal actions.
  // The length is `game.NumDistinctActions()` for player nodes, and
  // `game.MaxChanceOutcomes()` for chance nodes.
  std::vector<int> LegalActionsMask(Player player) const;

  // Convenience function for turn-based games.
  std::vector<int> LegalActionsMask() const {
    return LegalActionsMask(CurrentPlayer());
  }

  // Returns a string representation of the specified action for the player.
  // The representation may depend on the current state of the game, e.g.
  // for chess the string "Nf3" would correspond to different starting squares
  // in different states (and hence probably different action ids).
  // This method will format chance outcomes if player == kChancePlayerId
  virtual std::string ActionToString(Player player, Action action_id) const = 0;
  std::string ActionToString(Action action_id) const {
    return ActionToString(CurrentPlayer(), action_id);
  }

  // Reverses the mapping done by ActionToString.
  // Note: This currently just loops over all legal actions, converts them into
  // a string, and checks equality, so it can be very slow.
  virtual Action StringToAction(Player player,
                                const std::string& action_str) const;
  Action StringToAction(const std::string& action_str) const {
    return StringToAction(CurrentPlayer(), action_str);
  }

  // Returns a string representation of the state. Also used as in the default
  // implementation of operator==.
  virtual std::string ToString() const = 0;

  // Returns true if these states are equal, false otherwise. Two states are
  // equal if they are the same world state; the interpretation might differ
  // across games. For instance, in an imperfect information game, the full
  // history might be relevant for distinguishing states whereas it might not be
  // relevant for single-player games or perfect information games such as
  // Tic-Tac-Toe, where only the current board state is necessary.
  virtual bool operator==(const State& other) const {
    return ToString() == other.ToString();
  }

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
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, returns.size());
    return returns[player];
  }

  // Is this state a chance node? Chance nodes are "states" whose actions
  // represent stochastic outcomes. "Chance" or "Nature" is thought of as a
  // player with a fixed (randomized) policy.
  virtual bool IsChanceNode() const {
    return CurrentPlayer() == kChancePlayerId;
  }

  // Is this a mean field node? In that case, no action should be performed, but
  // instead the global state distribution should be updated with
  // UpdateDistribution(). See more details in games/mfg/README.md.
  virtual bool IsMeanFieldNode() const {
    return CurrentPlayer() == kMeanFieldPlayerId;
  }

  // Is this state a player node, with a single player acting?
  virtual bool IsPlayerNode() const { return CurrentPlayer() >= 0; }

  // Is this state a node that requires simultaneous action choices from more
  // than one player? If this is ever true, then the game should be marked as
  // a simultaneous game.
  bool IsSimultaneousNode() const {
    return CurrentPlayer() == kSimultaneousPlayerId;
  }

  // Is the specified player acting at this state?
  bool IsPlayerActing(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, NumPlayers());
    return CurrentPlayer() == player || IsSimultaneousNode();
  }

  // We store (player, action) pairs in the history.
  struct PlayerAction {
    Player player;
    Action action;
    bool operator==(const PlayerAction&) const;
  };

  // For backward-compatibility reasons, this is the history of actions only.
  // To get the (player, action) pairs, use `FullHistory` instead.
  std::vector<Action> History() const {
    std::vector<Action> history;
    history.reserve(history_.size());
    for (auto& h : history_) history.push_back(h.action);
    return history;
  }

  // The full (player, action) history.
  const std::vector<PlayerAction>& FullHistory() const { return history_; }

  // A string representation for the history. There should be a one to one
  // mapping between histories (i.e. sequences of actions for all players,
  // including chance) and the `State` objects.
  std::string HistoryString() const { return absl::StrJoin(History(), ", "); }

  // Return how many moves have been done so far in the game.
  // When players make simultaneous moves, this counts only as a one move.
  // Chance transitions count also as one move.
  // Note that game transformations are not required to preserve the move
  // number in the transformed game.
  int MoveNumber() const { return move_number_; }

  // Is this a first state in the game, i.e. the initial state (root node)?
  bool IsInitialState() const { return history_.empty(); }

  // For imperfect information games. Returns an identifier for the current
  // information state for the specified player.
  // Different ground states can yield the same information state for a player
  // when the only part of the state that differs is not observable by that
  // player (e.g. opponents' cards in Poker.)
  //
  // The identifiers must be unique across all players.
  // This allows an algorithm to maintain a single table of identifiers
  // instead of maintaining a table per player to avoid name collisions.
  //
  // A simple way to do so is for example, in a card game, if both players can
  // hold the card Jack, the identifier can contain player identification as
  // well, like P1Jack and P2Jack. However prefixing by player number is not
  // a requirement. The only thing that is necessary is that it is unambiguous
  // who is the observer.
  //
  // Games that do not have imperfect information do not need to implement
  // these methods, but most algorithms intended for imperfect information
  // games will work on perfect information games provided the InformationState
  // is returned in a form they support. For example, InformationState()
  // could simply return the history for a perfect information game.
  //
  // A valid InformationStateString must be returned at terminal states, since
  // this is required in some applications (e.g. final observation in an RL
  // environment).
  //
  // The information state should be perfect-recall, i.e. if two states
  // have a different InformationState, then all successors of one must have
  // a different InformationState to all successors of the other.
  // For example, in tic-tac-toe, the current state of the board would not be
  // a perfect-recall representation, but the sequence of moves played would
  // be.
  //
  // If you implement both InformationState and Observation, the two must be
  // consistent for all the players (even the non-acting player(s)).
  // By consistency we mean that when you maintain an Action-Observation
  // history (AOH) for different ground states, the (in)equality of two AOHs
  // implies the (in)equality of two InformationStates. In other words, AOH is a
  // factored representation of InformationState.
  //
  // For details, see Section 3.1 of https://arxiv.org/abs/1908.09453
  // or Section 2.1 of https://arxiv.org/abs/1906.11110

  // There are currently no use-case for calling this function with
  // `kChancePlayerId`. Thus, games are expected to raise an error in those
  // cases using (and it's tested in api_test.py). Use this:
  //   SPIEL_CHECK_GE(player, 0);
  //   SPIEL_CHECK_LT(player, num_players_);
  virtual std::string InformationStateString(Player player) const {
    SpielFatalError("InformationStateString is not implemented.");
  }
  std::string InformationStateString() const {
    return InformationStateString(CurrentPlayer());
  }

  // Vector form, useful for neural-net function approximation approaches.
  // The size of the vector must match Game::InformationStateShape()
  // with values in lexicographic order. E.g. for 2x4x3, order would be:
  // (0,0,0), (0,0,1), (0,0,2), (0,1,0), ... , (1,3,2).
  // This function should resize the supplied vector if required.
  //
  // A valid InformationStateTensor must be returned at terminal states, since
  // this is required in some applications (e.g. final observation in an RL
  // environment).
  //
  // There are currently no use-case for calling this function with
  // `kChancePlayerId`. Thus, games are expected to raise an error in those
  // cases.
  //
  // Implementations should start with (and it's tested in api_test.py):
  //   SPIEL_CHECK_GE(player, 0);
  //   SPIEL_CHECK_LT(player, num_players_);
  virtual void InformationStateTensor(Player player,
                                      absl::Span<float> values) const {
    SpielFatalError("InformationStateTensor unimplemented!");
  }
  std::vector<float> InformationStateTensor(Player player) const;
  std::vector<float> InformationStateTensor() const {
    return InformationStateTensor(CurrentPlayer());
  }
  virtual void InformationStateTensor(Player player,
                                      std::vector<float>* values) const;

  // We have functions for observations which are parallel to those for
  // information states. An observation should have the following properties:
  //  - It has at most the same information content as the information state
  //  - The complete history of observations and our actions over the
  //    course of the game is sufficient to reconstruct the information
  //    state for any players at any point in the game.
  //
  // For example, an observation is the cards revealed and bets made in Poker,
  // or the current state of the board in Chess.
  // Note that neither of these are valid information states, since the same
  // observation may arise from two different observation histories (i.e. they
  // are not perfect recall).
  //
  // Observations should cover all observations: a combination of both public
  // and private observations. They are not factored into these individual
  // constituent parts.
  //
  // A valid observation must be returned at terminal states, since this is
  // required in some applications (e.g. final observation in an RL
  // environment).
  //
  // Implementations should start with (and it's tested in api_test.py):
  //   SPIEL_CHECK_GE(player, 0);
  //   SPIEL_CHECK_LT(player, num_players_);
  virtual std::string ObservationString(Player player) const {
    SpielFatalError("ObservationString is not implemented.");
  }
  std::string ObservationString() const {
    return ObservationString(CurrentPlayer());
  }

  // Returns the view of the game, preferably from `player`'s perspective.
  //
  // Implementations should start with (and it's tested in api_test.py):
  //   SPIEL_CHECK_GE(player, 0);
  //   SPIEL_CHECK_LT(player, num_players_);
  virtual void ObservationTensor(Player player,
                                 absl::Span<float> values) const {
    SpielFatalError("ObservationTensor unimplemented!");
  }
  std::vector<float> ObservationTensor(Player player) const;
  std::vector<float> ObservationTensor() const {
    return ObservationTensor(CurrentPlayer());
  }
  void ObservationTensor(Player player, std::vector<float>* values) const;

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
  // One must call history_.pop_back() and --move_number_ in the implementations
  // (and do these appropriately especially in simultaneous games).
  virtual void UndoAction(Player player, Action action) {
    SpielFatalError("UndoAction function is not overridden; not undoing.");
  }

  // Change the state of the game by applying the specified actions, one per
  // player, for simultaneous action games. This function encodes the logic of
  // the game rules. Element i of the vector is the action for player i.
  //
  // Every player must submit a action. If some of the players have no legal
  // actions at this node, then 0 should be passed instead.
  //
  // Simultaneous games should implement DoApplyActions.
  void ApplyActions(const std::vector<Action>& actions);

  // A helper version of ApplyActions that first does legality checks.
  void ApplyActionsWithLegalityChecks(const std::vector<Action>& actions);


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

  // Returns the type of the state. Either Chance, Terminal, MeanField or
  // Decision. See StateType definition for definitions of the different types.
  StateType GetType() const;

  // Serializes a state into a string.
  //
  // The default implementation writes out a sequence of actions, one per line,
  // taken from the initial state. Note: this default serialization scheme will
  // not work games whose chance mode is kSampledStochastic, as there is
  // currently no general way to set the state's seed to ensure that it samples
  // the same chance event at chance nodes.
  //
  // If overridden, this must be the inverse of Game::DeserializeState.
  virtual std::string Serialize() const;

  // Resamples a new history from the information state from player_id's view.
  // This resamples a private for the other players, but holds player_id's
  // privates constant, and the public information constant.
  // The privates are sampled uniformly at each chance node. For games with
  // partially-revealed actions that require some policy, we sample uniformly
  // from the list of actions that are consistent with what player_id observed.
  // For rng, we need something that returns a double in [0, 1). This value will
  // be interpreted as a cumulative distribution function, and will be used to
  // sample from the legal chance actions. A good choice would be
  // absl/std::uniform_real_distribution<double>(0., 1.).
  virtual std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const {
    SpielFatalError("ResampleFromInfostate() not implemented.");
  }

  // Returns a vector of states & probabilities that are consistent with the
  // infostate from the view of the current player. By default, this is not
  // implemented and returns an empty list. This doesn't make any attempt to
  // correct for the opponent's policy in the probabilities, and so this is
  // wrong for any state that's not the first non-chance node.
  virtual std::unique_ptr<HistoryDistribution>
  GetHistoriesConsistentWithInfostate(int player_id) const {
    return {};
  }

  virtual std::unique_ptr<HistoryDistribution>
  GetHistoriesConsistentWithInfostate() const {
    return GetHistoriesConsistentWithInfostate(CurrentPlayer());
  }

  // Returns a vector of all actions that are consistent with the information
  // revealed by taking action. E.g. in Poker, this does nothing but return the
  // current action as poker only has public actions. In a game like Battleship,
  // where the placement phase is hidden, this would return all possible
  // placements.
  virtual std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const {
    SpielFatalError(
        "ActionsConsistentWithInformationFrom has not been implemented.");
    return {};
  }

  // These functions only apply to mean field games.
  // Mean field game support in open_spiel is experimental, and these functions
  // are subject to change.

  // At the current mean field node, the support of the state distribution that
  // needs to be updated. States are identified by their corresponding string
  // representation. In multi-population mean field nodes, the support will
  // typically include states for all the populations.
  // This should only be called when when CurrentPlayer() == kMeanFieldPlayerId.
  // This can return an empty list in case the distribution is not needed at
  // this time.
  virtual std::vector<std::string> DistributionSupport() {
    SpielFatalError("DistributionSupport has not been implemented");
  }
  // Update the state distribution. `distribution[i]` must correspond to
  // `DistributionSupport()[i]`. After this is called, the state will be of
  // Chance type.
  // This should only be called when when CurrentPlayer() == kMeanFieldPlayerId.
  virtual void UpdateDistribution(const std::vector<double>& distribution) {
    SpielFatalError("UpdateDistribution has not been implemented");
  }

  // Only makes sense for mean field games. This is the population a state
  // belongs to. It returns 0 by default, so multi-population mean field games
  // should override this function.
  virtual int MeanFieldPopulation() const;

 protected:
  // See ApplyAction.
  virtual void DoApplyAction(Action action_id) {
    SpielFatalError("DoApplyAction is not implemented.");
  }
  // See ApplyActions.
  virtual void DoApplyActions(const std::vector<Action>& actions) {
    SpielFatalError("DoApplyActions is not implemented.");
  }

  // The game that created this state, plus some static information about it,
  // cached here for efficient access.
  const std::shared_ptr<const Game> game_;
  const int num_distinct_actions_;
  const int num_players_;

  // Information that changes over the course of the game.
  std::vector<PlayerAction> history_;
  int move_number_;
};

std::ostream& operator<<(std::ostream& stream, const State& state);

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
  Game(const Game&) = delete;
  Game& operator=(const Game&) = delete;

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
  virtual std::unique_ptr<State> NewInitialState(const std::string& str) const {
    SpielFatalError("NewInitialState from string is not implemented.");
  }

  // Returns newly allocated initial states. In most cases, this will be a
  // single state.
  // Games with multi-population mean field dynamics have multiple initial
  // states, one per population. In that case, N initial states will be
  // returned, from population 0 to population N-1 (where N is the number of
  // populations, which is equal to the number of players).
  virtual std::vector<std::unique_ptr<State>> NewInitialStates() const;

  // Maximum number of distinct chance outcomes for chance nodes in the game.
  virtual int MaxChanceOutcomes() const { return 0; }

  // If the game is parameterizable, returns an object with the current
  // parameter values, including defaulted values. Returns empty parameters
  // otherwise.
  GameParameters GetParameters() const {
    absl::MutexLock lock(&mutex_defaulted_parameters_);
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
  // performance may suffer if the range is not tight. Loss/draw/win outcomes
  // are common among games and should use the standard values of {-1,0,1}.
  virtual double MinUtility() const = 0;
  virtual double MaxUtility() const = 0;

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
  // returned in a 1-D vector by State::InformationStateTensor -
  // see the documentation of that function for details of the data layout.
  virtual std::vector<int> InformationStateTensorShape() const {
    SpielFatalError("InformationStateTensorShape unimplemented.");
  }
  virtual TensorLayout InformationStateTensorLayout() const {
    return TensorLayout::kCHW;
  }

  // The size of the (flat) vector needed for the information state tensor-like
  // format.
  int InformationStateTensorSize() const {
    std::vector<int> shape = InformationStateTensorShape();
    return shape.empty() ? 0
                         : absl::c_accumulate(shape, 1, std::multiplies<int>());
  }

  // Describes the structure of the observation representation in a
  // tensor-like format. This is especially useful for experiments involving
  // reinforcement learning and neural networks. Note: the actual observation is
  // returned in a 1-D vector by State::ObservationTensor -
  // see the documentation of that function for details of the data layout.
  virtual std::vector<int> ObservationTensorShape() const {
    SpielFatalError("ObservationTensorShape unimplemented.");
  }
  virtual TensorLayout ObservationTensorLayout() const {
    return TensorLayout::kCHW;
  }

  // The size of the (flat) vector needed for the observation tensor-like
  // format.
  int ObservationTensorSize() const {
    std::vector<int> shape = ObservationTensorShape();
    return shape.empty() ? 0
                         : absl::c_accumulate(shape, 1, std::multiplies<int>());
  }

  // Describes the structure of the policy representation in a
  // tensor-like format. This is especially useful for experiments involving
  // reinforcement learning and neural networks. Note: the actual policy is
  // expected to be in the shape of a 1-D vector.
  virtual std::vector<int> PolicyTensorShape() const {
    return {NumDistinctActions()};
  }

  // Returns a newly allocated state built from a string. Caller takes ownership
  // of the state.
  //
  // The default implementation assumes a sequence of actions, one per line,
  // that is taken from the initial state.
  //
  // If this method is overridden, then it should be the inverse of
  // State::Serialize (i.e. that method should also be overridden).
  virtual std::unique_ptr<State> DeserializeState(const std::string& str) const;

  // The maximum length of any one game (in terms of number of decision nodes
  // visited in the game tree). For a simultaneous action game, this is the
  // maximum number of joint decisions. In a turn-based game, this is the
  // maximum number of individual decisions summed over all players. Outcomes
  // of chance nodes are not included in this length.
  virtual int MaxGameLength() const = 0;

  // The maximum number of chance nodes occurring in any history of the game.
  // This is typically something like the number of times dice are rolled.
  virtual int MaxChanceNodesInHistory() const {
    if (GetType().chance_mode == GameType::ChanceMode::kDeterministic) {
      return 0;
    }
    SpielFatalError("MaxChanceNodesInHistory() is not implemented");
  }

  // The maximum number of moves in the game. The value State::MoveNumber()
  // must never be higher than this value.
  virtual int MaxMoveNumber() const {
    return MaxGameLength() + MaxChanceNodesInHistory();
  }

  // The maximum length of any history in the game.
  // The value State::History().size() must never be higher than this value.
  virtual int MaxHistoryLength() const {
    if (GetType().dynamics == GameType::Dynamics::kSimultaneous) {
      // The history of simultaneous move games is flattened, so count number
      // of actions of each player.
      return MaxGameLength() * NumPlayers() + MaxChanceNodesInHistory();
    }
    if (GetType().dynamics == GameType::Dynamics::kSequential) {
      return MaxGameLength() + MaxChanceNodesInHistory();
    }
    SpielFatalError("Unknown game dynamics.");
  }

  // A string representation of the game, which can be passed to
  // DeserializeGame. The difference with Game::ToString is that it also
  // serializes internal RNG state used with sampled stochastic game
  // implementations.
  std::string Serialize() const;

  // A string representation of the game, which can be passed to LoadGame.
  std::string ToString() const;

  // Returns true if these games are equal, false otherwise.
  virtual bool operator==(const Game& other) const {
    return ToString() == other.ToString();
  }

  // Get and set game's internal RNG state for de/serialization purposes. These
  // two methods only need to be overridden by sampled stochastic games that
  // need to hold an RNG state. Note that stateful game implementations are
  // discouraged in general.
  virtual std::string GetRNGState() const {
    SpielFatalError("GetRNGState unimplemented.");
  }
  // SetRNGState is const despite the fact that it changes game's internal
  // state. Sampled stochastic games need to be explicit about mutability of the
  // RNG, i.e. have to use the mutable keyword.
  virtual void SetRNGState(const std::string& rng_state) const {
    SpielFatalError("SetRNGState unimplemented.");
  }

  // Returns an Observer, used to obtain observations of the game state.
  // The observations are created according to requested observation type.
  // Games can include additional observation fields when requested by
  // `params`.
  // See `observer.h` for further information.
  virtual std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;

  // Returns a string representation of the specified action for the player,
  // independent of the state.
  virtual std::string ActionToString(Player player, Action action_id) const {
    return absl::StrCat("Action(id=", action_id, ", player=", player, ")");
  }

  // Returns an observer that was registered, based on its name.
  std::shared_ptr<Observer> MakeRegisteredObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const;
  // Returns an observer that uses the observation or informationstate tensor
  // or string as defined directly on the state.
  std::shared_ptr<Observer> MakeBuiltInObserver(
      absl::optional<IIGObservationType> iig_obs_type) const;

  // Public member functions below only apply to games with mean field dynamics.

  // Creates a new initial state for the given population (which must be in [0,
  // NumPlayers())). This must be implemented for multi-population mean field
  // games.
  virtual std::unique_ptr<State> NewInitialStateForPopulation(
      int population) const {
    SpielFatalError("NewInitialStateForPopulation is not implemented.");
  }

 protected:
  Game(GameType game_type, GameParameters game_parameters)
      : game_type_(game_type), game_parameters_(game_parameters) {}

  // Access to game parameters. Returns the value provided by the user. If not:
  // - Defaults to the value stored as the default in
  // game_type.parameter_specification if the `default_value` is absl::nullopt
  // - Returns `default_value` if provided.
  template <typename T>
  T ParameterValue(const std::string& key,
                   absl::optional<T> default_value = absl::nullopt) const {
    // Return the value if found.
    auto iter = game_parameters_.find(key);
    if (iter != game_parameters_.end()) {
      return iter->second.value<T>();
    }

    // Pick the defaulted value.
    GameParameter default_game_parameter;
    if (default_value.has_value()) {
      default_game_parameter = GameParameter(default_value.value());
    } else {
      auto default_iter = game_type_.parameter_specification.find(key);
      if (default_iter == game_type_.parameter_specification.end()) {
        SpielFatalError(absl::StrCat("The parameter for ", key,
                                     " is missing in game ", ToString()));
      }
      default_game_parameter = default_iter->second;
    }

    // Return the default value, storing it.
    absl::MutexLock lock(&mutex_defaulted_parameters_);
    iter = defaulted_parameters_.find(key);
    if (iter == defaulted_parameters_.end()) {
      // We haven't previously defaulted this value, so store the default we
      // used.
      defaulted_parameters_[key] = default_game_parameter;
    } else {
      // Already defaulted, so check we are being consistent.
      // Using different default values at different times means the game isn't
      // well-defined.
      if (default_game_parameter != iter->second) {
        SpielFatalError(absl::StrCat("Parameter ", key, " is defaulted to ",
                                     default_game_parameter.ToReprString(),
                                     " having previously been defaulted to ",
                                     iter->second.ToReprString(), " in game ",
                                     ToString()));
      }
    }
    return default_game_parameter.value<T>();
  }

  // The game type.
  GameType game_type_;

  // Any parameters supplied when constructing the game.
  GameParameters game_parameters_;

  // Track the parameters for which a default value has been used. This
  // enables us to report the actual value used for every parameter.
  mutable GameParameters defaulted_parameters_
      ABSL_GUARDED_BY(mutex_defaulted_parameters_);
  mutable absl::Mutex mutex_defaulted_parameters_;
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

std::shared_ptr<const Game> DeserializeGame(const std::string& serialized);

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

// Normalize a policy into a proper discrete distribution where the
// probabilities sum to 1.
void NormalizePolicy(ActionsAndProbs* policy);

// Used to sample a policy or chance outcome distribution.
// Probabilities of the actions must sum to 1.
// The parameter z should be a sample from a uniform distribution on the range
// [0, 1). Returns the sampled action and its probability.
std::pair<Action, double> SampleAction(const ActionsAndProbs& outcomes,
                                       double z);
std::pair<Action, double> SampleAction(const ActionsAndProbs& outcomes,
                                       absl::BitGenRef rng);

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

// Convert GameTypes from and to strings. Used for serialization of objects
// that contain them.
// Note: these are not finished! They will be finished by an external
// contributor. See https://github.com/deepmind/open_spiel/issues/234 for
// details.
std::string GameTypeToString(const GameType& game_type);
GameType GameTypeFromString(const std::string& game_type_str);

std::ostream& operator<<(std::ostream& os, const State::PlayerAction& action);

// Utility functions used mostly for debugging. This calls State::ActionToString
// for every action.
std::vector<std::string> ActionsToStrings(const State& state,
                                          const std::vector<Action>& actions);

// Calls ActionsToStrings and then calls absl::StrJoin to concatenate all the
// strings together.
std::string ActionsToString(const State& state,
                            const std::vector<Action>& actions);

// A utility to broadcast an error message with game and state info.
// It is a wrapper around SpielFatalError and meant to facilitate debugging.
void SpielFatalErrorWithStateInfo(const std::string& error_msg,
                                  const Game& game,
                                  const State& state);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_SPIEL_H_
