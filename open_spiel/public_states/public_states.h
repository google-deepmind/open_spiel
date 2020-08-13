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

// To use public states you must enable building with both
// BUILD_WITH_PUBLIC_STATES=ON and BUILD_WITH_EIGEN=ON env vars.

#ifndef OPEN_SPIEL_PUBLIC_STATES_H_
#define OPEN_SPIEL_PUBLIC_STATES_H_

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

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/bit_gen_ref.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/eigen/pyeig.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/random.h"

// This files specifies the public state API for OpenSpiel.
// It is an imperfect-recall abstraction for Factored-Observation Games [1].
// Many of the decisions are described in the README documentation.
//
// [1] https://arxiv.org/abs/1906.11110

namespace open_spiel {
namespace public_states {

// Static information for a game. This will determine what algorithms,
// solution concepts and automatic consistency checks are applicable.
struct GameWithPublicStatesType {
  // A short name with no spaces that uniquely identifies the game, e.g.
  // "msoccer". This is the key used to distinguish games. It must be the same
  // as GameType::short_name of the underlying Base API.
  std::string short_name;

  // Provides methods that compute values needed to run counter-factual
  // regret minimization: reach probs and cf. values.
  bool provides_cfr_computation;

  // Does the implementation provide IsStateCompatible() implementations?
  // These are useful for automatic consistency checks with Base API.
  //
  // They provide a way of comparing imperfect-recall public / private
  // informations with perfect recall public-observation histories /
  // action private-observation histories, as these are uniquely defined for
  // each State.
  bool provides_state_compatibility_check;
};

// An abstract class that represents a private information in the game.
//
// This is an imperfect-recall variant of private information. This means there
// might be multiple Action-PrivateObservation histories that yield the same
// private information.
//
// The private information does not contain any piece of public information!
class PrivateInformation {
 public:
  explicit PrivateInformation(std::shared_ptr<const Game> base_game);
  PrivateInformation(const PrivateInformation&) = default;
  virtual ~PrivateInformation() = default;

  // The player that owns this private information.
  virtual Player GetPlayer() const {
    SpielFatalError("GetPlayer() is not implemented.");
  }

  // A number that uniquely identifies position of this private information
  // within a ReachProbs vector.
  //
  // Equality of PrivateInformation implies the same ReachProbsIndex().
  virtual int ReachProbsIndex() const {
    SpielFatalError("ReachProbsIndex() is not implemented.");
  }

  // A number that uniquely identifies position of this private information
  // within a neural network input.
  //
  // Equality of PrivateInformation implies the same NetworkIndex().
  virtual int NetworkIndex() const {
    SpielFatalError("NetworkIndex() is not implemented.");
  }

  // Can State produce this private information?
  //
  // Implementing this method is optional, but highly recommended, as it
  // helps with testing consistency of the implementation with Base API.
  //
  // See also GameWithPublicStatesType::provides_state_compatibility_check
  virtual bool IsStateCompatible(const State&) const {
    SpielFatalError("IsStateCompatible() is not implemented.");
  }

  // A human-readable string representation.
  // Equality of PrivateInformation implies they have the same ToString()
  virtual std::string ToString() const {
    SpielFatalError("ToString() is not implemented.");
  }

  virtual std::unique_ptr<PrivateInformation> Clone() const {
    SpielFatalError("Clone() is not implemented.");
  }

  // Serializes a private information into a string.
  //
  // If overridden, this must be the inverse of
  // GameWithPublicStates::DeserializePrivateInformation
  //
  // Two PrivateInformations are equal if and only if they have the same
  // Serialize() outputs.
  virtual std::string Serialize() const {
    SpielFatalError("Serialize() is not implemented.");
  }

  // Compare whether the other private information is equal.
  virtual bool operator==(const PrivateInformation& other) const {
    SpielFatalError("operator==() is not implemented.");
  }

  bool operator!=(const PrivateInformation& other) const {
    return !operator==(other);
  }

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetGame() const { return base_game_; }

 protected:
  // A pointer to the game that created this public information.
  const std::shared_ptr<const Game> base_game_;
};

// An edge in the public tree. This is the same as output of
// State::PublicObservationString()
using PublicTransition = std::string;

// A container for counter-factual values for each private state (private
// information) within a public state. The cf. values correspond conceptually
// to the V-function (state value function) from reinforcement learning, except
// for the reach probability terms.
// The values are always accompanied by the owning player. The values must be
// always within the range of game's Max/Min Returns.
struct CfPrivValues {
  const Player player;
  ArrayXd cfvs;
};

std::ostream& operator<<(std::ostream& stream, const CfPrivValues& values);

// A container for counter-factual action-values for each action of a private
// state (private information). The cf. action-values correspond conceptually to
// the Q-function (state-action value function) from reinforcement learning,
// except for the reach probability terms.
// The values are always accompanied by the owning player. The values must be
// always within the range of game's Max/Min Returns.
struct CfActionValues {
  const Player player;
  ArrayXd cfavs;
};

std::ostream& operator<<(std::ostream& stream, const CfActionValues& values);

// A container for reach probabilities of the player, for each of its
// private state within a public state.
// The values are always accompanied by the owning player.
// The reach probs must sum to 1.
struct ReachProbs {
  const Player player;
  ArrayXd probs;
};

// Defines symbol for PublicState, implemented later.
class GameWithPublicStates;

// A helper constant for constructing tensors.
// The PublicState::ToTensor() must be of the same size for all public states,
// and therefore there might be some "holes" when we encode smaller public
// states, either be it features or ranges. This value is used for such unused
// slots.
constexpr inline double kTensorUnusedSlotValue = -1.;

// A public state is perfect recall - it corresponds to an object specified by
// public-observation history and provides methods on top of it.
// It corresponds to a specific node within a public tree.
class PublicState {
 public:
  explicit PublicState(std::shared_ptr<const GameWithPublicStates> public_game);
  // Construct public state based on public observation history.
  // The default implementation trivially tries to apply the public transitions,
  // while checking each supplied transition is valid.
  explicit PublicState(std::shared_ptr<const GameWithPublicStates> public_game,
                       std::vector<PublicTransition> pub_obs_history);
  PublicState(const PublicState&) = default;
  virtual ~PublicState() = default;

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Perspectives over the public state.">
  // ---------------------------------------------------------------------------

  // Return the public observation history.
  const std::vector<PublicTransition>& GetPublicObservationHistory() const {
    SPIEL_CHECK_GE(pub_obs_history_.size(), 1);
    return pub_obs_history_;
  }

  // Return the last public transition that was made
  // to get to this public state.
  const PublicTransition& LastTransition() const {
    return pub_obs_history_.back();
  }

  // Return numbers of the private informations consistent with the
  // public information (for each player).
  virtual std::vector<int> NumDistinctPrivateInformations() const {
    SpielFatalError("NumDistinctPrivateInformations() is not implemented.");
  }

  // Returns all the possible private informations for requested player that
  // are possible for the public information of this public state.
  // Their ordering within the returned vector must be consistent with
  // their ReachProbsIndex, the first element having ReachProbsIndex = 0
  // and the last element having ReachProbsIndex = returned_list.size()-1
  virtual std::vector<PrivateInformation> GetPrivateInformations(Player) const {
    SpielFatalError("GetPrivateInformations() is not implemented.");
  }

  // Return all States that are consistent with this public state in the sense
  // that they have the same public observation history. However, there may
  // be an exponential number of them, given that we are doing imperfect
  // recall abstraction. Therefore, return a minimally sized set of these
  // states such that there is no other state that is isomorphic to any
  // of them.
  virtual std::vector<std::unique_ptr<State>> GetPublicSet() const {
    SpielFatalError("GetPublicSet() is not implemented.");
  }

  // Return an information state string description
  // for this public state + private information.
  virtual std::string GetInformationState(const PrivateInformation&) const {
    SpielFatalError("GetPublicSet() is not implemented.");
  }

  // Return all states that are consistent with the player’s private
  // information and public state.
  virtual std::vector<std::unique_ptr<State>> GetInformationSet(
      const PrivateInformation&) const {
    SpielFatalError("GetInformationSet() is not implemented.");
  }

  // Return State that corresponds to the combination of player’s private
  // informations and this public state.
  virtual std::unique_ptr<State> GetWorldState(
      const std::vector<PrivateInformation*>&) const {
    SpielFatalError("GetWorldState() is not implemented.");
  }

  // Return the private information at specified State and player
  // at this public state.
  virtual std::unique_ptr<PrivateInformation> GetPrivateInformation(
      const State&, Player) const {
    SpielFatalError("GetPrivateInformation() is not implemented.");
  }

  // Return actions that are available to player in his information state,
  // corresponding the the this public state and player's private information.
  virtual std::vector<Action> GetPrivateActions(
      const PrivateInformation& information) const {
    SpielFatalError("GetPrivateActions() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Fetch a random subset from a perspective">
  // ---------------------------------------------------------------------------

  virtual std::unique_ptr<State> ResampleFromPublicSet(Random*) const {
    SpielFatalError("ResampleFromPublicSet() is not implemented.");
  }

  virtual std::unique_ptr<State> ResampleFromInformationSet(
      const PrivateInformation&, Random*) const {
    SpielFatalError("ResampleFromInformationSet() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Traversal of public state">
  // ---------------------------------------------------------------------------

  // Updates the public state when a public-tree action (transition) is made.
  // Games should implement DoApplyPublicTransition.
  //
  // Mutates public state!
  void ApplyPublicTransition(const PublicTransition& transition) {
    SPIEL_CHECK_FALSE(IsTerminal());
    DoApplyPublicTransition(transition);
    pub_obs_history_.push_back(transition);
  }

  // Updates the public state when a world-level action is made.
  // Games should implement DoApplyStateAction. This should work also for
  // simultaneous-move games.
  //
  // Mutates public state!
  virtual PublicTransition ApplyStateAction(
      const std::vector<PrivateInformation*>& privates, Action action) {
    PublicTransition transition = DoApplyStateAction(privates, action);
    pub_obs_history_.push_back(transition);
    return transition;
  }

  std::unique_ptr<PublicState> Child(const PublicTransition& transition) const {
    std::unique_ptr<PublicState> child = Clone();
    child->ApplyPublicTransition(transition);
    return child;
  }

  virtual std::vector<PublicTransition> LegalTransitions() const {
    SpielFatalError("LegalTransitions() is not implemented.");
  }

  // For each private information of a specified player return the number
  // of private actions. Note that if the player is not acting in this public
  // state, the returned vector should be empty.
  virtual std::vector<int> CountPrivateActions(Player player) const {;
    SpielFatalError("CountPrivateActions() is not implemented.");
  }

  // Undoes the last transition, which must be supplied. This is a method
  // to get a parent public state.
  // One must call pub_obs_history_.pop_back() in the implementations.
  //
  // Mutates public state!
  virtual void UndoTransition(const PublicTransition&) {
    SpielFatalError("UndoTransition() is not implemented.");
  }

  // Checks if this public transition can be applied in this public state.
  // The implementation checks if the transition is in the list provided
  // by LegalTransitions().
  bool IsTransitionLegal(const PublicTransition& transition) const {
    for (const PublicTransition& valid_transition : LegalTransitions()) {
      if (valid_transition == transition) return true;
    }
    return false;
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Public state types">
  // ---------------------------------------------------------------------------

  // Is this a chance public state?
  virtual bool IsChance() const {
    SpielFatalError("IsChance() is not implemented.");
  }

  // Is this public state terminal?
  virtual bool IsTerminal() const {
    SpielFatalError("IsTerminal() is not implemented.");
  }

  // Is this a player public state?
  virtual bool IsPlayer() const {
    SpielFatalError("IsPlayer() is not implemented.");
  }

  // Collection of currently acting players, if this is a player public state.
  // There is only one player acting within a public state for
  // GameType::Dynamics::kSequential.
  virtual std::vector<Player> ActingPlayers() const {
    SpielFatalError("IsPlayer() is not implemented.");
  }

  // Is the specified player acting in this public state?
  virtual bool IsPlayerActing(Player player) const {
    for (const Player& acting_player : ActingPlayers()) {
      if (acting_player == player) return true;
    }
    return false;
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Terminals">
  // ---------------------------------------------------------------------------

  // Return utility of a terminal world state for each player.
  virtual std::vector<double> TerminalReturns(
      const std::vector<PrivateInformation*>& privates) const {
    std::unique_ptr<State> terminal_state = GetWorldState(privates);
    SPIEL_CHECK_TRUE(terminal_state->IsTerminal());
    return terminal_state->Returns();
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="CFR-related computations">
  // ---------------------------------------------------------------------------

  // Reach probs ---------------------------------------------------------------

  // Compute reach probs of a player if it was going to transition to a child
  // public state. The reach prob is updated using the private strategy of the
  // player if he is acting. This function works also for players who do not
  // act in this public state, they just supply an empty strategy.
  //
  // Strategy of the player: [private_states x private_actions]
  // Each element of the vector should be a valid probability distribution.
  virtual ReachProbs ComputeReachProbs(
      const PublicTransition&, const std::vector<ArrayXd>& strategy,
      ReachProbs) const {
    SpielFatalError("ComputeReachProbs() is not implemented.");
  }

  // Counter-factual values ----------------------------------------------------

  // Return the counter-factual values for terminals for requested player,
  // given each players' respective reach probs. The reach probs of the
  // requested player may be empty, as they are not used in the computation.
  virtual CfPrivValues TerminalCfValues(
      const std::vector<ReachProbs>&, Player) const {
    SpielFatalError("TerminalCfValues() is not implemented.");
  }

  // Compute counter-factual values of private states (information states)
  // corresponding to the imperfect-recall private informations.  The cf. values
  // correspond conceptually to the V-function (state value function)
  // from reinforcement learning.
  //
  // This is a vectorized version for computation of the counter-factual value
  // of a private state:
  //
  //   v(I) = \sum \sigma(I,a) v(I,a)
  //
  // We are provided the children cf. action-values and the player policies
  // to reach them. The action-values have the same size as the privates
  // strategy.
  //
  // children_values: [private_state I  x  cf. action values]
  // children_policy: [private_state I  x  policy for each action]
  // Returned values: [cf. value per private_state I]
  virtual CfPrivValues ComputeCfPrivValues(
      const std::vector<CfActionValues>& children_values,
      const std::vector<ArrayXd>& children_policy) const {
    SpielFatalError("ComputeCfPrivValues() is not implemented.");
  }

  // Compute counter-factual action-values (i.e. values when the player follows
  // each private action with 100% probability). The cf. action-values
  // correspond conceptually to the Q-function (state-action value function)
  // from reinforcement learning.
  //
  // This is a vectorized version for computation of the counter-factual
  // action-value v(I,a) of a private state I. Within the private tree
  // of a player v(I,a) can be computed as sum of the child infosets J:
  //
  //   v(I,a) = \sum v(J)
  //
  // Note that these child infosets J belong to the next public state.
  //
  // We are provided these children cf. values and we return cf. action-values:
  //
  // children_values: [private_state I  x  cf. values for child infosets J]
  // Returned values: [private_state I  x  cf. action-value for each 'a' of 'I']
  virtual std::vector<CfActionValues> ComputeCfActionValues(
      const std::vector<CfPrivValues>& children_values) const {
    SpielFatalError("ComputeCfActionValues() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Neural networks">
  // ---------------------------------------------------------------------------

  // Return feature representation of this public state.
  virtual std::vector<double> PublicFeaturesTensor() const {
    SpielFatalError("PublicFeaturesTensor() is not implemented.");
  }

  // Encode reach probabilities into a tensor of size
  // product of MaxDistinctPrivateInformationsCount
  virtual std::vector<double> ReachProbsTensor(
      const std::vector<ReachProbs>&) const;

  // Return tensor that builds tensor representation for the public state
  // given the reach probs of all players. The default representation
  // concatenates at first the ReachProbs of all players and then it adds
  // PublicFeaturesTensor().
  // Additionally, the ReachProbs are placed within the tensor at locations
  // specified by PrivateInformation::NetworkIndex()
  virtual std::vector<double> ToTensor(const std::vector<ReachProbs>&) const;

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Miscellaneous">
  // ---------------------------------------------------------------------------

  // Human readable description of the public state.
  virtual std::string ToString() const {
    return absl::StrJoin(pub_obs_history_, ",");
  }

  // Depth of the public state within the public tree.
  // The root public state has depth of 0, but contains already
  // the first public observation - start of the game.
  virtual int MoveNumber() const { return pub_obs_history_.size() - 1; }

  // Is the current public state root of the public tree?
  virtual bool IsRoot() const { return MoveNumber() == 0; }

  virtual std::unique_ptr<PublicState> Clone() const {
    SpielFatalError("Clone() is not implemented.");
  }

  // Serializes a public state into a string.
  //
  // If overridden, this must be the inverse of
  // GameWithPublicStates::DeserializePublicState
  virtual std::string Serialize() const {
    SpielFatalError("Serialize() is not implemented.");
  }

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetBaseGame() const { return base_game_; }

  // Get the game object that generated this state.
  std::shared_ptr<const GameWithPublicStates> GetPublicGame() const {
    return public_game_;
  }

  // Compare if the public state has exactly the same public observation
  // history. This is not the same as comparing two PublicInformations!
  bool operator==(const PublicState& other) const {
    return pub_obs_history_ == other.pub_obs_history_;
  }

  bool operator!=(const PublicState& other) const { return !operator==(other); }

  // </editor-fold>

 protected:
  // See ApplyPublicTransition.
  // Mutates public state!
  virtual void DoApplyPublicTransition(const PublicTransition&) {
    SpielFatalError("DoApplyPublicTransition() is not implemented.");
  }
  // See ApplyStateAction.
  // Mutates public state!
  virtual PublicTransition DoApplyStateAction(
      const std::vector<PrivateInformation*>& informations, Action a) {
    std::unique_ptr<State> state = GetWorldState(informations);
    SPIEL_CHECK_FALSE(state->IsTerminal());
    state->ApplyAction(a);
    PublicTransition transition =
        observer_->StringFrom(*state, kDefaultPlayerId);
    ApplyPublicTransition(transition);
    return transition;
  }

  // Public observations received so far.
  std::vector<PublicTransition> pub_obs_history_ = {
      kStartOfGamePublicObservation};

  // A pointer to the game that created this public state.
  const std::shared_ptr<const GameWithPublicStates> public_game_;
  const std::shared_ptr<const Game> base_game_;
  const std::shared_ptr<const Observer> observer_;
};

// An abstract game class that provides methods for constructing
// public state API objects and asking for properties of the public tree.
class GameWithPublicStates
    : public std::enable_shared_from_this<GameWithPublicStates> {
 public:
  GameWithPublicStates(std::shared_ptr<const Game> base_game)
      : base_game_(std::move(base_game)) {}
  GameWithPublicStates(const GameWithPublicStates&) = default;
  virtual ~GameWithPublicStates() = default;

  // Create a new initial public state, that is a root of the public tree.
  virtual std::unique_ptr<PublicState> NewInitialPublicState() const {
    SpielFatalError("NewInitialPublicState() is not implemented.");
  }

  // Create reach probs that players have for the root public state.
  virtual std::vector<ReachProbs> NewInitialReachProbs() const {
    SpielFatalError("NewInitialReachProbs() is not implemented.");
  }

  // Provide information about the maximum number of distinct private
  // informations per player in any public state in the game. Note that this
  // should not be an arbitrary upper bound, but indeed the maximum number,
  // because it serves to specify the sizes of neural network inputs. Some
  // algorithms will likely use it to preallocate memory.
  //
  // Example: in HUNL Poker players receive 2 cards from the pile of 52 cards,
  // which makes it 52 * 51 / 2 = 1326 for each player.
  virtual std::vector<int> MaxDistinctPrivateInformationsCount() const {
    SpielFatalError(
        "MaxDistinctPrivateInformationsCount() is not implemented.");
  }

  // Returns the sum of the maximum number of distinct private informations
  // over all the players
  int SumMaxDistinctPrivateInformations() const {
    return absl::c_accumulate(MaxDistinctPrivateInformationsCount(), 0);
  }

  // Returns the number of public features. Each public state should be
  // represented by the same number of features.
  virtual int NumPublicFeatures() const {
    SpielFatalError("NumPublicFeatures() is not implemented.");
  }

  // Returns the size of the neural network input.
  virtual int NetworkInputSize() const {
    return NumPublicFeatures() + SumMaxDistinctPrivateInformations();
  }

  // Returns a newly allocated private information built from a string.
  // Caller takes ownership of the object.
  //
  // If this method is overridden, then it should be the inverse of
  // PrivateInformation::Serialize (i.e. that method should also be overridden).
  virtual std::unique_ptr<PrivateInformation> DeserializePrivateInformation()
      const {
    SpielFatalError("DeserializePrivateInformation() is not implemented.");
  }

  // Returns a newly allocated public state built from a string.
  // Caller takes ownership of the object.
  //
  // If this method is overridden, then it should be the inverse of
  // PublicState::Serialize (i.e. that method should also be overridden).
  virtual std::unique_ptr<PublicState> DeserializePublicState() const {
    SpielFatalError("DeserializePublicState() is not implemented.");
  }

  int NumPlayers() const { return base_game_->NumPlayers(); }

  std::shared_ptr<const Game> GetBaseGame() const { return base_game_; }

 protected:
  std::shared_ptr<const Game> base_game_;
};

#define REGISTER_SPIEL_GAME_WITH_PUBLIC_STATES(info, factory) \
  GameWithPublicStatesRegisterer CONCAT(game, __COUNTER__)(info, factory);

class GameWithPublicStatesRegisterer {
 public:
  using CreateFunc = std::function<std::shared_ptr<const GameWithPublicStates>(
      std::shared_ptr<const Game>)>;

  GameWithPublicStatesRegisterer(const GameWithPublicStatesType& game_type,
                                 CreateFunc creator);

  static std::shared_ptr<const GameWithPublicStates> CreateByName(
      const std::string& short_name, const GameParameters& params);
  static std::shared_ptr<const GameWithPublicStates> CreateByGame(
      std::shared_ptr<const Game> base_game);

  static std::vector<std::string> RegisteredNames();
  static std::vector<GameWithPublicStatesType> RegisteredGames();
  static bool IsValidName(const std::string& short_name);
  static void RegisterGame(const GameWithPublicStatesType& game_type,
                           CreateFunc creator);

 private:
  // Returns a "global" map of registrations (i.e. an object that lives from
  // initialization to the end of the program). Note that we do not just use
  // a static data member, as we want the map to be initialized before first
  // use.
  static std::map<std::string, std::pair<GameWithPublicStatesType, CreateFunc>>&
  factories() {
    static std::map<std::string,
                    std::pair<GameWithPublicStatesType, CreateFunc>>
        impl;
    return impl;
  }
};

// Returns true if the game is registered with public state API,
// false otherwise.
bool IsGameRegisteredWithPublicStates(const std::string& short_name);

// Returns a list of registered games' short names for games that have
// public state API.
std::vector<std::string> RegisteredGamesWithPublicStates();

// Returns a list of registered game types for games that have public state API.
std::vector<GameWithPublicStatesType> RegisteredGameTypesWithPublicStates();

// Returns a new game object from the specified string, which is the short
// name plus optional parameters, e.g. "go(komi=4.5,board_size=19)"
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    const std::string& game_string);

// Returns a new game object with the specified parameters.
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    const std::string& short_name, const GameParameters& params);

// Returns a new game object with the specified parameters; reads the name
// of the game from the 'name' parameter (which is not passed to the game
// implementation).
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    GameParameters params);

// Returns a new game object from the underlying Base API game object.
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    std::shared_ptr<const Game> base_game);

std::string SerializeGameWithPublicState(const GameWithPublicStates& game,
                                         const PublicState& state);

std::pair<std::shared_ptr<const GameWithPublicStates>,
          std::unique_ptr<PublicState>>
DeserializeGameWithPublicState(const std::string& serialized_state);

}  // namespace public_states
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PUBLIC_STATES_H_
