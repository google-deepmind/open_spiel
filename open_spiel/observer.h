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

#ifndef OPEN_SPIEL_OBSERVER_H_
#define OPEN_SPIEL_OBSERVER_H_

// This class is the primary method for getting observations from games.
// Each Game object has a MakeObserver() method which returns an Observer
// object given a specification of the required observation type.

// To access observation from C++, first initialize an observer and observation
// for the game (one time only).
//
//    auto observer = game->MakeObserver(iig_obs_type, params);
//    Observation observation(*game, observer);
//
// Then for states in a trajectory, get a tensor observation using:
//
//    observation.SetFrom(state, player);   // populates observation.Tensor()
//
// The resultant tensor is accessible from observation.Tensor(). Note that
// the decomposition of the tensor into named pieces is not currently available
// through this API (it is available in Python).
//
// To obtain a string observation:
//
//    std::string string_obs = observation.StringFrom(state, player);
//
// Access from Python follows a similar pattern, with the addition of support
// for accessing pieces of the observation tensor by name. See `observation.py`
// and `observation_test.py`.

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/inlined_vector.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Forward declarations
class Game;
class State;

using ObservationParams = GameParameters;

// Viewing a span as a multi-dimensional tensor.
struct DimensionedSpan {
  absl::InlinedVector<int, 4> shape;
  absl::Span<float> data;

  DimensionedSpan(absl::Span<float> data, absl::InlinedVector<int, 4> shape)
      : shape(std::move(shape)), data(data) {
    int m = std::accumulate(
        this->shape.begin(), this->shape.end(), 1, std::multiplies<int>());
    SPIEL_CHECK_EQ(m, data.size());
  }

  float& at(int idx) const {
    SPIEL_DCHECK_EQ(shape.size(), 1);
    return data[idx];
  }

  float& at(int idx1, int idx2) const {
    SPIEL_DCHECK_EQ(shape.size(), 2);
    return data[idx1 * shape[1] + idx2];
  }

  float& at(int idx1, int idx2, int idx3) const {
    SPIEL_DCHECK_EQ(shape.size(), 3);
    return data[(idx1 * shape[1] + idx2) * shape[2] + idx3];
  }

  float& at(int idx1, int idx2, int idx3, int idx4) const {
    SPIEL_DCHECK_EQ(shape.size(), 4);
    return data[((idx1 * shape[1] + idx2) * shape[2] + idx3) * shape[3] + idx4];
  }
};

// An Allocator is responsible for returning memory for an Observer.
class Allocator {
 public:
  // Returns zero-initialized memory into which the data should be written.
  // `name` is the name of this piece of the tensor; the allocator may
  // use it to label the tensor when accessed by the clients.
  virtual DimensionedSpan Get(absl::string_view name,
                              const absl::InlinedVector<int, 4>& shape) = 0;

  virtual ~Allocator() = default;
};

// Allocates memory from a single block. This is intended for use when it
// is already known how much memory an observation consumes. The allocator
// owns a fixed-size block of memory and returns pieces of it in sequence.
class ContiguousAllocator : public Allocator {
 public:
  ContiguousAllocator(absl::Span<float> data) : data_(data), offset_(0) {
    absl::c_fill(data, 0);
  }
  DimensionedSpan Get(absl::string_view name,
                      const absl::InlinedVector<int, 4>& shape) override;

 private:
  absl::Span<float> data_;
  int offset_;
};


// Information about a tensor (shape and type).
struct TensorInfo {
  std::string name;
  std::vector<int> shape;

  std::string DebugString() const {
    return absl::StrCat("TensorInfo(name='", name, "', shape=(",
                        absl::StrJoin(shape, ","), ")");
  }
};

// Allocates new memory for each allocation request and keeps track
// of tensor names and shapes. This is intended to use when it's not yet
// known how much memory an observation consumes.
class TrackingVectorAllocator : public Allocator {
 public:
  TrackingVectorAllocator() {}
  DimensionedSpan Get(absl::string_view name,
                      const absl::InlinedVector<int, 4>& shape);
  bool IsNameUnique(absl::string_view name);

  std::vector<TensorInfo> tensors;
  std::vector<float> data;
};

// Specification of which players' private information we get to see.
enum class PrivateInfoType {
  kNone,          // No private information.
  kSinglePlayer,  // Private information for the observing player only (i.e.
                  // the player passed to WriteTensor / StringFrom).
  kAllPlayers     // Private information for all players.
};

// Observation types for imperfect-information games.

// The public / private observations factorize observations into their
// (mostly) non-overlapping public and private parts. They may overlap only for
// the start of the game and time. See also fog/ directory for details.
//
// The public observations correspond to information that all the players know
// that all the players know, like upward-facing cards on a table.
// Perfect information games, like Chess, have only public observations.
//
// All games have non-empty public observations. The minimum public
// information is time: we assume that all the players can perceive absolute
// time (which can be accessed via the MoveNumber() method). The implemented
// games must be 1-timeable, a property that is trivially satisfied with all
// human-played board games, so you typically don't have to worry about this.
// (You'd have to knock players out / consider Einstein's time-relativistic
// effects to make non-timeable games.).
//
// The public observations are used to create a sequence of observations:
// a public observation history. Because of the sequential structure, when you
// return any non-empty public observation, you implicitly encode time as well
// within this sequence.
//
// Public observations are not required to be "common knowledge" observations.
// Example: In imperfect-info version of card game Goofspiel, players make
// bets with cards on their hand, and their imperfect information consists of
// not knowing exactly what cards the opponent currently holds, as the players
// only learn public information whether they have won/lost/draw the bet.
// However, when the player bets a card "5" and learns it drew the round,
// it can infer that the opponent must have also bet the card "5", just as the
// player did. In principle we could ask the game to make this inference
// automatically, and return observation "draw-5". We do not require this, as
// it is in general expensive to compute. Returning public observation "draw"
// is sufficient.
//
// In the initial state this function must return
// kStartOfGamePublicObservation. If there is no public observation available
// except time, the implementation must return kClockTickObservation. Note
// that empty strings for public observations are forbidden - they correspond
// to kInvalidPublicObservation.

// The private observations correspond to the part of the observation that
// is not public. In Poker, this would be the cards the player holds in his
// hand. Note that this does not imply that other players don't have access
// to this information.
//
// For example, consider there is a mirror behind an unaware player, betraying
// his hand in the reflection. Even if everyone was aware of the mirror, then
// this information still may not be public, because the players do not know
// for certain that everyone is aware of this. It would become public if and
// only if all the players were aware of the mirror, and they also knew that
// indeed everyone else knows about it too. Then this would effectively make
// it the same as if the player just placed his cards on the table for
// everyone to see.
//
// If there is no private observation available, the implementation should
// return kNothingPrivateObservation. Perfect information games have no
// private observations and should return only this constant.
// Imperfect-information games should return a different string
// at least once in at least one possible trajectory of the game (otherwise
// they would be considered perfect-info games).
struct IIGObservationType {
  // If true, include public information in the observation.
  bool public_info;

  // Whether the observation is perfect recall (identical to an info state).
  // If true, the observation must be sufficient to reconstruct the complete
  // history of actions and observations for the observing player.
  bool perfect_recall;

  // Which players' private information to include in the observation.
  PrivateInfoType private_info;

  bool operator==(const IIGObservationType&);
};

// Default observation type for imperfect information games.
// Corresponds to the ObservationTensor / ObservationString methods.
inline constexpr IIGObservationType kDefaultObsType{
    .public_info = true,
    .perfect_recall = false,
    .private_info = PrivateInfoType::kSinglePlayer};

// Default observation type for imperfect information games.
// Corresponds to the InformationStateTensor / InformationStateString methods.
inline constexpr IIGObservationType kInfoStateObsType{
    .public_info = true,
    .perfect_recall = true,
    .private_info = PrivateInfoType::kSinglePlayer};

// Incremental public observation, mainly used for imperfect information games.
inline constexpr IIGObservationType kPublicObsType{
    .public_info = true,
    .perfect_recall = false,
    .private_info = PrivateInfoType::kNone};

// Complete public observation, mainly used for imperfect information games.
inline constexpr IIGObservationType kPublicStateObsType{
    .public_info = true,
    .perfect_recall = true,
    .private_info = PrivateInfoType::kNone};

// An Observer is something which can produce an observation of a State,
// e.g. a Tensor or collection of Tensors or a string.
// Observers are game-specific. They are created by a Game object, and
// may only be applied to a State class generated from the same Game instance.
class Observer {
 public:
  Observer(bool has_string, bool has_tensor)
      : has_string_(has_string), has_tensor_(has_tensor) {
    SPIEL_CHECK_TRUE(has_string || has_tensor);
  }

  // Write a tensor observation to the memory returned by the Allocator.
  virtual void WriteTensor(const State& state, int player,
                           Allocator* allocator) const = 0;

  // Return a string observation. For human-readability or for tabular
  // algorithms on small games.
  virtual std::string StringFrom(const State& state, int player) const = 0;

  // What observations do we support?
  bool HasString() const { return has_string_; }
  bool HasTensor() const { return has_tensor_; }

  virtual ~Observer() = default;

 protected:
  // TODO(author11) Remove when all games support both types of observations
  bool has_string_;
  bool has_tensor_;
};

// Holds an Observer and a vector for it to write values into.
class Observation {
 public:
  // Create
  Observation(const Game& game, std::shared_ptr<Observer> observer);

  // Returns the internal buffer into which observations are written.
  absl::Span<float> Tensor() { return absl::MakeSpan(buffer_); }

  // Returns information on the component tensors of the observation.
  const std::vector<TensorInfo>& tensor_info() const { return tensors_; }

  // Gets the observation from the State and player and stores it in
  // the internal tensor.
  void SetFrom(const State& state, int player);

  // Returns the string observation for the State and player.
  std::string StringFrom(const State& state, int player) const {
    return observer_->StringFrom(state, player);
  }

  // Return compressed representation of the observations. This is useful for
  // memory-intensive algorithms, e.g. that store large replay buffers.
  //
  // The first byte of the compressed data is reserved for the specific
  // compression scheme. Note that currently there is only one supported, which
  // requires bitwise observations.
  std::string Compress() const;
  void Decompress(absl::string_view compressed);

  // What observations do we support?
  // TODO(author11) Remove when all games support both types of observations
  bool HasString() const { return observer_->HasString(); }
  bool HasTensor() const { return observer_->HasTensor(); }

 private:
  std::shared_ptr<Observer> observer_;
  std::vector<float> buffer_;
  std::vector<TensorInfo> tensors_;
};

// Allows to registers observers to a game. Usage:
// ObserverRegisterer unused_name(game_name, observer_name, creator);
//
// Once an observer is registered, it can be created by
// game.MakeObserver(iig_obs_type, observer_name)
class ObserverRegisterer {
 public:
  // Function type which creates an observer. The game and params argument
  // cannot be assumed to exist beyond the scope of this call.
  using CreateFunc = std::function<std::shared_ptr<Observer>(
      const Game& game, absl::optional<IIGObservationType> iig_obs_type,
      const ObservationParams& params)>;

  ObserverRegisterer(const std::string& game_name,
                     const std::string& observer_name,
                     CreateFunc creator);
  static void RegisterObserver(const std::string& game_name,
                               const std::string& observer_name,
                               CreateFunc creator);

  static std::shared_ptr<Observer> CreateByName(
      const std::string& observer_name,
      const Game& game,
      absl::optional<IIGObservationType> iig_obs_type,
      const ObservationParams& params);

 private:
  // Returns a "global" map of registrations (i.e. an object that lives from
  // initialization to the end of the program). Note that we do not just use
  // a static data member, as we want the map to be initialized before first
  // use.
  static std::map<std::pair<std::string, std::string>, CreateFunc>&
  observers() {
    static std::map<std::pair<std::string, std::string>, CreateFunc> impl;
    return impl;
  }
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_OBSERVER_H_
