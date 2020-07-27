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
// Each Game object have a MakeObserver() method which returns one of these
// objects given a specification of the required observation type.

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/inlined_vector.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Forward declarations
class Game;
class State;

// Viewing a span as a multi-dimensional tensor.
struct DimensionedSpan {
  absl::InlinedVector<int, 4> shape;
  absl::Span<float> data;

  DimensionedSpan(absl::Span<float> data, absl::InlinedVector<int, 4> shape)
      : shape(std::move(shape)), data(data) {}

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
    SPIEL_DCHECK_EQ(shape.size(), 3);
    return data[((idx1 * shape[1] + idx2) * shape[2] + idx3) * shape[3] + idx4];
  }
};

// An Allocator is responsible for returning memory for an Observer.
class Allocator {
 public:
  // Returns zero-initialized memory into which the data should be written.
  // `name` is the name of this piece of the tensor; the allocator may
  // make use it to label the tensor when accessed by clients
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

// Specification of which players' private information we get to see.
enum class PrivateInfoType {
  kNone,          // No private information
  kSinglePlayer,  // Private information for the observing player only (i.e.
                  // the player passed to WriteTensor / StringFrom)
  kAllPlayers     // Private information for all players
};

// Observation types for imperfect-information games.
struct IIGObservationType {
  // If true, include public information in the observation.
  bool public_info;

  // Whether the observation is perfect recall (info state).
  // If true, observation must be sufficient to  reconstruct the complete
  // history of actions and observations for the observing player
  bool perfect_recall;

  // Which players' private information to include in the observation
  PrivateInfoType private_info;
};

// An Observer is something which can produce an observation of a State,
// e.g. a Tensor or collection of Tensors or a string.
// Observers are game-specific. They are created by a Game object, and
// may only be applied to a State class generated from the same Game instance.
class Observer {
 public:
  // Write a tensor observation to the memory returned by the Allocator.
  virtual void WriteTensor(const State& state, int player,
                           Allocator* allocator) const = 0;

  // Return a string observation. For human-readability or for tabular
  // algorithms on small games.
  virtual std::string StringFrom(const State& state, int player) const = 0;

  virtual ~Observer() = default;
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

 private:
  std::shared_ptr<Observer> observer_;
  std::vector<float> buffer_;
  std::vector<TensorInfo> tensors_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_OBSERVER_H_
