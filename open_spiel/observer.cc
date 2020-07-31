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

#include "open_spiel/observer.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

DimensionedSpan ContiguousAllocator::Get(
    absl::string_view name, const absl::InlinedVector<int, 4>& shape) {
  const int size = absl::c_accumulate(shape, 1, std::multiplies<int>());
  auto piece = data_.subspan(offset_, size);
  offset_ += size;
  return DimensionedSpan(piece, shape);
}

namespace {

class InformationStateObserver : public Observer {
 public:
  InformationStateObserver(const Game& game)
      : has_string_(game.GetType().provides_information_state_string),
        has_tensor_(game.GetType().provides_information_state_tensor),
        size_(has_tensor_ ? game.InformationStateTensorSize() : 0) {
    if (has_tensor_) {
      auto shape = game.InformationStateTensorShape();
      shape_.assign(shape.begin(), shape.end());
    }
  }
  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override {
    auto tensor = allocator->Get("info_state", shape_);
    state.InformationStateTensor(player, tensor.data);
  }

  std::string StringFrom(const State& state, int player) const override {
    return state.InformationStateString(player);
  }

  bool HasString() const { return has_string_; }
  bool HasTensor() const { return has_tensor_; }

 private:
  absl::InlinedVector<int, 4> shape_;
  bool has_string_;
  bool has_tensor_;
  int size_;
};

class DefaultObserver : public Observer {
 public:
  DefaultObserver(const Game& game)
      : has_string_(game.GetType().provides_observation_string),
        has_tensor_(game.GetType().provides_observation_tensor),
        size_(has_tensor_ ? game.ObservationTensorSize() : 0) {
    if (has_tensor_) {
      auto shape = game.ObservationTensorShape();
      shape_.assign(shape.begin(), shape.end());
    }
  }

  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override {
    auto tensor = allocator->Get("observation", shape_);
    state.ObservationTensor(player, tensor.data);
  }

  std::string StringFrom(const State& state, int player) const override {
    return state.ObservationString(player);
  }

  bool HasString() const { return has_string_; }
  bool HasTensor() const { return has_tensor_; }

 private:
  absl::InlinedVector<int, 4> shape_;
  bool has_string_;
  bool has_tensor_;
  int size_;
};

}  // namespace

std::shared_ptr<Observer> Game::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  // This implementation is just for games which don't yet support the new API.
  // New games should implement MakeObserver themselves to return a
  // game-specific observer.
  if (!params.empty()) SpielFatalError("Observer params not supported.");
  if (!iig_obs_type) return absl::make_unique<DefaultObserver>(*this);
  // TODO(author11) Reinstate this check
  // SPIEL_CHECK_EQ(GetType().information,
  //                GameType::Information::kImperfectInformation);
  if (iig_obs_type->public_info && !iig_obs_type->perfect_recall &&
      iig_obs_type->private_info == PrivateInfoType::kSinglePlayer) {
    if (game_type_.provides_observation_tensor ||
        game_type_.provides_observation_string)
      return absl::make_unique<DefaultObserver>(*this);
  }
  if (iig_obs_type->public_info && iig_obs_type->perfect_recall &&
      iig_obs_type->private_info == PrivateInfoType::kSinglePlayer) {
    if (game_type_.provides_information_state_tensor ||
        game_type_.provides_information_state_string)
      return absl::make_unique<InformationStateObserver>(*this);
  }
  SpielFatalError("Requested Observer type not available.");
}

class TrackingVectorAllocator : public Allocator {
 public:
  TrackingVectorAllocator() {}
  DimensionedSpan Get(absl::string_view name,
                      const absl::InlinedVector<int, 4>& shape) {
    tensors.push_back(
        TensorInfo{std::string(name), {shape.begin(), shape.end()}});
    const int begin_size = data.size();
    const int size = absl::c_accumulate(shape, 1, std::multiplies<int>());
    data.resize(begin_size + size);
    return DimensionedSpan(absl::MakeSpan(data).subspan(begin_size, size),
                           shape);
  }

  std::vector<TensorInfo> tensors;
  std::vector<float> data;
};

Observation::Observation(const Game& game, std::shared_ptr<Observer> observer)
    : observer_(std::move(observer)) {
  // Get an observation of the initial state to set up.
  if (HasTensor()) {
    auto state = game.NewInitialState();
    TrackingVectorAllocator allocator;
    observer_->WriteTensor(*state, /*player=*/0, &allocator);
    buffer_ = std::move(allocator.data);
    tensors_ = std::move(allocator.tensors);
  }
}

void Observation::SetFrom(const State& state, int player) {
  ContiguousAllocator allocator(absl::MakeSpan(buffer_));
  observer_->WriteTensor(state, player, &allocator);
}

// We may in the future support multiple compression schemes.
// The Compress() method should select the most effective scheme adaptively.
enum CompressionScheme : char {
  kCompressionNone,   // We weren't able to compress the data.
  kCompressionBinary  // Data is binary (all elements zero or one).
};
constexpr int kNumHeaderBytes = 1;

// Binary compression.
struct BinaryCompress {
  static constexpr int kBitsPerByte = 8;

  static std::string Compress(absl::Span<const float> buffer) {
    const int num_bytes = (buffer.size() + kBitsPerByte - 1) / kBitsPerByte;
    std::string str(num_bytes + kNumHeaderBytes, '\0');
    str[0] = kCompressionBinary;

    for (int byte = 0; byte < num_bytes; ++byte) {
      for (int bit = 0; bit < kBitsPerByte; ++bit) {
        if (buffer[byte * kBitsPerByte + bit]) {
          str[kNumHeaderBytes + byte] += (1 << bit);
        }
      }
    }
    return str;
  }

  static void Decompress(absl::string_view compressed,
                         absl::Span<float> buffer) {
    const int num_bytes = (buffer.size() + kBitsPerByte - 1) / kBitsPerByte;
    absl::c_fill(buffer, 0);
    SPIEL_CHECK_EQ(compressed.size(), num_bytes + kNumHeaderBytes);
    for (int byte = 0; byte < num_bytes; ++byte) {
      for (int bit = 0; bit < kBitsPerByte; ++bit) {
        if (compressed[kNumHeaderBytes + byte] & (1 << bit)) {
          buffer[byte * kBitsPerByte + bit] = 1;
        }
      }
    }
  }
};

// No compression.
struct NoCompress {
  static std::string Compress(absl::Span<const float> buffer) {
    const int num_bytes = sizeof(float) * buffer.size();
    std::string str(num_bytes + 1, '\0');
    str[0] = kCompressionNone;
    memcpy(&str[kNumHeaderBytes], &buffer[0], num_bytes);
    return str;
  }

  static void Decompress(absl::string_view compressed,
                         absl::Span<float> buffer) {
    const int num_bytes = sizeof(float) * buffer.size();
    SPIEL_CHECK_EQ(compressed.size(), num_bytes + kNumHeaderBytes);
    memcpy(&buffer[0], &compressed[kNumHeaderBytes], num_bytes);
  }
};

std::string Observation::Compress() const {
  const bool data_is_binary =
      absl::c_all_of(buffer_, [](float x) { return x == 0 || x == 1; });
  return data_is_binary ? BinaryCompress::Compress(buffer_)
                        : NoCompress::Compress(buffer_);
}

void Observation::Decompress(absl::string_view compressed) {
  SPIEL_CHECK_GT(compressed.size(), 0);
  switch (compressed[0]) {
    case kCompressionBinary:
      return BinaryCompress::Decompress(compressed, absl::MakeSpan(buffer_));
    case kCompressionNone:
      return NoCompress::Decompress(compressed, absl::MakeSpan(buffer_));
    default:
      SpielFatalError(absl::StrCat("Unrecognized compression scheme in '",
                                   compressed, "'"));
  }
}

}  // namespace open_spiel
