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

#include "open_spiel/observer.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

SpanTensor ContiguousAllocator::Get(absl::string_view name,
                                    const absl::InlinedVector<int, 4>& shape) {
  const int size = absl::c_accumulate(shape, 1, std::multiplies<int>());
  SPIEL_DCHECK_LE(offset_, data_.size());
  auto buffer = data_.subspan(offset_, size);
  offset_ += size;
  return SpanTensor(SpanTensorInfo(name, shape), buffer);
}

namespace {

class InformationStateObserver : public Observer {
 public:
  InformationStateObserver(const Game& game)
      : Observer(
            /*has_string=*/game.GetType().provides_information_state_string,
            /*has_tensor=*/game.GetType().provides_information_state_tensor),
        size_(has_tensor_ ? game.InformationStateTensorSize() : 0) {
    if (has_tensor_) {
      auto shape = game.InformationStateTensorShape();
      shape_.assign(shape.begin(), shape.end());
    }
  }
  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override {
    auto tensor = allocator->Get("info_state", shape_);
    state.InformationStateTensor(player, tensor.data());
  }

  std::string StringFrom(const State& state, int player) const override {
    return state.InformationStateString(player);
  }

 private:
  absl::InlinedVector<int, 4> shape_;
  int size_;
};

class DefaultObserver : public Observer {
 public:
  DefaultObserver(const Game& game)
      : Observer(/*has_string=*/
                 game.GetType().provides_observation_string,
                 /*has_tensor=*/game.GetType().provides_observation_tensor),
        size_(has_tensor_ ? game.ObservationTensorSize() : 0) {
    if (has_tensor_) {
      auto shape = game.ObservationTensorShape();
      shape_.assign(shape.begin(), shape.end());
    }
  }

  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override {
    SPIEL_CHECK_TRUE(has_tensor_);
    auto tensor = allocator->Get("observation", shape_);
    state.ObservationTensor(player, tensor.data());
  }

  std::string StringFrom(const State& state, int player) const override {
    return state.ObservationString(player);
  }

 private:
  absl::InlinedVector<int, 4> shape_;
  int size_;
};

std::string PrivateInfoTypeToString(const PrivateInfoType& type) {
  if (type == PrivateInfoType::kNone) return "kNone";
  if (type == PrivateInfoType::kSinglePlayer) return "kSinglePlayer";
  if (type == PrivateInfoType::kAllPlayers) return "kAllPlayers";
  SpielFatalError("Unknown PrivateInfoType!");
}

std::string IIGObservationTypeToString(const IIGObservationType& obs_type) {
  return absl::StrCat(
      "IIGObservationType",
      "{perfect_recall=", obs_type.perfect_recall ? "true" : "false",
      ", public_info=", obs_type.public_info ? "true" : "false",
      ", private_info=", PrivateInfoTypeToString(obs_type.private_info), "}");
}

// A dummy class that provides private observations for games with perfect
// information. As these games have no private information, we return dummy
// values.
class NoPrivateObserver : public Observer {
 public:
  NoPrivateObserver(const Game& game)
      : Observer(/*has_string=*/true, /*has_tensor=*/true) {}
  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override {}
  std::string StringFrom(const State& state, int player) const override {
    return "";
  }
};
}  // namespace

std::shared_ptr<Observer> Game::MakeRegisteredObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const ObservationParams& params) const {
  auto iter = params.find("name");
  if (iter == params.end()) {
    SpielFatalError("A 'name' parameter is expected to create a registered "
                    "observer");
  }
  auto name = iter->second.string_value();
  return ObserverRegisterer::CreateByName(name, *this, iig_obs_type, params);
}

std::shared_ptr<Observer> Game::MakeBuiltInObserver(
    absl::optional<IIGObservationType> iig_obs_type) const {
  if (!iig_obs_type) return absl::make_unique<DefaultObserver>(*this);

  const bool perfect_info_game =
      game_type_.information == GameType::Information::kPerfectInformation;

  // Perfect information games can provide public information regardless
  // of requested PrivateInfoType (as they have no private information).
  if (perfect_info_game) {
    // Handle the dummy case, where we do not use any public information.
    // The game will just have empty private observations.
    if (!iig_obs_type->public_info)
      return absl::make_unique<NoPrivateObserver>(*this);
    if (game_type_.provides_information_state() && iig_obs_type->perfect_recall)
      return absl::make_unique<InformationStateObserver>(*this);
    if (game_type_.provides_observation() && !iig_obs_type->perfect_recall)
      return absl::make_unique<DefaultObserver>(*this);
  }

  // TODO(author11) Reinstate this check
  // SPIEL_CHECK_EQ(GetType().information,
  //                GameType::Information::kImperfectInformation);
  if (iig_obs_type.value() == kDefaultObsType) {
    if (game_type_.provides_observation())
      return absl::make_unique<DefaultObserver>(*this);
  }
  if (iig_obs_type.value() == kInfoStateObsType) {
    if (game_type_.provides_information_state())
      return absl::make_unique<InformationStateObserver>(*this);
  }
  SpielFatalError(absl::StrCat("Requested Observer type not available: ",
                               IIGObservationTypeToString(*iig_obs_type)));
}

std::shared_ptr<Observer> Game::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const ObservationParams& params) const {
  // This implementation falls back to the orginal information state and
  // observation methods in case of empty parameters and otherwise creates
  // a registered observer based on its name.
  // New games can register observers which can be selected by name, or override
  // MakeObserver to return a game-specific observer.
  if (params.empty()) {
    return MakeBuiltInObserver(iig_obs_type);
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

SpanTensor TrackingVectorAllocator::Get(
    absl::string_view name, const absl::InlinedVector<int, 4>& shape) {
  SPIEL_DCHECK_TRUE(IsNameUnique(name));

  SpanTensorInfo info(name, shape);
  tensors_info_.push_back(info);

  const int offset = data_.size();
  const int size = info.size();
  data_.resize(offset + size);
  return SpanTensor(std::move(info),
                    absl::MakeSpan(data_).subspan(offset, size));
}

bool TrackingVectorAllocator::IsNameUnique(absl::string_view name) const {
  for (const SpanTensorInfo& info : tensors_info_) {
    if (info.name() == name) {
      return false;
    }
  }
  return true;
}

Observation::Observation(const Game& game, std::shared_ptr<Observer> observer)
    : observer_(std::move(observer)) {
  // Get an observation of the initial state to set up.
  if (HasTensor()) {
    auto state = game.NewInitialState();
    TrackingVectorAllocator allocator;
    observer_->WriteTensor(*state, /*player=*/0, &allocator);
    buffer_ = allocator.data();
    tensors_info_ = allocator.tensors_info();
  }
}

void Observation::SetFrom(const State& state, int player) {
  ContiguousAllocator allocator(absl::MakeSpan(buffer_));
  observer_->WriteTensor(state, player, &allocator);
}

std::vector<SpanTensor> Observation::tensors() {
  std::vector<SpanTensor> result;
  result.reserve(tensors_info_.size());
  int offset = 0;
  for (const SpanTensorInfo& info : tensors_info_) {
    const int size = info.size();
    result.emplace_back(info, absl::MakeSpan(buffer_).subspan(offset, size));
    offset += size;
  }
  return result;
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

    for (int i = 0; i < buffer.size(); ++i) {
      if (buffer[i]) {
        const int byte = i / kBitsPerByte;
        const int bit = i % kBitsPerByte;
        str[kNumHeaderBytes + byte] += (1 << bit);
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

bool IIGObservationType::operator==(const IIGObservationType& other) {
  return public_info == other.public_info &&
         perfect_recall == other.perfect_recall &&
         private_info == other.private_info;
}

ObserverRegisterer::ObserverRegisterer(const std::string& game_name,
                                       const std::string& observer_name,
                                       CreateFunc creator) {
  RegisterObserver(game_name, observer_name, creator);
}

void ObserverRegisterer::RegisterObserver(const std::string& game_name,
                                          const std::string& observer_name,
                                          CreateFunc creator) {
  auto key = std::pair(game_name, observer_name);
  if (observers().find(key) != observers().end()) {
    SpielFatalError(absl::StrCat("Duplicate observer '", key.second, "'",
                                 " for game '", key.first, "'"));
  }
  observers()[key] = creator;
}

std::shared_ptr<Observer> ObserverRegisterer::CreateByName(
    const std::string& observer_name,
    const Game& game,
    absl::optional<IIGObservationType> iig_obs_type,
    const ObservationParams& params) {
  auto key = std::pair(game.GetType().short_name, observer_name);
  auto it = observers().find(key);
  if (it == observers().end()) {
    SpielFatalError(absl::StrCat("No observer '", key.second, "'",
                                 " found for game '", key.first, "'"));
  }
  return it->second(game, iig_obs_type, params);
}

std::vector<float> TensorFromObserver(const State& state,
                                      const Observer& observer) {
  TrackingVectorAllocator allocator;
  observer.WriteTensor(state, /*player=*/state.CurrentPlayer(), &allocator);
  return std::move(allocator.data());
}

std::vector<int> ObserverTensorShape(const State& state,
                                     const Observer& observer) {
  TrackingVectorAllocator allocator;
  observer.WriteTensor(state, /*player=*/0, &allocator);
  if (allocator.tensors_info().size() == 1) {
    return allocator.tensors_info().front().vector_shape();
  } else {
    return {static_cast<int>(allocator.data().size())};
  }
}

}  // namespace open_spiel
