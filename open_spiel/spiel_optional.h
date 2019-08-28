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

#ifndef THIRD_PARTY_OPEN_SPIEL_SPIEL_OPTIONAL_H_
#define THIRD_PARTY_OPEN_SPIEL_SPIEL_OPTIONAL_H_

#include <utility>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// A type convertible to any Optional<> instantiation to represent an optional
// without value.
struct Nullopt {
  // It must not be default-constructible, so we use a dummy parameter for the
  // constructor.
  struct Init {};
  static Init kInit;
  explicit constexpr Nullopt(const Init&) {}
};

extern const Nullopt kNullopt;

// Optional type more or less compatible with std::optional, boost::optional,
// and absl::optional.
template <typename T>
class Optional {
 public:
  using ValueType = T;

  // Creates an empty Optional (no value set).
  Optional() : has_value_(false) {}
  Optional(const Nullopt&) : has_value_(false) {}

  template <typename Tref>
  void SetValue(Tref&& new_value) {
    has_value_ = true;
    value_ = std::forward<Tref>(new_value);
  }

  void CopyFrom(const Optional& other) {
    has_value_ = other.has_value_;
    if (has_value_) {
      SetValue(other.value_);
    }
  }

  void MoveFrom(Optional&& other) {
    has_value_ = other.has_value_;
    if (has_value_) {
      SetValue(std::move(other.value_));
    }
  }

  Optional(const Optional<T>& other) { CopyFrom(other); }
  Optional(Optional<T>&& other) { MoveFrom(std::move(other)); }

  Optional(const T& value) { SetValue(value); }
  Optional(T&& value) { SetValue(std::move(value)); }

  Optional& operator=(const Optional<T>& other) {
    CopyFrom(other);
    return *this;
  }
  Optional& operator=(Optional<T>&& other) {
    MoveFrom(std::move(other));
    return *this;
  }

  Optional& operator=(const T& new_value) {
    SetValue(new_value);
    return *this;
  }
  Optional& operator=(T&& new_value) {
    SetValue(std::move(new_value));
    return *this;
  }

  explicit operator bool() const { return has_value_; }

  T& operator*() {
    SPIEL_CHECK_TRUE(has_value_);
    return value_;
  }
  const T& operator*() const {
    SPIEL_CHECK_TRUE(has_value_);
    return value_;
  }

  T* operator->() { return &(this->operator*()); }
  const T* operator->() const { return &(this->operator*()); }

 private:
  bool has_value_;
  T value_;
};

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_SPIEL_OPTIONAL_H_
