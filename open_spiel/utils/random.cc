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


#include "open_spiel/utils/random.h"

namespace open_spiel {

namespace {
std::uniform_real_distribution<double> uniformDist;
}  // namespace

double RandomMT::RandomUniform() { return uniformDist(generator_); }

double RandomFixedSequence::RandomUniform() {
  double v = values_[position_];
  if (++position_ == values_.size()) position_ = 0;
  return v;
}

}  // namespace open_spiel
