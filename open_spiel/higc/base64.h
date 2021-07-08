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

#ifndef OPEN_SPIEL_HIGC_BASE64_
#define OPEN_SPIEL_HIGC_BASE64_

#include <iostream>
#include <string>
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"

namespace open_spiel {
namespace higc {

void base64_encode(std::ostream& os, char const* buf, size_t len);
std::string base64_decode(absl::string_view encoded_string);

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_BASE64_
