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

#ifndef OPEN_SPIEL_HIGC_UTILS_
#define OPEN_SPIEL_HIGC_UTILS_

#include <chrono>
#include <iostream>

namespace open_spiel {
namespace higc {

void sleep_ms(int ms);
int time_elapsed(const std::chrono::time_point<std::chrono::system_clock>& start);

// Non-blocking reading of line from input stream.
// If a whole line is read, it returns true and the line is saved into out_line.
// If only a partial result is available (without the end of the line),
// it is saved into a provided external buffer.
bool getline_async(std::istream& is, std::string& line_out, std::string& buf);

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_UTILS_
