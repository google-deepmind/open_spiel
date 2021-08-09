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


#include <thread>
#include <mutex>
#include <exception>
#include <filesystem>
#include <unistd.h>

#include "open_spiel/spiel.h"
#include "open_spiel/higc/base64.h"
#include "open_spiel/higc/referee.h"
#include "open_spiel/higc/utils.h"

namespace open_spiel {
namespace higc {

void BotChannel::StartRead(int time_limit) {
  SPIEL_CHECK_FALSE(shutdown_);
  SPIEL_CHECK_TRUE(wait_for_message_);
  SPIEL_CHECK_EQ(comm_error_, 0);
  time_limit_ = time_limit;
  time_out_ = false;
  cancel_read_ = false;
  wait_for_message_ = false;
}

void BotChannel::CancelReadBlocking() {
  cancel_read_ = true;
  std::lock_guard<std::mutex> lock(mx_read);  // Wait until reading is cancelled.
}

void BotChannel::Write(const std::string& s) {
  if (comm_error_ < 0) return;  // Do not write anything anymore after error.

  int written_bytes = write(in(), s.c_str(), s.size());
  if (written_bytes == -1) {
    comm_error_ = -1;
  } else if (written_bytes != s.size()) {
    comm_error_ = errno;
  }
}

void BotChannel::Write(char c) {
  if (comm_error_ < 0) return;  // Do not write anything anymore after error.

  int written_bytes = write(in(), &c, 1);
  if (written_bytes == -1) {
    comm_error_ = -1;
  } else if (written_bytes != 1) {
    comm_error_ = errno;
  }
}

void BotChannel::ShutDown() {
  shutdown_ = true;
  cancel_read_ = true;
}

std::unique_ptr<BotChannel> MakeBotChannel(int bot_index,
                                           std::string executable) {
  auto popen = std::make_unique<Subprocess>(
      std::vector<std::string>{executable});
  return std::make_unique<BotChannel>(bot_index, std::move(popen));
}


}  // namespace higc
}  // namespace open_spiel
