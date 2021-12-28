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


#include "open_spiel/higc/channel.h"

#include <unistd.h>

#include <exception>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "open_spiel/higc/utils.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace higc {

void BotChannel::StartRead(int time_limit) {
  SPIEL_CHECK_FALSE(shutdown_);
  SPIEL_CHECK_TRUE(wait_for_referee_);
  time_limit_ = time_limit;
  time_out_ = false;
  cancel_read_ = false;
  wait_for_referee_ = false;
}

void BotChannel::CancelReadBlocking() {
  cancel_read_ = true;
  std::lock_guard<std::mutex> lock(
      mx_read);  // Wait until reading is cancelled.
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
  if (comm_error_ != 0) return;  // Do not write anything anymore after error.

  int written_bytes = write(in(), &c, 1);
  if (written_bytes == -1) {
    comm_error_ = -1;
  } else if (written_bytes != 1) {
    comm_error_ = errno;
  }
}

bool BotChannel::ReadLineAsync() {
  int chars_read = 0;
  bool line_read = false;
  response_.clear();

  do {
    // Read a single character (non-blocking).
    char c;
    chars_read = read(out(), &c, 1);
    if (chars_read == 1) {
      if (c == '\n') {
        response_ = buf_;
        buf_ = "";
        line_read = true;
      } else {
        buf_.append(1, c);
      }
    }
  } while (chars_read > 0 && !line_read && buf_.size() < kMaxLineLength);

  if (buf_.size() >= kMaxLineLength) {
    comm_error_ = EMSGSIZE;
  }

  return line_read;
}

void BotChannel::ShutDown() {
  shutdown_ = true;
  cancel_read_ = true;
}

std::unique_ptr<BotChannel> MakeBotChannel(int bot_index,
                                           const std::string& shell_command) {
  auto popen = std::make_unique<Subprocess>(shell_command);
  return std::make_unique<BotChannel>(bot_index, std::move(popen));
}

// Read a response message from the bot in a separate thread.
void ReadLineFromChannelStdout(BotChannel* c) {
  SPIEL_CHECK_TRUE(c);
  // Outer loop for repeated match playing.
  while (!c->shutdown_) {
    // Wait until referee sends a message to the bot.
    while (c->wait_for_referee_) {
      sleep_ms(1);
      if (c->shutdown_) return;
    }

    {
      std::lock_guard<std::mutex> lock(c->mx_read);

      auto time_start = std::chrono::system_clock::now();
      while (  // Keep reading the current line,
          !c->ReadLineAsync()
          // if there is no error,
          && c->comm_error() == 0
          // no timeout,
          && !(c->time_out_ = (time_elapsed(time_start) > c->time_limit_))
          // and no reading cancellation.
          && !c->cancel_read_) {
        sleep_ms(1);
        if (c->shutdown_) return;
      }

      c->wait_for_referee_ = true;
    }
  }
}

// Global cerr mutex.
std::mutex mx_cerr;

// Read a stderr output from the bot in a separate thread.
// Forward all bot's stderr to the referee's stderr.
// Makes sure that lines are not tangled together by using a mutex.
void ReadLineFromChannelStderr(BotChannel* c) {
  SPIEL_CHECK_TRUE(c);
  int read_bytes;
  std::array<char, 1024> buf;
  while (!c->shutdown_) {
    read_bytes = read(c->err(), &buf[0], 1024);
    if (read_bytes > 0) {
      std::lock_guard<std::mutex> lock(mx_cerr);  // Have nice stderr outputs.
      std::cerr << "Bot#" << c->bot_index_ << ": ";
      for (int i = 0; i < read_bytes; ++i) std::cerr << buf[i];
      std::cerr << std::flush;
    }
    sleep_ms(1);
  }
}

}  // namespace higc
}  // namespace open_spiel
