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

#ifndef OPEN_SPIEL_HIGC_CHANNEL_
#define OPEN_SPIEL_HIGC_CHANNEL_

#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "open_spiel/higc/subprocess.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace higc {

constexpr int kMaxLineLength = 1024;

// Communication channel with the bot.
class BotChannel {
 public:
  BotChannel(int bot_index, std::unique_ptr<Subprocess> popen)
      : bot_index_(bot_index), popen_(std::move(popen)) {
    response_.reserve(kMaxLineLength);
    buf_.reserve(kMaxLineLength);
  }
  int in() { return popen_->stdin(); }
  int out() { return popen_->stdout(); }
  int err() { return popen_->stderr(); }

  void StartRead(int time_limit);
  void CancelReadBlocking();
  void ShutDown();

  // Was line successfully read into response() yet?
  bool ReadLineAsync();
  void Write(const std::string& s);
  void Write(char c);

  bool is_waiting_for_referee() const { return wait_for_referee_; }
  bool has_read() const { return !response_.empty(); }
  bool is_time_out() const { return time_out_; }
  int comm_error() const { return comm_error_; }
  std::string response() const { return response_; }

 private:
  // Did some communication error occur? Store an error code returned
  // by `errno` for write() or read() functions.
  // See also <asm-generic/errno.h> for a list of error codes.
  int comm_error_ = 0;

  int bot_index_;
  std::unique_ptr<Subprocess> popen_;
  std::string response_;  // A complete line response.
  std::string buf_;       // Incomplete response buffer.
  bool time_out_ = false;

  std::atomic<bool> shutdown_ = false;
  std::atomic<bool> wait_for_referee_ = true;
  int time_limit_ = 0;
  bool cancel_read_ = false;
  std::mutex mx_read;

  // Reading thread loops.
  friend void ReadLineFromChannelStdout(BotChannel* c);
  friend void ReadLineFromChannelStderr(BotChannel* c);
};

std::unique_ptr<BotChannel> MakeBotChannel(int bot_index,
                                           const std::string& shell_command);

void ReadLineFromChannelStdout(BotChannel* c);
void ReadLineFromChannelStderr(BotChannel* c);

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_CHANNEL_
