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


#ifndef OPEN_SPIEL_HIGC_SUBPROCESS_
#define OPEN_SPIEL_HIGC_SUBPROCESS_

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace open_spiel {
namespace higc {

// Automatically handle error cases without bloating the code.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)
#define RUN(fn, ...)                                 \
  if (fn(__VA_ARGS__) == -1) {                       \
    std::perror("subprocess: " #fn "failed at " AT); \
    std::exit(1);                                    \
  }

class Subprocess {
  pid_t child_pid_;
  int in_pipe_[2];
  int out_pipe_[2];
  int err_pipe_[2];

 public:
  Subprocess(const std::string& shell_command, bool should_block = false) {
    // Create pipes for input/output/error communication.
    RUN(pipe, in_pipe_);
    RUN(pipe, out_pipe_);
    RUN(pipe, err_pipe_);

    // Make sure to set all file descriptors of the pipes to be non-blocking.
    if (!should_block) {
      RUN(fcntl, in_pipe_[WRITE], F_SETFL, O_NONBLOCK);
      RUN(fcntl, out_pipe_[READ], F_SETFL, O_NONBLOCK);
      RUN(fcntl, err_pipe_[READ], F_SETFL, O_NONBLOCK);
    }

    // Clone the calling process, creating an exact copy.
    // Returns -1 for errors, 0 to the new process,
    // and the process ID of the new process to the old process.
    child_pid_ = fork();
    if (child_pid_ == -1) {
      std::perror("subprocess: fork failed");
      std::exit(1);
    }
    if (child_pid_ == 0) child(shell_command);

    // The code below will be executed only by parent.
    RUN(close, in_pipe_[READ]);
    RUN(close, out_pipe_[WRITE]);
    RUN(close, err_pipe_[WRITE]);
  }

  int stdin() { return in_pipe_[WRITE]; }
  int stdout() { return out_pipe_[READ]; }
  int stderr() { return err_pipe_[READ]; }
  pid_t child_pid() const { return child_pid_; }

 private:
  enum ends_of_pipe { READ = 0, WRITE = 1 };

  // Code run only by the child process.
  void child(const std::string& shell_command) {
    // Connect the pipe ends to STDIO for the child.
    RUN(dup2, in_pipe_[READ], STDIN_FILENO)
    RUN(dup2, out_pipe_[WRITE], STDOUT_FILENO)
    RUN(dup2, err_pipe_[WRITE], STDERR_FILENO)

    // Close all parent pipes, as they have been rerouted.
    for (auto& pipe : {in_pipe_, out_pipe_, err_pipe_}) {
      RUN(close, pipe[READ]);
      RUN(close, pipe[WRITE]);
    }

    std::vector<std::string> cargs;
    cargs.push_back("/bin/sh");
    cargs.push_back("-c");
    std::string command = shell_command;  // Drop const.
    cargs.push_back(command.data());

    char **argv = new char* [cargs.size()+1];
    argv[cargs.size()] = nullptr;
    for (int i = 0; i < cargs.size(); ++i) {
      argv[i] = const_cast<char*>(cargs[i].c_str());
    }

    // Execute the command.
    RUN(execvp, argv[0], argv);
    delete [] argv;
  }
};

#undef RUN
#undef AT
#undef STRINGIFY
#undef TOSTRING

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_SUBPROCESS_
