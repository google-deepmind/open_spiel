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


#ifndef OPEN_SPIEL_HIGC_SUBPROCESS_
#define OPEN_SPIEL_HIGC_SUBPROCESS_

#include <string>
#include <vector>
#include <sys/wait.h>
#include <fcntl.h>

namespace open_spiel {
namespace higc {

// Automatically handle error cases without bloating the code.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)
#define RUN(fn, ...)                                                           \
  if(fn(__VA_ARGS__) == -1) {                                                  \
    std::perror("subprocess: " #fn "failed at " AT);                           \
    std::exit(1);                                                              \
  }

class Subprocess {
  pid_t child_pid;
  int in_pipe[2];
  int out_pipe[2];
  int err_pipe[2];
 public:
  Subprocess(std::vector<std::string> args) {
    // Create pipes for input/output/error communication.
    RUN(pipe, in_pipe);
    RUN(pipe, out_pipe);
    RUN(pipe, err_pipe);

    // Make sure to set all file descriptors of the pipes to be non-blocking.
    RUN(fcntl, in_pipe[WRITE], F_SETFL, O_NONBLOCK);
    RUN(fcntl, out_pipe[READ], F_SETFL, O_NONBLOCK);
    RUN(fcntl, err_pipe[READ], F_SETFL, O_NONBLOCK);

    // Clone the calling process, creating an exact copy.
    // Returns -1 for errors, 0 to the new process,
    // and the process ID of the new process to the old process.
    child_pid = fork();
    if (child_pid == -1) {
      std::perror("subprocess: fork failed");
      std::exit(1);
    }
    if (child_pid == 0) child(args);

    // The code below will be executed only by parent.
    RUN(close, in_pipe[READ]);
    RUN(close, out_pipe[WRITE]);
    RUN(close, err_pipe[WRITE]);
  }

  int stdin() { return in_pipe[WRITE]; };
  int stdout() { return out_pipe[READ]; }
  int stderr() { return err_pipe[READ]; };

 private:
  enum ends_of_pipe { READ = 0, WRITE = 1 };

  // Code run only by the child process.
  void child(std::vector<std::string>& argv) {
    // Connect the pipe ends to STDIO for the child.
    RUN(dup2, in_pipe[READ], STDIN_FILENO)
    RUN(dup2, out_pipe[WRITE], STDOUT_FILENO)
    RUN(dup2, err_pipe[WRITE], STDERR_FILENO)

    // Close all parent pipes, as they have been rerouted.
    for (auto& pipe : {in_pipe, out_pipe, err_pipe}) {
      RUN(close, pipe[READ]);
      RUN(close, pipe[WRITE]);
    }

    // Prepare data format valid for execvp.
    std::vector<char*> cargs;
    for (std::string& arg : argv) cargs.push_back(arg.data());
    cargs.push_back(nullptr);

    // Execute the command.
    RUN(execvp, cargs[0], &cargs[0]);
  }
};

#undef RUN
#undef AT
#undef STRINGIFY
#undef TOSTRING

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_SUBPROCESS_
