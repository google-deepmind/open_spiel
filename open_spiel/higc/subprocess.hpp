//
// subprocess C++ library - https://github.com/tsaarni/cpp-subprocess
//
// The MIT License (MIT)
//
// Copyright (c) 2015 Tero Saarni
//

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/wait.h>
#include <sys/prctl.h>
#include "fdstream.hpp"

namespace subprocess {

class popen {
 public:

  popen(const std::string& cmd, std::vector<std::string> argv)
      : in_stream(nullptr),
        out_stream(nullptr),
        err_stream(nullptr) {
    if (pipe(in_pipe) == -1 ||
        pipe(out_pipe) == -1 ||
        pipe(err_pipe) == -1) {
      throw std::system_error(errno, std::system_category());
    }

    run(cmd, argv);
  }

  ~popen() {
    delete in_stream;
    if (out_stream != nullptr) delete out_stream;
    delete err_stream;
  }

  std::ostream& stdin() { return *in_stream; };

  std::istream& stdout() {
    if (out_stream == nullptr) {
      throw std::system_error(EBADF, std::system_category());
    }
    return *out_stream;
  };

  std::istream& stderr() { return *err_stream; };

  int wait() {
    int status = 0;
    waitpid(pid, &status, 0);
    return WEXITSTATUS(status);
  };

 private:

  enum ends_of_pipe { READ = 0, WRITE = 1 };

  struct raii_char_str {
    raii_char_str(std::string s) : buf(s.c_str(), s.c_str() + s.size() + 1) {};
    operator char*() const { return &buf[0]; };
    mutable std::vector<char> buf;
  };

  void run(const std::string& cmd, std::vector<std::string> argv) {
    argv.insert(argv.begin(), cmd);

    // Clone the calling process, creating an exact copy.
    // Return -1 for errors, 0 to the new process,
    // and the process ID of the new process to the old process.
    pid = fork();
    if (pid == 0) child(argv);

    // The code below will be executed only by parent.

    close(in_pipe[READ]);
    close(out_pipe[WRITE]);
    close(err_pipe[WRITE]);

    in_stream = new boost::fdostream(in_pipe[WRITE]);
    if (out_pipe[READ] != -1) {
      out_stream = new boost::fdistream(out_pipe[READ]);
    }
    err_stream = new boost::fdistream(err_pipe[READ]);
  }

  // Code run only by the child process.
  void child(const std::vector<std::string>& argv) {
    if (dup2(in_pipe[READ], STDIN_FILENO) == -1 ||
        dup2(out_pipe[WRITE], STDOUT_FILENO) == -1 ||
        dup2(err_pipe[WRITE], STDERR_FILENO) == -1) {
      std::perror("subprocess: dup2() failed");
      return;
    }

    // Ask kernel to deliver SIGTERM in case the parent dies.
    prctl(PR_SET_PDEATHSIG, SIGTERM);

    close(in_pipe[READ]);
    close(in_pipe[WRITE]);
    if (out_pipe[READ] != -1) close(out_pipe[READ]);
    close(out_pipe[WRITE]);
    close(err_pipe[READ]);
    close(err_pipe[WRITE]);

    std::vector<raii_char_str> real_args(argv.begin(), argv.end());
    std::vector<char*> cargs(real_args.begin(), real_args.end());
    cargs.push_back(nullptr);

    if (execvp(cargs[0], &cargs[0]) == -1) {
      std::perror("subprocess: execvp() failed");
      return;
    }
  }

  pid_t pid;

  int in_pipe[2];
  int out_pipe[2];
  int err_pipe[2];

  std::ostream* in_stream;
  std::istream* out_stream;
  std::istream* err_stream;
};

} // namespace subprocess
