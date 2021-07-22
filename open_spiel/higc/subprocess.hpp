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

namespace {

// Provides a layer of compatibility for C/POSIX.
template <typename _CharT, typename _Traits = std::char_traits<_CharT> >
class stdio_filebuf : public std::basic_filebuf<_CharT, _Traits> {
 public:
  /**
   *  @param  fd   An open file descriptor.
   *  @param  mode Same meaning as in a standard filebuf.
   *  @param  size Optimal or preferred size of internal buffer,
   *               in chars.
   *
   *  This constructor associates a file stream buffer with an open
   *  POSIX file descriptor. The file descriptor will be automatically
   *  closed when the stdio_filebuf is closed/destroyed.
  */
  stdio_filebuf(int fd, std::ios_base::openmode mode,
                size_t size = static_cast<size_t>(BUFSIZ)) {
    this->_M_file.sys_open(fd, mode);
    if (this->is_open()) {
      this->_M_mode = mode;
      this->_M_buf_size = size;
      this->_M_allocate_internal_buffer();
      this->_M_reading = false;
      this->_M_writing = false;
      this->_M_set_buffer(-1);
    }
  }
  ~stdio_filebuf() = default;
  stdio_filebuf(stdio_filebuf&&) = default;
  stdio_filebuf& operator=(stdio_filebuf&&) = default;
  // The underlying file descriptor.
  int fd() { return this->_M_file.fd(); }
  // The underlying FILE*
  FILE* file() { return this->_M_file.file(); }
};

} // namespace

namespace subprocess {

class popen {
 public:

  popen(const std::string& cmd, std::vector<std::string> argv)
      : in_filebuf(nullptr),
        out_filebuf(nullptr),
        err_filebuf(nullptr),
        in_stream(nullptr),
        out_stream(nullptr),
        err_stream(nullptr) {
    if (pipe(in_pipe) == -1 ||
        pipe(out_pipe) == -1 ||
        pipe(err_pipe) == -1) {
      throw std::system_error(errno, std::system_category());
    }

    run(cmd, argv);
  }

  popen(const std::string& cmd,
        std::vector<std::string> argv,
        std::ostream& pipe_stdout)
      : in_filebuf(nullptr),
        out_filebuf(nullptr),
        err_filebuf(nullptr),
        in_stream(nullptr),
        out_stream(nullptr),
        err_stream(nullptr) {
    auto filebuf = dynamic_cast<stdio_filebuf<char>*>(pipe_stdout.rdbuf());
    out_pipe[READ] = -1;
    out_pipe[WRITE] = filebuf->fd();

    if (pipe(in_pipe) == -1 ||
        pipe(err_pipe) == -1) {
      throw std::system_error(errno, std::system_category());
    }

    run(cmd, argv);
  }

  ~popen() {
    delete in_filebuf;
    delete in_stream;
    if (out_filebuf != nullptr) delete out_filebuf;
    if (out_stream != nullptr) delete out_stream;
    delete err_filebuf;
    delete err_stream;
  }

  std::ostream& stdin() { return *in_stream; };

  std::istream& stdout() {
    if (out_stream == nullptr)
      throw std::system_error(EBADF,
                              std::system_category());
    return *out_stream;
  };

  std::istream& stderr() { return *err_stream; };

  int wait() {
    int status = 0;
    waitpid(pid, &status, 0);
    return WEXITSTATUS(status);
  };

  void close() {
    in_filebuf->close();
  }

 private:

  enum ends_of_pipe { READ = 0, WRITE = 1 };

  struct raii_char_str {
    raii_char_str(std::string s) : buf(s.c_str(), s.c_str() + s.size() + 1) {};
    operator char*() const { return &buf[0]; };
    mutable std::vector<char> buf;
  };

  void run(const std::string& cmd, std::vector<std::string> argv) {
    argv.insert(argv.begin(), cmd);

    pid = ::fork();

    if (pid == 0) child(argv);

    ::close(in_pipe[READ]);
    ::close(out_pipe[WRITE]);
    ::close(err_pipe[WRITE]);

    in_filebuf = new stdio_filebuf<char>(in_pipe[WRITE], std::ios_base::out, 1);
    in_stream = new std::ostream(in_filebuf);

    if (out_pipe[READ] != -1) {
      out_filebuf = new stdio_filebuf<char>(out_pipe[READ], std::ios_base::in, 1);
      out_stream = new std::istream(out_filebuf);
    }

    err_filebuf = new stdio_filebuf<char>(err_pipe[READ], std::ios_base::in, 1);
    err_stream = new std::istream(err_filebuf);
  }

  void child(const std::vector<std::string>& argv) {
    if (dup2(in_pipe[READ], STDIN_FILENO) == -1 ||
        dup2(out_pipe[WRITE], STDOUT_FILENO) == -1 ||
        dup2(err_pipe[WRITE], STDERR_FILENO) == -1) {
      std::perror("subprocess: dup2() failed");
      return;
    }

    ::close(in_pipe[READ]);
    ::close(in_pipe[WRITE]);
    if (out_pipe[READ] != -1) ::close(out_pipe[READ]);
    ::close(out_pipe[WRITE]);
    ::close(err_pipe[READ]);
    ::close(err_pipe[WRITE]);

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

  stdio_filebuf<char>* in_filebuf;
  stdio_filebuf<char>* out_filebuf;
  stdio_filebuf<char>* err_filebuf;

  std::ostream* in_stream;
  std::istream* out_stream;
  std::istream* err_stream;
};

} // namespace: subprocess