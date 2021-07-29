#include <string>
#include <vector>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <fcntl.h>

namespace subprocess {

class popen {
 public:
  popen(const std::vector<std::string>& args) {
    if (pipe(in_pipe) == -1 ||
        pipe(out_pipe) == -1 ||
        pipe(err_pipe) == -1) {
      throw std::system_error(errno, std::system_category());
    }

    fcntl(in_pipe[WRITE], F_SETFL, O_NONBLOCK);
    fcntl(out_pipe[READ], F_SETFL, O_NONBLOCK);
    fcntl(err_pipe[READ], F_SETFL, O_NONBLOCK);

    // Clone the calling process, creating an exact copy.
    // Return -1 for errors, 0 to the new process,
    // and the process ID of the new process to the old process.
    pid = fork();
    if (pid == 0) child(args);

    // The code below will be executed only by parent.
    close(in_pipe[READ]);
    close(out_pipe[WRITE]);
    close(err_pipe[WRITE]);
  }

  ~popen() {}

  int stdin() { return in_pipe[WRITE]; };
  int stdout() { return out_pipe[READ]; }
  int stderr() { return err_pipe[READ]; };

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
};

} // namespace subprocess
