#include <iostream>
#include <string>
#include "fdstream.hpp"

// for FILE, popen(), pclose():
#include <stdio.h>

void popen_test (std::string const& command)
{
  FILE* fp;

  // open pipe to read from
  if ((fp=popen(command.c_str(),"r")) == NULL) {
    throw "popen() failed";
  }

  // and initialize input stream to read from it
  boost::fdistream in(fileno(fp));

  // print all characters with indent
  std::cout << "output of " << command << ":\n";
  char c;
  while (in.get(c)) {
    std::cout.put(c);
    if (c == '\n') {
      std::cout.put('>');
      std::cout.put(' ');
    }
  }
  std::cout.put('\n');

  pclose(fp);
}

int main()
{
  popen_test("ls -l");
  popen_test("dir");
  popen_test("date");
}

