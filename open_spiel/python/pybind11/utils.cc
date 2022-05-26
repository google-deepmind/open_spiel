#include "open_spiel/python/pybind11/utils.h"

#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {

namespace py = ::pybind11;

void init_pyspiel_utils(py::module& m) {
  // read_contents_from_file(string filename, string mode)
  m.def("read_contents_from_file", file::ReadContentsFromFile,
        "Read the entire contents of a file.");

  // write_contents_to_file(string filename, string mode, string contents)
  m.def("write_contents_to_file", open_spiel::file::WriteContentsToFile,
        "Write the contents of the string to the specified filename.");
}

}  // namespace open_spiel
