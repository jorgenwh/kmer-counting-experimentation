#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(f2i_C, m) {
  m.doc() = "Dummy documentation for the f2i_C module";
}
