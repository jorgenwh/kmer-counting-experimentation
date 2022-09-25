#include <iostream>
#include <inttypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "hashtable.h"

namespace py = pybind11;

PYBIND11_MODULE(f2i_C, m) {
  m.doc() = "Documentation for the f2i_C module";

  py::class_<CuHashTable>(m, "CuHashTable")
    .def(py::init([](py::array_t<uint64_t> &keys) { 
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      return new CuHashTable(data, size);
    }))
    .def("__repr__", &CuHashTable::to_string)
    //.def("count", &CuHashTable::count)
    .def("count", [](CuHashTable &self, py::array_t<uint64_t> &keys) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      self.count(data, size);
    })
    .def("countcu", [](CuHashTable &self, long data_ptr, uint32_t size) {
      uint64_t *data = reinterpret_cast<uint64_t*>(data_ptr);
      self.countcu(data, size);
    })
    ;
}
