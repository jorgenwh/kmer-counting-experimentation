#include <iostream>
#include <inttypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "naive_hashtable.h"

namespace py = pybind11;

PYBIND11_MODULE(f2i_C, m) {
  m.doc() = "Documentation for the f2i_C module";

  py::class_<NaiveHashTable>(m, "NaiveHashTable")
    .def(py::init([](py::array_t<uint64_t> &keys) { 
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      return new NaiveHashTable(data, size);
    }))
    .def("size", &NaiveHashTable::size)
    .def("capacity", &NaiveHashTable::capacity)
    .def("__repr__", &NaiveHashTable::to_string)
    .def("count", [](NaiveHashTable &self, py::array_t<uint64_t> &keys) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      self.count(data, size);
    })
    .def("countcu", [](NaiveHashTable &self, long data_ptr, uint32_t size) {
      uint64_t *data = reinterpret_cast<uint64_t*>(data_ptr);
      self.countcu(data, size);
    })
    .def("get", [](NaiveHashTable &self, py::array_t<uint64_t> &keys) {
      py::buffer_info buf = keys.request();

      const uint64_t *keys_data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();

      auto ret = py::array_t<uint64_t>(buf.size);
      uint64_t *counts_data = ret.mutable_data();

      self.get(keys_data, counts_data, size);

      return ret;
    })
    .def("getcu", [](NaiveHashTable &self, long keys_ptr, long counts_ptr, uint32_t size) {
      uint64_t *keys = reinterpret_cast<uint64_t*>(keys_ptr);
      uint64_t *counts = reinterpret_cast<uint64_t*>(counts_ptr);
      self.getcu(keys, counts, size);
    })
    ;
}
