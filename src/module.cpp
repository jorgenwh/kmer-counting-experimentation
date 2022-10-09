#include <iostream>
#include <inttypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "naive_hashtable.h"
#include "cops_hashtable.h"

namespace py = pybind11;

PYBIND11_MODULE(cuht_module, m) {
  m.doc() = "Documentation for the cuht (cuda hashtable) module";

  py::class_<NaiveHashTable>(m, "NaiveHashTable")
    .def(py::init([](py::array_t<uint64_t> &keys, const uint32_t capacity) { 
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      const bool cuda_keys = false;
      return new NaiveHashTable(data, cuda_keys, size, capacity);
    }))
    .def(py::init([](long keys_ptr, const uint32_t size, const uint32_t capacity) { 
      const uint64_t *data = reinterpret_cast<uint64_t*>(keys_ptr);
      const bool cuda_keys = true;
      return new NaiveHashTable(data, cuda_keys, size, capacity);
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

      auto ret = py::array_t<uint32_t>(buf.size);
      uint32_t *counts_data = ret.mutable_data();

      self.get(keys_data, counts_data, size);

      return ret;
    })
    .def("getcu", [](NaiveHashTable &self, long keys_ptr, long counts_ptr, uint32_t size) {
      uint64_t *keys = reinterpret_cast<uint64_t*>(keys_ptr);
      uint32_t *counts = reinterpret_cast<uint32_t*>(counts_ptr);
      self.getcu(keys, counts, size);
    })
    ;

  py::class_<COPSHashTable>(m, "COPSHashTable")
    .def(py::init([](py::array_t<uint64_t> &keys, const uint32_t capacity, const uint32_t cg_size) { 
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      const bool cuda_keys = false;
      return new COPSHashTable(data, cuda_keys, size, capacity, cg_size);
    }))
    .def(py::init([](long keys_ptr, const uint32_t size, const uint32_t capacity, const uint32_t cg_size) { 
      const uint64_t *data = reinterpret_cast<uint64_t*>(keys_ptr);
      const bool cuda_keys = true;
      return new COPSHashTable(data, cuda_keys, size, capacity, cg_size);
    }))
    .def("size", &COPSHashTable::size)
    .def("capacity", &COPSHashTable::capacity)
    .def("__repr__", &COPSHashTable::to_string)
    .def("count", [](COPSHashTable &self, py::array_t<uint64_t> &keys) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      self.count(data, size);
    })
    .def("countcu", [](COPSHashTable &self, long data_ptr, uint32_t size) {
      uint64_t *data = reinterpret_cast<uint64_t*>(data_ptr);
      self.countcu(data, size);
    })
    .def("get", [](COPSHashTable &self, py::array_t<uint64_t> &keys) {
      py::buffer_info buf = keys.request();

      const uint64_t *keys_data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();

      auto ret = py::array_t<uint32_t>(buf.size);
      uint32_t *counts_data = ret.mutable_data();

      self.get(keys_data, counts_data, size);

      return ret;
    })
    .def("getcu", [](COPSHashTable &self, long keys_ptr, long counts_ptr, uint32_t size) {
      uint64_t *keys = reinterpret_cast<uint64_t*>(keys_ptr);
      uint32_t *counts = reinterpret_cast<uint32_t*>(counts_ptr);
      self.getcu(keys, counts, size);
    })
    ;
}
