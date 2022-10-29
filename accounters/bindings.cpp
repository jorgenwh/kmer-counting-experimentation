#include <inttypes.h>
#include <string>
#include <vector>
#include <assert.h>
#include <iostream>
#include <bitset>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuhashtable.h"
#include "cpphashtable.h"
#include "kmers.h"

namespace py = pybind11;

PYBIND11_MODULE(accounters_C, m) {
  m.doc() = "...";

  m.def("ascii_to_kmer_hashes", [](std::string &kmers, uint32_t kmer_size) {
    uint32_t num_kmers = kmers.size() / kmer_size;

    auto shape = std::vector<size_t>({num_kmers});
    auto ret = py::array_t<uint8_t>(shape);

    uint8_t *ret_data = ret.mutable_data();
    const char *bases = kmers.c_str();

    encode_kmers(bases, ret_data, kmer_size);

    return ret;
  });

  m.def("kmer_hashes_to_ascii", [](py::array_t<uint64_t> &kmers) {
    const uint64_t *hashes = kmers.data();
    const uint64_t size = kmers.size();

    auto shape = std::vector<size_t>({size, 32});
    auto ret = py::array_t<uint8_t>(shape);
    uint8_t *ret_data = ret.mutable_data();

    hashes_to_ascii(hashes, size, ret_data, size*32);

    return ret;
  });

  py::class_<CuHashTable>(m, "CuHashTable")
    .def(py::init([](py::array_t<uint64_t> &keys, const uint32_t capacity) { 
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      const bool cuda_keys = false;
      return new CuHashTable(data, cuda_keys, size, capacity);
    }))
    .def(py::init([](long keys_ptr, const uint32_t size, const uint32_t capacity) { 
      const uint64_t *data = reinterpret_cast<uint64_t*>(keys_ptr);
      const bool cuda_keys = true;
      return new CuHashTable(data, cuda_keys, size, capacity);
    }))
    .def("size", &CuHashTable::size)
    .def("capacity", &CuHashTable::capacity)
    .def("__repr__", &CuHashTable::to_string)
    .def("count", [](CuHashTable &self, py::array_t<uint64_t> &keys) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      self.count(data, size);
    })
    .def("count", [](CuHashTable &self, long data_ptr, uint32_t size) {
      uint64_t *data = reinterpret_cast<uint64_t*>(data_ptr);
      self.countcu(data, size);
    })
    .def("get", [](CuHashTable &self, py::array_t<uint64_t> &keys) {
      py::buffer_info buf = keys.request();

      const uint64_t *keys_data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();

      auto ret = py::array_t<uint32_t>(buf.size);
      uint32_t *counts_data = ret.mutable_data();

      self.get(keys_data, counts_data, size);

      return ret;
    })
    .def("get", [](CuHashTable &self, long keys_ptr, long counts_ptr, uint32_t size) {
      uint64_t *keys = reinterpret_cast<uint64_t*>(keys_ptr);
      uint32_t *counts = reinterpret_cast<uint32_t*>(counts_ptr);
      self.getcu(keys, counts, size);
    })
    ;

  py::class_<CppHashTable>(m, "CppHashTable")
    .def(py::init([](py::array_t<uint64_t> &keys, const uint32_t capacity) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      return new CppHashTable(data, size, capacity);
    }))
    .def("count", [](CppHashTable &self, py::array_t<uint64_t> &keys, uint32_t threads) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      uint32_t max_threads = std::thread::hardware_concurrency();
      if (threads <= 0 || threads > max_threads) {
        threads = max_threads;
      }
      self.count(data, size, threads);
    })
    .def("get", [](CppHashTable &self, py::array_t<uint64_t> &keys, uint32_t threads) {
      py::buffer_info buf = keys.request();
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      uint32_t max_threads = std::thread::hardware_concurrency();
      if (threads <= 0 || threads > max_threads) {
        threads = max_threads;
      }

      auto ret = py::array_t<uint32_t>(buf.size);
      uint32_t *counts_data = ret.mutable_data();

      self.get(data, counts_data, size, threads);

      return ret;
    })
    .def("size", &CppHashTable::size)
    .def("capacity", &CppHashTable::capacity)
    .def("__repr__", &CppHashTable::to_string)
    ;
}
