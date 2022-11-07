#include <inttypes.h>
#include <string>
#include <vector>
#include <assert.h>
#include <iostream>
#include <bitset>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuhashtable.h"
#include "cpphashtable.h"
#include "kmers.h"

namespace py = pybind11;

PYBIND11_MODULE(accounters_C, m) {
  m.doc() = "...";

  m.def("test", []() {
    uint64_t x = 0x3FFFFFFF00000002;

    std::cout << std::bitset<64>(x) << "\n";
    auto y = x << 3;
    std::cout << std::bitset<64>(y) << "\n\n";

    std::cout << std::bitset<64>(x) << "\n";
    auto z = x >> 3;
    std::cout << std::bitset<64>(z) << "\n";
  });

  m.def("get_unique_complements", [](py::array_t<uint64_t> &kmers, const uint8_t kmer_size) {
    py::buffer_info buf = kmers.request();

    const uint64_t *kmers_data = (uint64_t *)kmers.data();
    const uint32_t size = kmers.size();

    std::unordered_set<uint64_t> unique_complements = get_unique_complements_set(kmers_data, size, kmer_size);

    const uint32_t unqcomp_size = unique_complements.size();
    auto ret = py::array_t<uint64_t>({unqcomp_size});
    uint64_t *unique_complements_data = ret.mutable_data();

    int i = 0;
    for (const auto &kmer: unique_complements) {
      unique_complements_data[i] = kmer;
      i++;
    }

    return ret;
  });

  m.def("convert_ACTG_to_ACGT_encoding", [](py::array_t<uint64_t> &kmers) {
    py::buffer_info buf = kmers.request();

    const uint64_t *kmer_data = (uint64_t *)kmers.data();
    const uint32_t size = kmers.size();

    auto ret = py::array_t<uint64_t>(buf.size);
    uint64_t *ret_data = ret.mutable_data();
    memset(ret_data, 0, sizeof(uint64_t)*size);

    convert_ACTG_to_ACGT_encoding(kmer_data, ret_data, size);

    return ret;
  });

  m.def("get_reverse_complements", [](py::array_t<uint64_t> &kmers, uint8_t kmer_size) {
    py::buffer_info buf = kmers.request();

    const uint64_t *kmers_data = (uint64_t *)kmers.data();
    const uint32_t size = kmers.size();

    auto ret = py::array_t<uint64_t>(buf.size);
    uint64_t *revcomps = ret.mutable_data();
    memset(revcomps, 0, sizeof(uint64_t)*size);

    get_reverse_complements(kmers_data, revcomps, size, kmer_size);

    return ret;
  });

  m.def("word_reverse_complement", [](const uint64_t kmer, const uint8_t kmer_size) {
    return word_reverse_complement(kmer, kmer_size);
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

    .def("count", [](CuHashTable &self, py::array_t<uint64_t> &keys, const bool count_revcomps, const uint8_t kmer_size) {
      const uint64_t *data = (uint64_t *)keys.data();
      const uint32_t size = keys.size();
      self.count(data, size, count_revcomps, kmer_size);
    })

    .def("count", [](CuHashTable &self, long data_ptr, uint32_t size, const bool count_revcomps, const uint8_t kmer_size) {
      uint64_t *data = reinterpret_cast<uint64_t*>(data_ptr);
      self.countcu(data, size, count_revcomps, kmer_size);
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
