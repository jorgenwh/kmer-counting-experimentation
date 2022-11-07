#include <stdio.h>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <thread>
#include <mutex>
#include <inttypes.h>

#include "common.h"
#include "cpphashtable.h"

CppHashTable::CppHashTable(const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  size_m = size;
  capacity_m = capacity;

  table_m.keys = new uint64_t[capacity_m];
  table_m.values= new uint32_t[capacity_m];
  memset(table_m.keys, 0xff, sizeof(uint64_t)*capacity_m);

  for (int i = 0; i < size; i++) {
    uint64_t key = keys[i];
    uint64_t hash = key % capacity_m;

    while (true) {
      uint64_t cur = table_m.keys[hash]; 

      if (cur == kEmpty || cur == key) {
        table_m.keys[hash] = key;
        table_m.values[hash] = 0;
        break;
      }
      hash = (hash + 1) % capacity_m;
    }
  }
}

void CppHashTable::counting_worker(const uint64_t *keys, const uint32_t size, 
    const uint32_t offset, const uint32_t num_threads) {
  for (uint32_t i = offset; i < size; i+=num_threads) {
    uint64_t key = keys[i];
    uint64_t hash = key % capacity_m;

    while (true) {
      uint64_t cur = table_m.keys[hash];
      
      if (cur == key) {
        { 
          std::lock_guard<std::mutex> lock(mu_m);
          table_m.values[hash]++;
        }
        break;
      }
      if (cur == kEmpty) { break; }
      hash = (hash + 1) % capacity_m;
    }
  }
}

void CppHashTable::count(
    const uint64_t *keys, const uint32_t size, const uint32_t num_threads) {
  std::vector<std::thread> threads(num_threads);
  uint32_t offsets[num_threads];
  for (uint32_t i = 0; i < num_threads; i++) {
    offsets[i] = i;
    threads[i] = std::thread(&CppHashTable::counting_worker, this, 
        keys, size, offsets[i], num_threads);
  }
  for (auto &thread : threads) {
    if (thread.joinable()) { thread.join(); }
  }
}

void CppHashTable::get(
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t num_threads) {
  for (int i = 0; i < size; i++) {
    uint64_t key = keys[i];
    uint64_t hash = key % capacity_m;

    while (true) {
      uint64_t cur = table_m.keys[hash]; 

      if (cur == key) {
        counts[i] = table_m.values[hash];
        break;
      }
      if (cur == kEmpty) {
        counts[i] = 0;
        break;
      }
      hash = (hash + 1) % capacity_m;
    }
  }
}

std::string CppHashTable::to_string() const {
  int print_size = (capacity_m < 40) ? capacity_m : 40;

  std::ostringstream oss;
  std::ostringstream keys_oss;
  std::ostringstream values_oss;

  keys_oss << "[";
  values_oss << "[";
  uint32_t elements = 0;
  for (int i = 0; i < capacity_m; i++) {
    uint64_t key = table_m.keys[i];
    uint32_t value = table_m.values[i];

    if (key == kEmpty) { continue; }

    if (elements != 0) { 
      keys_oss << ", "; 
      values_oss << ", "; 
    }

    keys_oss << key;
    values_oss << value;
    
    elements++;
    if (elements >= print_size) { break; }
  }
  keys_oss << "]";
  values_oss << "]";

  oss << "Counter(" << keys_oss.str() << ", " << values_oss.str();
  oss << ", size=" << size_m << ", capacity=" << capacity_m << ")";

  return oss.str();
}
