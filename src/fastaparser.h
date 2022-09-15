#ifndef FASTAPARSER_H_
#define FASTAPARSER_H_

#include <stdio.h>
#include <iostream>
#include <inttypes.h>
#include <fstream>
#include <string.h>

struct chunk {
  size_t size;
  uint8_t *sequence;
};
typedef chunk chunk_t;

class FastaParser {
public:
  FastaParser() = default;
  FastaParser(const char *filename) : chunk_size_m(5000000) {
    init(filename);
  }
  FastaParser(const char *filename, int chunk_size) : chunk_size_m(chunk_size) {
    init(filename);
  }
  ~FastaParser() { file_stream_m.close(); }

  chunk_t read_chunk();
  
private:
  std::ifstream file_stream_m;
  int chunk_size_m;
  bool done_m = false;

  void init(const char *filename);
};

#endif // FASTAPARSER_H_
