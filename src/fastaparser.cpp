#include <stdio.h>
#include <iostream>
#include <inttypes.h>
#include <fstream>
#include <string.h>

#include "fastaparser.h"

void FastaParser::init(const char *filename) {
  file_stream_m = std::ifstream(filename);
  done_m = false;
}

chunk_t FastaParser::read_chunk() {
  int read = 0;
  char *buffer = new char[chunk_size_m];

  file_stream_m.read(buffer, chunk_size_m);
  read += file_stream_m.gcount();

  chunk_t c;
  c.size = read;
  c.sequence = new uint8_t[read];
  memcpy(c.sequence, buffer, read);
  delete[] buffer;

  return c;
}
