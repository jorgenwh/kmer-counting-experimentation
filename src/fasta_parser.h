#ifndef FASTA_PARSER_H_
#define FASTA_PARSER_H_

#include <stdio.h>
#include <string>

#define DEFAULT_CHUNK_SIZE 5000000

struct chunk_t {
  size_t size;
  char *sequence;
};

class FastaParser {
public:
  FastaParser() = default;
  FastaParser(std::string &filename)
    : filename_m(filename), chunk_size_m(DEFAULT_CHUNK_SIZE) {};
  FastaParser(std::string &filename, size_t chunk_size)
    : filename_m(filename), chunk_size_m(chunk_size) {};

  chunk_t *get_chunk();
  bool is_done() const { return is_done_m; }
  size_t bytes_read() const { return bytes_read_m; }
private:
  std::string filename_m;
  size_t chunk_size_m;

  size_t bytes_read_m = 0;
  bool is_done_m = false;
};

#endif // FASTA_PARSER_H_
