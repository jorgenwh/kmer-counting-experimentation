#include <stdio.h>
#include <string>

#include "fasta_parser.h"

int main(int argc, char **argv) {

  std::string filename = argv[0];
  printf("%s\n", filename);

  //size_t chunk_size = std::stoi(argv[1]);
  size_t chunk_size = 100;

  FastaParser parser(filename, chunk_size);

  return 0;
}
