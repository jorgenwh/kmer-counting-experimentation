#include <iostream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <fstream>
#include <inttypes.h>
#include <bitset>

#include "dnabitset.h"
#include "fastaparser.h"

int main(int argc, char **argv) {
  const char *filename = argv[1];
  filename = "data/testreads5.fa";
  std::cout << "Filename: " << filename << "\n\n";

  FastaParser fasta_parser(filename);

  chunk_t chunk = fasta_parser.read_chunk();
  for (int i = 0; i < chunk.size; i++) {
    std::cout << chunk.sequence[i];
  }

  delete[] chunk.sequence;
  return 0;
}
