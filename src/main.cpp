#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>

#include "dnabitset.h"

int main(int argc, char **argv) {
  const char *filename = argv[1];
  std::cout << "Filename: " << filename << "\n";

  const char *dna = "acgtacgtacgtaccagttg";
  size_t dna_size = 20;
  printf("%s\n", dna);

  DNABitset dnabs(dna, dna_size);
  std::cout << dnabs.to_string() << "\n";

  return 0;
}
