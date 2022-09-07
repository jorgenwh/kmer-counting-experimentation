#include <stdio.h>
#include <string>

#include "fasta_parser.h"

chunk_t *FastaParser::get_chunk() {
  chunk_t *chunk = new chunk_t();
  chunk->size = chunk_size_m;
  chunk->sequence = new char[chunk_size_m];

  return chunk;
}
