import time
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from counters import NaiveCounter as NaiveCuhtCounter
from counters import COPSCounter as COPSCuhtCounter

cuht_counter_types = {
        "naive" : NaiveCuhtCounter,
        "cops"  : COPSCuhtCounter
}

def get_arguments():
    parser = argparse.ArgumentParser("Script checking that counts computed by f2i_C.NaiveHashTable and npstructures.Counter are equal.")
    parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-cuht_counter_type", choices=["naive", "cops"], required=True)
    parser.add_argument("-cuht_capacity", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    return parser.parse_args()


fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"
args = get_arguments()

def check(fasta_filename, keys_filename, xp, counter_size, cuht_counter_type, cuht_capacity, chunk_size):
    keys = np.load(keys_filename)[:counter_size]
    keys = xp.asanyarray(keys)

    nps_counter = nps.Counter(keys=keys)
    cuht_counter = cuht_counter_type(keys=keys, capacity=cuht_capacity)

    c = 0
    for chunk in bnp.open(fasta_filename, chunk_size=chunk_size):
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)

        nps_counter.count(kmers.ravel())
        cuht_counter.count(kmers.ravel())

        c += 1
        print(f"Counting kmer chunks ... {c}\r", end="")
    print(f"Counting kmer chunks ... {c}")

    nps_counts = nps_counter[keys.ravel()]
    cuht_counts = cuht_counter[keys.ravel()]

    assert isinstance(nps_counts, xp.ndarray)
    assert isinstance(cuht_counts, xp.ndarray)
    xp.testing.assert_array_equal(nps_counts, cuht_counts)
    print("Assert passed")

if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp

    print(f"Backend array module : {array_module.__name__}")
    print(f"Counter size         : {args.counter_size}")
    print(f"cuht counter type    : {args.cuht_counter_type}")
    print(f"cuht capacity        : {args.cuht_capacity}")
    print(f"Chunk size           : {args.chunk_size}")

    cuht_counter_type = cuht_counter_types[args.cuht_counter_type]
    
    check(fasta_filename, keys_filename, array_module, args.counter_size, cuht_counter_type, args.cuht_capacity, args.chunk_size)
