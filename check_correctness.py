import time
import importlib
import shutil
import pickle
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from temp.f2i_C import NaiveHashTable 


class CuCounter(NaiveHashTable):
    def __init__(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray))
        super().__init__(keys)

    def count(self, kmers):
        if isinstance(kmers, np.ndarray):
            super().count(kmers)
        elif isinstance(kmers, cp.ndarray):
            super().countcu(kmers.data.ptr, kmers.size)
        else:
            print("Error: invalid type provided as kmers")

    def __getitem__(self, keys):
        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys)
            super().getcu(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 
        else:
            print("Error: invalid type provided as keys")


def get_arguments():
    parser = argparse.ArgumentParser("Script checking that counts computed by f2i_C.NaiveHashTable and npstructures.Counter are equal.")
    parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    return parser.parse_args()


fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"
args = get_arguments()

def check(fasta_filename, keys_filename, xp, counter_size, chunk_size):
    keys = np.load(keys_filename)[:counter_size]

    nps_counter = nps.Counter(keys=keys)
    f2i_C_counter = CuCounter(keys=keys)

    c = 0
    for chunk in bnp.open(fasta_filename, chunk_size=chunk_size):
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)

        nps_counter.count(kmers.ravel())
        f2i_C_counter.count(kmers.ravel())

        c += 1
        print(f"Counting kmers ... {c}\r", end="")
    print(f"Counting kmers ... {c}")
    
    nps_counts = nps_counter[keys.ravel()]
    f2i_C_counts = f2i_C_counter[keys.ravel()]
    
    xp.testing.assert_array_equal(nps_counts, f2i_C_counts)
    print("Assert passed")

if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp

    print(f"Backend array module : {array_module.__name__}")
    print(f"Counter size         : {args.counter_size}")
    print(f"Chunk size           : {args.chunk_size}")
    
    check(fasta_filename, keys_filename, array_module, args.counter_size, args.chunk_size)
