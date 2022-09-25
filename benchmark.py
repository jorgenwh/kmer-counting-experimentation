import time
import importlib
import shutil
import pickle
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from temp.f2i_C import CuHashTable


class CuCounter(CuHashTable):
    def count(self, kmers):
        if isinstance(kmers, cp.ndarray):
            super().countcu(kmers.data.ptr, kmers.size)
        else:
            super().count(kmers)


def get_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking script for reading a fasta file and achieving a frequency count for kmers")
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--counter", choices=["nps", "f2i_C"], required=True)
    parser.add_argument("--counter_size", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)

    return parser.parse_args()


arguments = get_arguments()

fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"


def pipeline(fasta_filename, keys_filename, xp, counter_type, counter_size, chunk_size):
    shsize = shutil.get_terminal_size().columns
    print(f"BACKEND_ARRAY_MODULE   : {xp.__name__}")
    print(f"COUNTER_TYPE           : {counter_type}")
    print(f"COUNTER_SIZE           : {counter_size}")
    print(f"CHUNK_SIZE             : {chunk_size}\n")

    time_data = {
            "backend_array_module": xp.__name__, 
            "counter_type": str(counter_type),
            "counter_size": counter_size,
            "chunk_size": chunk_size}
        
    keys = np.load(keys_filename)[:counter_size]
    #keys = xp.asanyarray(keys)

    t = time.time()
    counter = counter_type(keys)
    t = time.time()
    counter_init_elapsed = time.time() - t

    chunk_creation_t = 0
    chunk_hashing_t = 0
    chunk_counting_t = 0

    t_ = time.time()

    num_chunks = 0
    for chunk in bnp.open(fasta_filename, chunk_size=chunk_size):
        t = time.time()
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)
        chunk_hashing_t += time.time() - t

        t = time.time()
        counter.count(kmers.ravel())
        chunk_counting_t += time.time() - t

        num_chunks+=1
        print(f"PROCESSING CHUNK: {num_chunks}", end="\r")
    print(f"PROCESSING CHUNK: {num_chunks}")

    total_t = time.time() - t_
    chunk_creation_t = total_t - (chunk_hashing_t + chunk_counting_t)

    print("\33[2m" + "".join(["-"] * (int(shsize/2)-4)), end="\33[0m")
    print("\33[1mTIMES\33[0m: ", end="")
    print("\33[2m" + "".join(["-"] * (int(shsize/2)-4)) + "\33[0m")
    print(f"CHUNK_CREATION_TIME    : {round(chunk_creation_t, 1)} seconds")
    print(f"CHUNK_HASHING_TIME     : {round(chunk_hashing_t, 1)} seconds")
    print(f"CHUNK_COUNTING_TIME    : {round(chunk_counting_t, 1)} seconds")
    print(f"TOTAL_FA2COUNTS_TIME   : {round(total_t, 1)} seconds")

    time_data["chunk_creation_time"] = chunk_creation_t
    time_data["chunk_hashing_time"] = chunk_hashing_t
    time_data["chunk_counting_time"] = chunk_counting_t
    time_data["total_time"] = total_t 

    return time_data


if __name__ == "__main__":
    if arguments.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if arguments.backend == "numpy" else "cupy"
    counter_type = nps.Counter if arguments.counter_type == "nps" else CuCounter
    
    time_data = pipeline(
            fasta_filename="data/fa/testreads20m.fa", 
            keys_filename="data/npy/uniquekmers.npy", 
            xp=array_module, 
            counter_type=counter_type, 
            counter_size=arguments.counter_size, 
            chunk_size=arguments.chunk_size)

