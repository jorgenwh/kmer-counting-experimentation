import time
import importlib
import shutil
import pickle
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from accounters import CuCounter, CupyCounter, CppCounter


parser = argparse.ArgumentParser(
        description="Benchmarking script used to evaluate the different counter objects provided by accounters")
parser.add_argument("-counter", choices=["cu", "cupy", "cpp", "nps"], required=True)
parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
parser.add_argument("-num_keys", type=int, required=True)
parser.add_argument("-chunk_size", type=int, required=True)
parser.add_argument("-counter_capacity", type=int, default=0)
parser.add_argument("--count_revcomps", action="store_true")

args = parser.parse_args()

fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmersACGT.npy"

counter_types = {
        "cu"   : CuCounter,
        "cupy" : CupyCounter,
        "cpp"  : CppCounter,
        "nps"  : nps.Counter
}


def benchmark(counter_type, xp, num_keys, chunk_size, 
        counter_capacity, count_revcomps):
    global keys_filename

    print("-"*shutil.get_terminal_size().columns)
    print("<<<        INFO        >>>")
    print(f"COUNTER                : {counter_type}")
    print(f"BACKEND_ARRAY_MODULE   : {xp.__name__}")
    print(f"NUM_KEYS               : {num_keys}")
    print(f"CHUNK_SIZE             : {chunk_size}")
    print(f"COUNTER_CAPACITY       : {counter_capacity}")
    print(f"COUNT_REVCOMPS         : {count_revcomps}")

    if count_revcomps:
        keys_filename = "data/npy/revcomps_randgen.npy"

    keys = np.load(keys_filename)[:num_keys]
    keys = xp.asanyarray(keys)

    _t = time.time()
    t = time.time()
    if counter_type == CuCounter:
        counter = CuCounter(keys=keys, capacity=counter_capacity)
    if counter_type == CupyCounter:
        counter = CupyCounter(keys=keys, capacity=counter_capacity)
    if counter_type == CppCounter:
        if xp == cp:
            keys = cp.asnumpy(keys)
        counter = CppCounter(keys=keys, capacity=counter_capacity)
    if counter_type == nps.Counter:
        counter = nps.Counter(keys=keys)
    init_t = time.time() - t

    chunk_generator = bnp.open(fasta_filename, chunk_size=chunk_size)
    chunk_t = 0
    hash_t = 0
    count_t = 0

    num_chunks = 0
    while True:
        try:
            t = time.time()
            chunk = next(chunk_generator) 
            cp.cuda.runtime.deviceSynchronize()
            chunk_t += (time.time() - t)
        except StopIteration:
            break

        t = time.time()
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding).ravel()
        hash_t += (time.time() - t)

        if xp == cp and counter_type == CppCounter:
            kmers = cp.asnumpy(kmers)

        t = time.time()
        counter.count(kmers)
        #counter.count(kmers, count_revcomps=True, kmer_size=31)
        cp.cuda.runtime.deviceSynchronize()
        count_t += (time.time() - t)

        num_chunks+=1
        print(f"PROCESSING CHUNK: {num_chunks} / ...", end="\r")
    print(f"PROCESSING CHUNK: {num_chunks} / {num_chunks}") 

    total_t = time.time() - _t

    print("<<<    TIMES (secs)    >>>")
    print(f"COUNTER INITIALIZATION : {round(init_t, 3)}")
    print(f"CHUNK CREATION         : {round(chunk_t, 3)}")
    print(f"CHUNK HASHING          : {round(hash_t, 3)}")
    print(f"CHUNK COUNTING         : {round(count_t, 3)}")
    print(f"TOTAL                  : {round(total_t, 3)}")


if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp
    counter_type = counter_types[args.counter]

    if array_module == cp and counter_type == CppCounter:
        print("Warning: CppCounter does not support CuPy arrays so they will therefore be copied to host ahead of any CppCounter operations.")
    
    benchmark(
            counter_type=counter_type, 
            xp=array_module, 
            num_keys=args.num_keys, 
            chunk_size=args.chunk_size,
            counter_capacity=args.counter_capacity,
            count_revcomps=args.count_revcomps
    )

