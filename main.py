import sys
import time
import argparse

import numpy as np
import cupy as cp

import bionumpy as bnp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cntr_keys", type=str, default=None)
    parser.add_argument("-max_chunks", type=int, default=2**32)
    parser.add_argument("--cupy", action="store_true", default=False)
    parser.add_argument("--compare", action="store_true", default=False)
    parser.add_argument("--do_assert", action="store_true", default=False)

    args = parser.parse_known_args()
    return args

args = get_args()[0]

if args.cupy and not args.compare:
    print("using cupy")
    bnp.set_backend(cp)
elif not args.compare:
    print("using numpy")

if args.do_assert and args.max_chunks >= 50:
    print("\33[91mWarning\33[0m: Asserting with a large number of chunks will use considerable memory on both host and device.")

#if args.cntr_keys is None:
    #print("Error: no counter keys provided.")
    #exit(1)

#counter_keys = np.load(args.cntr_keys)
#exit()

def time_hash(verbose=False):
    outputs = []
    n = 0
    t1 = time.time()
    for chunk in bnp.open(sys.argv[1]):
        if n >= args.max_chunks:
            break

        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)
        if args.do_assert:
            outputs.append(kmers.ravel())
        n += 1
        if verbose:
            print(f"chunks processed: {n}", end="\r")

    t2 = time.time()
    if verbose:
        print(f"chunks processed: {n}")
    return t2 - t1, outputs

if args.compare:
    print("running numpy vs cupy comparison ...")
    print("running numpy")
    np_elapsed_secs, np_outputs = time_hash(verbose=True)
    bnp.set_backend(cp)
    print("running cupy")
    cp_elapsed_secs, cp_outputs = time_hash(verbose=True)

    if args.do_assert:
        print("------------------------------")
        assert len(np_outputs) == len(cp_outputs)
        l = len(np_outputs)
        for i in range(l):
            print(f"asserting hashed kmers {i+1}/{l}", end="\r")
            np.testing.assert_array_equal(np_outputs[i], cp.asnumpy(cp_outputs[i]))
        if l:
            print(f"asserting hashed kmers {i+1}/{l}")
        print("all asserts passed")

    print("---------- RESULTS ----------")
    print(f"numpy elapsed seconds : {round(np_elapsed_secs, 1)}")
    print(f" cupy elapsed seconds : {round(cp_elapsed_secs, 1)}")
    
else:
    elapsed, outputs = time_hash(verbose=True)
    m = "cupy" if args.cupy else "numpy"
    print(f"{m} elapsed seconds : {round(elapsed, 1)}")
