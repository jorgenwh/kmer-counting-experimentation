import sys
import time
import argparse

import numpy as np
import cupy as cp

import bionumpy as bnp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cntr_keys", type=str, default=None)
    parser.add_argument("--cupy", action="store_true", default=False)
    parser.add_argument("--comp", action="store_true", default=False)

    args = parser.parse_known_args()
    return args

args = get_args()[0]

if args.cupy and not args.comp:
    print("using cupy")
    bnp.set_backend(cp)
elif not args.comp:
    print("using numpy")

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
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)
        outputs.append(kmers.ravel())
        if verbose:
            print(f"chunks processed: {n+1}", end="\r")
        n += 1
    t2 = time.time()
    if verbose:
        print(f"chunks processed: {n+1}")
    return t2 - t1, outputs

if args.comp:
    print("running numpy vs cupy comparison ...")
    print("running numpy")
    np_elapsed_secs, np_outputs = time_hash(verbose=True)
    bnp.set_backend(cp)
    print("running cupy")
    cp_elapsed_secs, cp_outputs = time_hash(verbose=True)
    print("------------------------------")

    #for np_kmers, cp_kmers in zip(np_outputs, cp_outputs):
        #np.testing.assert_array_equal(np_kmers, cp.asnumpy(cp_kmers))
    assert len(np_outputs) == len(cp_outputs)
    l = len(np_outputs)
    for i in range(l):
        print("asserting hashed kmers {i+1}/{l}", end="\r")
        np.testing.assert_array_equal(np_outputs[i], cp.asnumpy(cp_outputs[i]))
        del np_outputs[i]
        del cp_outputs[i]
    print("asserting hashed kmers {i+1}/{l}")

    print("---------- RESULTS ----------")
    print(f"numpy elapsed seconds : {round(np_elapsed_secs, 1)}")
    print(f" cupy elapsed seconds : {round(cp_elapsed_secs, 1)}")
    
else:
    elapsed, outputs = time_hash(verbose=True)
    m = "cupy" if args.cupy else "numpy"
    print(f"{m} elapsed seconds : {round(elapsed, 1)}")
