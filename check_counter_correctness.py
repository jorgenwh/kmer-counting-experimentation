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
        description="Script used to assert the correctness of the counts found by all counters provided by accounters")

args = parser.parse_args()

fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"

counter_types = [CuCounter, CupyCounter, CppCounter, nps.Counter]

def generate_random_data():
    keys = np.unique(np.random.randint(0, 0xffffffffffffffff, 1000, dtype=np.uint64))
    kmers = np.unique(np.random.randint(0, 0xffffffffffffffff, 5000, dtype=np.uint64))
    kmers = np.concatenate((kmers, np.random.choice(keys, size=2500)))
    np.random.shuffle(kmers)
    return keys, kmers

def create_ref(keys, kmers):
    ref = {}
    for key in keys:
        ref[key] = 0
    for key in kmers:
        if key in ref:
            ref[key] += 1
    return ref

def assert_equal_counts(keys, counter, ref):
    counts = counter[keys]
    for key, count in zip(keys, counts):
        if key not in ref:
            ref_count = 0
        else:
            ref_count = ref[key]
        assert count == ref_count

def test_counters(counter_types):
    keys, kmers = generate_random_data()
    cp_keys = cp.asanyarray(keys)
    cp_kmers = cp.asanyarray(kmers)

    ref = create_ref(keys, kmers)

    for counter_type in counter_types:
        counter = counter_type(keys=keys) 
        counter.count(kmers)
        assert_equal_counts(keys, counter, ref)

    print("All asserts passed")

if __name__ == "__main__":
    test_counters(counter_types=counter_types)
