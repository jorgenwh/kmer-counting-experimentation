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
    parser = argparse.ArgumentParser("Script checking that counts computed by f2i_C.CuHashTable and npstructures.Counter are equal.")
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    return parser.parse_args()


args = get_arguments()
