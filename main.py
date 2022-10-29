import sys
import time
import math
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

if __name__ == "__main__":
    obj = np.load("data/npz/counter_index_only_variants_with_revcomp.npz")

    # keys
    counter_keys_data = obj[
            "counter_index_only_variants_with_revcomp-counter-_keys-_data"
    ]
    counter_keys_shape_codes = obj[
            "counter_index_only_variants_with_revcomp-counter-_keys-shape-_codes"
    ]

    print(f"counter._keys._data.shape={counter_keys_data.shape}")
    print(f"counter._keys._data.dtype={counter_keys_data.dtype}")
    print(f"counter._keys._data.nbytes={counter_keys_data.nbytes}")
    print(f"counter._keys._shape._codes.shape={counter_keys_shape_codes.shape}")
    print(f"counter._keys._shape._codes.dtype={counter_keys_shape_codes.dtype}")
    print(f"counter._keys._shape._codes.nbytes={counter_keys_shape_codes.nbytes}\n")

    counter_keys_data = xp.asanyarray(counter_keys_data)
    counter_keys_shape_codes = xp.asanyarray(counter_keys_shape_codes)

    counter_keys_shape = nps.RaggedShape(
            codes=counter_keys_shape_codes, 
            is_coded=True)

    counter_keys = nps.RaggedArray(
            data=counter_keys_data, 
            shape=counter_keys_shape, 
            dtype=counter_keys_data.dtype)

    # values
    counter_values_data = obj[
            "counter_index_only_variants_with_revcomp-counter-_values-_data"
    ]
    counter_values_data = counter_values_data.astype(np.uint16)
    counter_values_shape_codes = obj[
            "counter_index_only_variants_with_revcomp-counter-_values-shape-_codes"
    ]

    print(f"counter._values._data.shape={counter_values_data.shape}")
    print(f"counter._values._data.dtype={counter_values_data.dtype}")
    print(f"counter._values._data.nbytes={counter_values_data.nbytes}")
    print(f"counter._values._shape._codes.shape={counter_values_shape_codes.shape}")
    print(f"counter._values._shape._codes.dtype={counter_values_shape_codes.dtype}")
    print(f"counter._values._shape._codes.nbytes={counter_values_shape_codes.nbytes}\n")

    counter_values_data = xp.asanyarray(counter_values_data)
    counter_values_shape_codes= xp.asanyarray(counter_values_shape_codes)

    counter_value_shape = nps.RaggedShape(
            codes=counter_values_shape_codes, 
            is_coded=True)

    counter_values = nps.RaggedArray(
            data=counter_values_data, 
            shape=counter_value_shape, 
            dtype=counter_values_data.dtype)

    print(type(counter_keys))
    print(type(counter_values))

    # counter
    counter = nps.Counter(counter_keys, counter_values)

    print(type(counter))

