# kmer-counting-experimentation

CUDA hashtable implementations used for kmer counting wrapped for use in Python.
benchmark.py and check\_correctness.py are used to benchmark the implementations (comparing them to npstructures' Counter implementation based on numpy/cupy, and checking the implementation's correctness, ensuring their counts are equal to those found by npstructure's Counter.

Current implementations:
- NaiveHashTable: a naive CUDA hashtable using linear probing with one CUDA thread per key insertion/count. The downside of this strategy is that it doesn't achieve coalesced memory access since the different threads will access memory wherever the hash value of the entry key places them.
- CGHashTable (Cooperative Group HashTable): a CUDA hashtable implementation based on WarpCore's idea of Cooperative Groups to achieve coalesced memory access (https://arxiv.org/pdf/2009.07914.pdf). Currently a work in progress.
