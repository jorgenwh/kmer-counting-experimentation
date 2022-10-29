import numpy as np
import cupy as cp

import bionumpy as bnp

bnp.set_backend(cp)

x = bnp.open("data/fa/testreads20m.fa")
print(x)
