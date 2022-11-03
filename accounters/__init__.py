#from . import counters
#from .counters import CuCounter
#from .counters import CppCounter
#from .counters import CupyCounter

from . import kmers
from .kmers import word_reverse_complement
from .kmers import kmer_word_to_bitstring
from .kmers import bitstring_to_ACGT
from .kmers import get_ACGT_reverse_complement

from accounters_C import word_reverse_complement as wrc
from accounters_C import get_reverse_complements
from accounters_C import convert_ACTG_to_ACGT_encoding
from accounters_C import get_unique_complements
from accounters_C import test 
