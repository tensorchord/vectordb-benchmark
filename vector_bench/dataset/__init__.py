from vector_bench.dataset.hdf5 import HDF5Reader
from vector_bench.dataset.pseudo import PseudoReader
from vector_bench.spec import EnumSelector


class DatasetReader(EnumSelector):
    RANDOM = PseudoReader
    H5 = HDF5Reader
