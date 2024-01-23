from __future__ import annotations

from vector_bench.spec import DatasetConfig, Distance, EnumSelector, FileType

GIST_960_L2 = DatasetConfig(
    vector_dim=960,
    num=1_000_000,
    distance=Distance.EUCLIDEAN,
    type=FileType.H5,
    path="datasets/gist-960-euclidean.hdf5",
    link="https://ann-benchmarks.com/gist-960-euclidean.hdf5",
)

RANDOM_128_L2 = DatasetConfig(
    vector_dim=128,
    num=100_000,
    distance=Distance.EUCLIDEAN,
    type=FileType.RANDOM,
    path="None",
    link="None",
)


class DataSource(EnumSelector):
    GIST_960_L2 = GIST_960_L2
    RANDOM_128_L2 = RANDOM_128_L2
