from __future__ import annotations

from vector_bench.spec import DatasetConfig, Distance, FileType

GIST_960_L2 = DatasetConfig(
    name="gist-960-euclidean",
    vector_dim=960,
    num=1000000,
    distance=Distance.EUCLIDEAN,
    type=FileType.H5,
    path="datasets/gist-960-euclidean.csv",
    link="https://ann-benchmarks.com/gist-960-euclidean.hdf5",
)

SOURCES = (
    GIST_960_L2,
)