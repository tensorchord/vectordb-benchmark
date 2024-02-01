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

GLOVE_25_COSINE = DatasetConfig(
    vector_dim=25,
    num=1_200_000,
    distance=Distance.COSINE,
    type=FileType.H5,
    path="datasets/glove-25-angular.hdf5",
    link="https://ann-benchmarks.com/glove-25-angular.hdf5",
)

GLOVE_100_COSINE = DatasetConfig(
    vector_dim=100,
    num=1_200_000,
    distance=Distance.COSINE,
    type=FileType.H5,
    path="datasets/glove-100-angular.hdf5",
    link="https://ann-benchmarks.com/glove-100-angular.hdf5",
)

DEEP_96_COSINE = DatasetConfig(
    vector_dim=96,
    num=10_000_000,
    distance=Distance.COSINE,
    type=FileType.H5,
    path="datasets/deep-image-96-angular.hdf5",
    link="https://ann-benchmarks.com/deep-image-96-angular.hdf5",
)

LAION_768_DOT_PRODUCT = DatasetConfig(
    vector_dim=512,
    num=5_000_000,
    distance=Distance.DOT_PRODUCT,
    type=FileType.H5,
    path="datasets/laion-768-ip.hdf5",
    link="https://myscale-datasets.s3.ap-southeast-1.amazonaws.com/laion-5m-test-ip.hdf5",
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
    GLOVE_25_COSINE = GLOVE_25_COSINE
    GLOVE_100_COSINE = GLOVE_100_COSINE
    DEEP_96_COSINE = DEEP_96_COSINE
    LAION_768_DOT_PRODUCT = LAION_768_DOT_PRODUCT
    RANDOM_128_L2 = RANDOM_128_L2
