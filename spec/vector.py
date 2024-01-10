from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Distance(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"


class FileType(Enum):
    CSV = "csv"
    JSON = "json"
    H5 = "h5"


@dataclass
class DatasetConfig:
    name: str
    vector_dim: int
    num: int
    distance: Distance
    type: FileType
    path: str
    link: str
    schema: dict


@dataclass
class DatabaseConfig:
    url: str
    dataset: DatasetConfig


@dataclass
class Record:
    id: int
    vector: list[float]
    metadata: Optional[dict]


@dataclass
class Query:
    vector: list[float]
    expect_ids: Optional[list[int]]
    expect_scores: Optional[list[float]]
    metadata: Optional[dict]
