from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - (dot_product(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y)


class Distance(Enum):
    EUCLIDEAN = euclidean
    COSINE = cosine
    DOT_PRODUCT = dot_product


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
    vector_dim: int
    url: str = "postgresql://postgres:password@127.0.0.1:5432/postgres"


@dataclass
class Record:
    id: int
    vector: list[float]
    metadata: Optional[dict] = None


@dataclass
class Query:
    vector: list[float]
    expect_ids: Optional[list[int]]
    expect_scores: Optional[list[float]]
    metadata: Optional[dict] = None
