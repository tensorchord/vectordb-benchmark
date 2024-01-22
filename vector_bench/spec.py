import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

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
    RANDOM = "random"


class EnumSelector(Enum):
    @classmethod
    def select(cls, name: str):
        for item in cls:
            if item.name.lower() == name:
                return item.value
        raise ValueError(f"Invalid name: {name}")

    @classmethod
    def list(cls):
        return [item.name.lower() for item in cls]


@dataclass
class DatasetConfig:
    vector_dim: int
    num: int
    distance: Distance
    type: FileType
    path: str
    link: str
    schema: dict = field(default_factory=dict)


@dataclass
class DatabaseConfig:
    vector_dim: int
    distance: Distance
    table: str = "benchmark"
    name: Literal["pgvecto_rs", "pgvector", "qdrant"] = "pgvecto_rs"
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


@dataclass
class BenchmarkResult:
    query: int = 0
    failure: int = 0
    worker_num: int = 0
    total_second: float = 0
    latency: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)

    def response_per_second(self) -> float:
        return (self.query - self.failure) / self.total_second

    def p95_latency(self) -> float:
        return np.percentile(self.latency, 95)

    def mean_precision(self) -> float:
        return np.mean(self.precision)

    def display(self):
        result = {
            "rps": self.response_per_second(),
            "p95_latency": self.p95_latency(),
            "mean_precision": self.mean_precision(),
            "total_request": self.query,
            "failure": self.failure,
            "worker_num": self.worker_num,
            "total_second": self.total_second,
        }
        print(json.dumps(result, indent=4))
