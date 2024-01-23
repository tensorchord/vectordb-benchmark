from __future__ import annotations

import abc

from vector_bench.spec import DatabaseConfig, Record


class BaseClient(abc.ABC):
    @abc.abstractmethod
    def insert_batch(self, records: list[Record]):
        pass

    @abc.abstractmethod
    def query(self, vector: list[float], top_k: int = 10):
        pass

    @abc.abstractclassmethod
    def from_config(cls, config: DatabaseConfig) -> BaseClient:
        pass
