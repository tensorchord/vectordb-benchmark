from __future__ import annotations

import abc

from vector_bench.spec import Record


class BaseClient(abc.ABC):
    @abc.abstractmethod
    def insert_batch(self, records: list[Record]):
        pass

    @abc.abstractmethod
    def query(self, vector: list[float], top_k: int):
        pass
