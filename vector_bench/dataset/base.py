from __future__ import annotations

import abc
from typing import Iterator

from vector_bench.spec import DatasetConfig, Query, Record


class BaseReader(abc.ABC):
    @abc.abstractmethod
    def read_record(self) -> Iterator[Record]:
        pass

    @abc.abstractmethod
    def read_query(self) -> Iterator[Query]:
        pass

    @abc.abstractclassmethod
    def from_config(cls, config: DatasetConfig) -> BaseReader:
        pass
