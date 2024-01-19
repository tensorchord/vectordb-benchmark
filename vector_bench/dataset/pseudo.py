from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from vector_bench.dataset.base import BaseReader, DatasetConfig
from vector_bench.log import logger
from vector_bench.spec import Distance, Query, Record


class PseudoReader(BaseReader):
    def __init__(
        self,
        num: int = 100_000,
        dim: int = 128,
        distance: Distance = Distance.DOT_PRODUCT,
    ) -> None:
        self.record_num = num
        self.query_num = 200
        self.dim = dim
        self.top_k = 10
        self.distance = distance

        self.storage_path = Path(f"/tmp/vector_bench_pseudo_num_{num}_dim_{dim}.np")
        if self.storage_path.is_file():
            logger.info("load pseudo dataset from %s", self.storage_path)
            with open(self.storage_path, "rb") as file:
                self.vectors = np.load(file, allow_pickle=False, fix_imports=False)
        else:
            self.vectors = np.random.rand(self.record_num, self.dim)
            logger.info("dump pseudo dataset to %s", self.storage_path)
            with open(self.storage_path, "wb") as file:
                np.save(file, self.vectors, allow_pickle=False, fix_imports=False)

    @classmethod
    def from_config(cls, config: DatasetConfig) -> BaseReader:
        return cls(config.num, config.vector_dim, config.distance)

    def read_record(self) -> Iterator[Record]:
        for i in range(self.record_num):
            yield Record(id=i, vector=self.vectors[i].tolist())

    def read_query(self) -> Iterator[Query]:
        for _ in tqdm(range(self.query_num)):
            query = np.random.rand(self.dim)
            distances = [self.distance(query, x) for x in self.vectors]
            nearest = np.argsort(distances)
            yield Query(
                vector=query.tolist(),
                expect_ids=nearest[: self.top_k].tolist(),
                expect_scores=[distances[i] for i in nearest[: self.top_k]],
            )


if __name__ == "__main__":
    reader = PseudoReader()
    reader.query_num = 5
    for query in reader.read_query():
        print(query.expect_ids, query.expect_scores)
