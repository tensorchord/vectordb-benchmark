import pickle
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

        self.record_path = Path(
            f"/tmp/vector_bench_pseudo_num_{num}_dim_{dim}_record.np"
        )
        self.query_path = Path(
            f"/tmp/vector_bench_pseudo_num_{num}_dim_{dim}_query.pickle"
        )

        if self.record_path.is_file():
            logger.info("load pseudo dataset from %s", self.record_path)
            with open(self.record_path, "rb") as file:
                self.records = np.load(file, allow_pickle=False, fix_imports=False)
        else:
            self.records = np.random.rand(self.record_num, self.dim)
            logger.info("dump pseudo dataset to %s", self.record_path)
            with open(self.record_path, "wb") as file:
                np.save(file, self.records, allow_pickle=False, fix_imports=False)
            logger.info("rm outdated query file %s", self.query_path)
            self.query_path.unlink(missing_ok=True)

        if self.query_path.is_file():
            logger.info("load pseudo query from %s", self.query_path)
            with open(self.query_path, "rb") as file:
                self.queries = pickle.load(file)
        else:
            logger.info("generate pseudo queries")
            self.queries = []
            for _ in tqdm(range(self.query_num)):
                query = np.random.rand(self.dim)
                distances = [self.distance(query, x) for x in self.records]
                nearest = np.argsort(distances)
                self.queries.append(
                    Query(
                        vector=query,
                        expect_ids=nearest[: self.top_k],
                        expect_scores=[distances[i] for i in nearest[: self.top_k]],
                    )
                )
            logger.info("dump pseudo query to %s", self.query_path)
            with open(self.query_path, "wb") as file:
                pickle.dump(self.queries, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_config(cls, config: DatasetConfig) -> BaseReader:
        return cls(config.num, config.vector_dim, config.distance)

    def read_record(self) -> Iterator[Record]:
        for i in range(self.record_num):
            yield Record(id=i, vector=self.records[i])

    def read_query(self) -> Iterator[Query]:
        for query in self.queries:
            yield query


if __name__ == "__main__":
    reader = PseudoReader()
    reader.query_num = 5
    for query in reader.read_query():
        print(query.expect_ids, query.expect_scores)
