from typing import Iterator

import numpy as np

from vector_bench.spec import Distance, Query, Record


class PseudoReader:
    def __init__(self) -> None:
        self.record_num = 100_000
        self.query_num = 100
        self.dim = 128
        self.top_k = 10
        self.distance = Distance.DOT_PRODUCT

        self.vectors = np.random.rand(self.record_num, self.dim)

    def read_record(self) -> Iterator[Record]:
        for i in range(self.record_num):
            yield Record(id=i, vector=self.vectors[i].tolist())

    def read_query(self) -> Iterator[Query]:
        for _ in range(self.query_num):
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
