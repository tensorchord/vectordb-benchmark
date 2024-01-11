from typing import Iterator

import h5py

from vector_bench.spec import Query, Record


class HDF5Reader:
    def __init__(self, path: str) -> None:
        self.path = path

    def read_record(self) -> Iterator[Record]:
        with h5py.File(self.path, "r") as file:
            for i, vec in enumerate(file["train"]):
                yield Record(
                    id=i,
                    vector=vec.tolist(),
                    metadata=None,
                )

    def read_query(self) -> Iterator[Query]:
        with h5py.File(self.path, "r") as file:
            for vec, ids, scores in zip(
                file["test"], file["neighbors"], file["distances"]
            ):
                yield Query(
                    vector=vec.tolist(),
                    expect_ids=ids.tolist(),
                    expect_scores=scores.tolist(),
                    metadata=None,
                )