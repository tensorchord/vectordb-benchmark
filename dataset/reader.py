from typing import Iterator

import h5py

from spec.vector import Query, Record


class H5Reader:
    def __init__(self, path: str) -> None:
        self.path = path

    def read_train(self) -> Iterator[Record]:
        with h5py.File(self.path, "r") as file:
            for i, vec in enumerate(file["train"]):
                yield Record(
                    id=i,
                    vector=vec.tolist(),
                    metadata=None,
                )

    def read_test(self) -> Iterator[Query]:
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
