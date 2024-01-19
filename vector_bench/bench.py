from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from itertools import islice
from time import perf_counter
from typing import Iterable

from vector_bench.client import DataBaseClient
from vector_bench.client.base import BaseClient
from vector_bench.dataset import DatasetReader
from vector_bench.dataset.base import BaseReader
from vector_bench.log import logger
from vector_bench.spec import BenchmarkResult, DatabaseConfig, DatasetConfig, Query


def batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class Benchmark:
    def __init__(
        self, db_config: DatabaseConfig, dataset_config: DatasetConfig
    ) -> None:
        self.client: BaseClient = DataBaseClient.select(db_config.name).from_config(
            db_config
        )
        self.reader: BaseReader = DatasetReader.select(dataset_config.name).from_config(
            dataset_config
        )
        self.result: BenchmarkResult = BenchmarkResult()

    def insert(self):
        logger.info("inserting records...")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.client.insert_batch, records)
                for records in batched(self.reader.read_record(), 100)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except (CancelledError, TimeoutError) as err:
                    logger.exception("failed to insert records", exc_info=err)

    def _recall(self, query: Query):
        start_time = perf_counter()
        records = self.client.query(query.vector, len(query.expect_ids))
        elapsed = perf_counter() - start_time
        self.result.query += 1
        self.result.latency.append(elapsed)
        self.result.recall.append(
            len(
                set(query.expect_ids).intersection(set(record.id for record in records))
            )
            / len(query.expect_ids)
        )

    def query(self) -> BenchmarkResult:
        logger.info("querying...")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._recall, query)
                for query in self.reader.read_query()
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except (CancelledError, TimeoutError) as err:
                    logger.exception("failed to query", exc_info=err)

        return self.result
