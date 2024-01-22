from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from itertools import islice
from time import perf_counter
from typing import Iterable, Optional

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
        self,
        db_config: DatabaseConfig,
        dataset_config: DatasetConfig,
        worker_num: Optional[int] = None,
    ) -> None:
        self.client: BaseClient = DataBaseClient.select(db_config.name).from_config(
            db_config
        )
        self.reader: BaseReader = DatasetReader.select(
            dataset_config.type.value
        ).from_config(dataset_config)
        self.worker_num: Optional[int] = worker_num
        self.query_result: BenchmarkResult = BenchmarkResult()

    def insert(self):
        logger.info("inserting records...")
        epoch_size, batch_size = 10000, 20
        with ThreadPoolExecutor(self.worker_num) as executor:
            logger.info("using %s executors", executor._max_workers)
            for i, epoch in enumerate(batched(self.reader.read_record(), epoch_size)):
                epoch_start = perf_counter()
                for future in as_completed(
                    executor.submit(self.client.insert_batch, records)
                    for records in batched(epoch, batch_size)
                ):
                    try:
                        future.result()
                    except (CancelledError, TimeoutError) as err:
                        logger.exception("failed to insert records", exc_info=err)
                logger.info(
                    "finished %s records with RPS(%.3f)",
                    (i + 1) * epoch_size,
                    epoch_size / (perf_counter() - epoch_start),
                )

    def _query_helper(self, query: Query):
        start_time = perf_counter()
        records = self.client.query(query.vector, len(query.expect_ids))
        elapsed = perf_counter() - start_time
        self.query_result.query += 1
        self.query_result.latency.append(elapsed)
        self.query_result.precision.append(
            len(
                set(query.expect_ids).intersection(set(record.id for record in records))
            )
            / len(query.expect_ids)
        )

    def query(self) -> BenchmarkResult:
        logger.info("querying...")
        epoch_size = 100
        with ThreadPoolExecutor(self.worker_num) as executor:
            logger.info("using %s executors", executor._max_workers)
            start = perf_counter()
            for i, epoch in enumerate(batched(self.reader.read_query(), epoch_size)):
                epoch_start = perf_counter()
                for future in as_completed(
                    executor.submit(self._query_helper, query) for query in epoch
                ):
                    try:
                        future.result()
                    except (CancelledError, TimeoutError) as err:
                        logger.exception("failed to query", exc_info=err)
                logger.info(
                    "finished %s queries with RPS (%.3f)",
                    (i + 1) * epoch_size,
                    epoch_size / (perf_counter() - epoch_start),
                )
            self.query_result.total_second = perf_counter() - start
            self.query_result.worker_num = executor._max_workers

        return self.query_result
