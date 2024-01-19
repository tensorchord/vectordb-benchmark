from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from itertools import islice
from time import perf_counter
from typing import Iterable

from tqdm import tqdm

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
        self.reader: BaseReader = DatasetReader.select(
            dataset_config.type.value
        ).from_config(dataset_config)
        self.result: BenchmarkResult = BenchmarkResult()

    def insert(self):
        logger.info("loading records...")
        records = list(self.reader.read_record())
        logger.info("inserting records...")
        batch_size = 100
        with tqdm(total=len(records)) as bar, ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.client.insert_batch, records)
                for records in batched(records, batch_size)
            ]
            logger.info("using %s executors", executor._max_workers)
            for future in as_completed(futures):
                try:
                    future.result()
                except (CancelledError, TimeoutError) as err:
                    logger.exception("failed to insert records", exc_info=err)
                bar.update(batch_size)

    def _query_helper(self, query: Query):
        start_time = perf_counter()
        records = self.client.query(query.vector, len(query.expect_ids))
        elapsed = perf_counter() - start_time
        self.result.query += 1
        self.result.latency.append(elapsed)
        self.result.precision.append(
            len(
                set(query.expect_ids).intersection(set(record.id for record in records))
            )
            / len(query.expect_ids)
        )

    def query(self) -> BenchmarkResult:
        logger.info("loading queries...")
        queries = list(self.reader.read_query())
        logger.info("querying...")
        with tqdm(total=len(queries)) as bar, ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._query_helper, query) for query in queries]
            logger.info("using %s executors", executor._max_workers)
            for future in as_completed(futures):
                try:
                    future.result()
                except (CancelledError, TimeoutError) as err:
                    logger.exception("failed to query", exc_info=err)
                bar.update(1)

        return self.result
