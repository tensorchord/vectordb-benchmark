from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterator

import httpx
from tqdm import tqdm

from vector_bench.log import logger
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
        cls.download_dataset(config)
        pass

    @staticmethod
    def download_dataset(config: DatasetConfig):
        path = Path(config.path)
        if path.is_file():
            logger.info("Dataset already exists at %s", path)
            return

        logger.info("Downloading dataset from %s", config.link)
        path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", config.link) as resp:
            resp.raise_for_status()
            size = int(resp.headers.get("Content-Length", 0))
            with path.open("wb") as f, tqdm.wrapattr(
                f,
                "write",
                total=size,
                bytes=True,
            ) as file:
                for chunk in resp.iter_bytes():
                    file.write(chunk)
