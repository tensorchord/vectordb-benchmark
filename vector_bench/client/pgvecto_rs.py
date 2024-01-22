from __future__ import annotations

import msgspec
import numpy as np
import psycopg
from psycopg import sql
from psycopg.adapt import Dumper, Loader
from psycopg.types import TypeInfo
from psycopg.types.json import Jsonb

from vector_bench.client.base import BaseClient
from vector_bench.log import logger
from vector_bench.spec import DatabaseConfig, Distance, Record


class VectorDumper(Dumper):
    def dump(self, obj):
        if isinstance(obj, np.ndarray):
            return str(obj).replace(" ", ",").encode()
        return str(obj).replace(" ", "").encode()


class VectorLoader(Loader):
    def load(self, buf):
        if isinstance(buf, memoryview):
            buf = bytes(buf)
        return np.array(buf.decode()[1:-1].split(","), dtype=np.float32)


async def register_vector_async(conn: psycopg.AsyncConnection):
    info = await TypeInfo.fetch(conn=conn, name="vector")
    register_vector_type(conn, info)


def register_vector(conn: psycopg.Connection):
    info = TypeInfo.fetch(conn=conn, name="vector")
    register_vector_type(conn, info)


def register_vector_type(conn: psycopg.Connection, info: TypeInfo):
    if info is None:
        raise ValueError("vector type not found")
    info.register(conn)

    class VectorTextDumper(VectorDumper):
        oid = info.oid

    adapters = conn.adapters
    adapters.register_dumper(list, VectorTextDumper)
    adapters.register_dumper(np.ndarray, VectorTextDumper)
    adapters.register_loader(info.oid, VectorLoader)


DISTANCE_TO_OP = {
    Distance.EUCLIDEAN: "vector_l2_ops",
    Distance.COSINE: "vector_cos_ops",
    Distance.DOT_PRODUCT: "vector_dot_ops",
}


class PgVectorsClient(BaseClient):
    LOAD_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vectors;
"""
    CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {} (
    id SERIAL PRIMARY KEY,
    emb vector({}) NOT NULL,
    metadata JSONB NOT NULL
);
"""
    CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS vector_search
ON {}
USING vectors (emb {});
"""
    INSERT = """
INSERT INTO {} (id, emb, metadata)
VALUES (%s, %s, %s)
"""
    SEARCH = """
SELECT id, emb, metadata, emb <-> %s AS score
FROM {}
ORDER BY score LIMIT %s;
"""

    url: str
    dim: int
    table: str

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> PgVectorsClient:
        client = cls()
        client.dim = config.vector_dim
        client.url = config.url
        client.table = config.table

        logger.info(
            "initializing pgvecto.rs database(table=%s, dim=%s)...",
            client.table,
            client.dim,
        )
        client.init_db()
        client.indexing(config.distance)
        return client

    def init_db(self):
        with psycopg.connect(self.url) as conn:
            conn.execute(self.LOAD_EXTENSION)
            register_vector(conn)
            conn.execute(
                sql.SQL(self.CREATE_TABLE).format(sql.Identifier(self.table), self.dim)
            )
            conn.commit()

    def indexing(self, distance: Distance):
        with psycopg.connect(self.url) as conn:
            conn.execute(
                sql.SQL(self.CREATE_INDEX).format(
                    sql.Identifier(self.table),
                    sql.Identifier(DISTANCE_TO_OP[distance]),
                )
            )
            conn.commit()

    async def insert(self, record: Record):
        async with await psycopg.AsyncConnection.connect(self.url) as conn:
            register_vector_async(conn)
            await conn.execute(
                sql.SQL(self.INSERT).format(sql.Identifier(self.table)),
                (
                    record.id,
                    record.vector,
                    Jsonb(record.metadata or {}, dumps=msgspec.json.encode),
                ),
            )
            await conn.commit()

    def insert_batch(self, records: list[Record]):
        with psycopg.connect(self.url) as conn:
            register_vector(conn)
            conn.commit()
            for record in records:
                conn.execute(
                    sql.SQL(self.INSERT).format(sql.Identifier(self.table)),
                    (
                        record.id,
                        record.vector,
                        Jsonb(record.metadata or {}, dumps=msgspec.json.encode),
                    ),
                )
            conn.commit()

    def query(self, vector: list[float], top_k: int = 5) -> list[Record]:
        with psycopg.connect(self.url) as conn:
            register_vector(conn)
            cur = conn.execute(
                sql.SQL(self.SEARCH).format(sql.Identifier(self.table)), (vector, top_k)
            )
            result = cur.fetchall()
            conn.commit()
        return [Record(id=row[0], vector=row[1], metadata=row[2]) for row in result]


if __name__ == "__main__":
    client = PgVectorsClient.from_config(DatabaseConfig(vector_dim=1024))

    from time import perf_counter

    start = perf_counter()
    client.insert_batch(
        [
            Record(4, [0.5] * 1024),
        ]
    )
    print(perf_counter() - start)
