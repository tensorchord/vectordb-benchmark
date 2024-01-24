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
            obj = f"[{','.join(map(str, obj))}]"
        return str(obj).replace(" ", "")


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


DISTANCE_TO_METHOD = {
    Distance.EUCLIDEAN: "vector_l2_ops",
    Distance.COSINE: "vector_cos_ops",
    Distance.DOT_PRODUCT: "vector_dot_ops",
}

DISTANCE_TO_OP = {
    Distance.EUCLIDEAN: "<->",
    Distance.COSINE: "<=>",
    Distance.DOT_PRODUCT: "<#>",
}


class PgVectorsClient(BaseClient):
    LOAD_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vectors;
"""
    CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    emb vector({dim}) NOT NULL,
    metadata JSONB NOT NULL
);
"""
    CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS vector_search
ON {table}
USING vectors (emb {method});
"""
    INSERT = """
INSERT INTO {table} (id, emb, metadata)
VALUES (%s, %s, %s)
"""
    # `psycopg` will panic if `op` is quoted
    SEARCH = """
SELECT id, emb, metadata, emb {op} %s AS score
FROM {{table}}
ORDER BY score LIMIT %s;
"""

    url: str
    dim: int
    table: str
    distance: Distance

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> PgVectorsClient:
        client = cls()
        client.dim = config.vector_dim
        client.url = config.url
        client.distance = config.distance
        client.table = f"{config.table}_pgvecto_rs"

        logger.info(
            "initializing pgvecto.rs database(table=%s, dim=%s)...",
            client.table,
            client.dim,
        )
        client.init_db()
        client.indexing()
        return client

    def init_db(self):
        with psycopg.connect(self.url) as conn:
            conn.execute(self.LOAD_EXTENSION)
            register_vector(conn)
            # init SQL
            self.sql_create_table = (
                sql.SQL(self.CREATE_TABLE)
                .format(table=sql.Identifier(self.table), dim=self.dim)
                .as_string(conn)
            )
            self.sql_create_index = (
                sql.SQL(self.CREATE_INDEX)
                .format(
                    table=sql.Identifier(self.table),
                    method=sql.Identifier(DISTANCE_TO_METHOD[self.distance]),
                )
                .as_string(conn)
            )
            self.sql_insert = (
                sql.SQL(self.INSERT)
                .format(table=sql.Identifier(self.table))
                .as_string(conn)
            )
            self.sql_query = (
                sql.SQL(self.SEARCH.format(op=DISTANCE_TO_OP[self.distance]))
                .format(
                    table=sql.Identifier(self.table),
                )
                .as_string(conn)
            )
            # create table
            conn.execute(self.sql_create_table)
            conn.commit()

    def indexing(self):
        with psycopg.connect(self.url) as conn:
            conn.execute(self.sql_create_index)
            conn.commit()

    async def insert(self, record: Record):
        async with await psycopg.AsyncConnection.connect(self.url) as conn:
            register_vector_async(conn)
            await conn.execute(
                self.sql_insert,
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
                    self.sql_insert,
                    (
                        record.id,
                        record.vector,
                        Jsonb(record.metadata or {}, dumps=msgspec.json.encode),
                    ),
                )
            conn.commit()

    def query(self, vector: np.ndarray, top_k: int = 5) -> list[Record]:
        with psycopg.connect(self.url) as conn:
            register_vector(conn)
            result = conn.execute(self.sql_query, (vector, top_k)).fetchall()
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
