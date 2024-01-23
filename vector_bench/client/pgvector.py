from __future__ import annotations

from struct import pack, unpack

import msgspec
import numpy as np
import psycopg
from psycopg import sql
from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format
from psycopg.types import TypeInfo
from psycopg.types.json import Jsonb

from vector_bench.client.base import BaseClient
from vector_bench.log import logger
from vector_bench.spec import DatabaseConfig, Distance, Record


def from_db(value):
    # could be ndarray if already cast by lower-level driver
    if value is None or isinstance(value, np.ndarray):
        return value

    return np.array(value[1:-1].split(","), dtype=np.float32)


def from_db_binary(value):
    if value is None:
        return value

    (dim, unused) = unpack(">HH", value[:4])
    return np.frombuffer(value, dtype=">f", count=dim, offset=4).astype(
        dtype=np.float32
    )


def to_db(value, dim=None):
    if value is None:
        return value

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("expected ndim to be 1")

        if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(
            value.dtype, np.floating
        ):
            raise ValueError("dtype must be numeric")

        value = value.tolist()

    if dim is not None and len(value) != dim:
        raise ValueError("expected %d dimensions, not %d" % (dim, len(value)))

    return "[" + ",".join([str(float(v)) for v in value]) + "]"


def to_db_binary(value):
    if value is None:
        return value

    value = np.asarray(value, dtype=">f")

    if value.ndim != 1:
        raise ValueError("expected ndim to be 1")

    return pack(">HH", value.shape[0], 0) + value.tobytes()


class VectorDumper(Dumper):
    def dump(self, obj):
        return to_db(obj).encode("utf8")


class VectorBinaryDumper(VectorDumper):
    format = Format.BINARY

    def dump(self, obj):
        return to_db_binary(obj)


class VectorLoader(Loader):
    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return from_db(data.decode("utf8"))


class VectorBinaryLoader(VectorLoader):
    format = Format.BINARY

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return from_db_binary(data)


def register_vector(context):
    info = TypeInfo.fetch(context, "vector")
    register_vector_info(context, info)


async def register_vector_async(context):
    info = await TypeInfo.fetch(context, "vector")
    register_vector_info(context, info)


def register_vector_info(context, info):
    if info is None:
        raise psycopg.ProgrammingError("vector type not found in the database")
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type("", (VectorDumper,), {"oid": info.oid})
    binary_dumper = type("", (VectorBinaryDumper,), {"oid": info.oid})

    adapters = context.adapters
    adapters.register_dumper("numpy.ndarray", text_dumper)
    adapters.register_dumper("numpy.ndarray", binary_dumper)
    adapters.register_loader(info.oid, VectorLoader)
    adapters.register_loader(info.oid, VectorBinaryLoader)


DISTANCE_TO_METHOD = {
    Distance.EUCLIDEAN: "vector_l2_ops",
    Distance.COSINE: "vector_cosine_ops",
    Distance.DOT_PRODUCT: "vector_ip_ops",
}

DISTANCE_TO_OP = {
    Distance.EUCLIDEAN: "<->",
    Distance.COSINE: "<=>",
    Distance.DOT_PRODUCT: "<#>",
}


class PgvectorClient(BaseClient):
    LOAD_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
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
USING hnsw (emb {});
"""
    INSERT = """
INSERT INTO {} (id, emb, metadata)
VALUES (%s, %s, %s)
"""
    SEARCH = """
SELECT id, emb, metadata, emb {} %s AS score
FROM {}
ORDER BY score LIMIT %s;
"""

    url: str
    dim: int
    table: str
    distance: Distance

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> PgvectorClient:
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
            conn.execute(
                sql.SQL(self.CREATE_TABLE).format(sql.Identifier(self.table), self.dim)
            )
            conn.commit()

    def indexing(self):
        with psycopg.connect(self.url) as conn:
            conn.execute(
                sql.SQL(self.CREATE_INDEX).format(
                    sql.Identifier(self.table),
                    sql.Identifier(DISTANCE_TO_METHOD[self.distance]),
                )
            )
            conn.commit()

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
                sql.SQL(self.SEARCH).format(
                    sql.Identifier(DISTANCE_TO_OP[self.distance]),
                    sql.Identifier(self.table),
                ),
                (vector, top_k),
            )
            result = cur.fetchall()
            conn.commit()
        return [Record(id=row[0], vector=row[1], metadata=row[2]) for row in result]
