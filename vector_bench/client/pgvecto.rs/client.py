from __future__ import annotations

import msgspec
import psycopg
from psycopg.types.json import Jsonb

from vector_bench.spec import DatabaseConfig, Record


class PgVectorsClient:
    LOAD_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vectors;
"""
    CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    emb vector({}) NOT NULL,
    metadata JSONB NOT NULL
);
"""
    CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS vector_search
ON chunks
USING vectors (emb vector_l2_ops);
"""
    INSERT = """
INSERT INTO chunks (id, emb) VALUES (%s, %s)
"""
    SEARCH = """
SELECT id, emb, metadata, emb <-> %s AS score
FROM chunks
ORDER BY score LIMIT %s;
"""

    url: str
    dim: int

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> PgVectorsClient:
        client = cls()
        client.dim = config.dataset.vector_dim
        client.url = config.url
        client.init_db()
        client.indexing()
        return client

    def init_db(self):
        with psycopg.connect(self.url) as conn, conn.cursor() as cur:
            cur.execute(self.LOAD_EXTENSION)
            cur.execute(psycopg.sql.SQL(self.CREATE_TABLE).format(self.dim))
            conn.commit()

    def indexing(self):
        with psycopg.connect(self.url) as conn, conn.cursor() as cur:
            cur.execute(self.CREATE_INDEX)
            conn.commit()

    async def insert(self, record: Record):
        async with await psycopg.AsyncConnection.connect(
            self.url
        ) as conn, conn.cursor() as cur:
            await cur.execute(
                self.INSERT,
                (
                    record.id,
                    str(record.vector),
                    Jsonb(record.metadata, dumps=msgspec.json.encode),
                ),
            )
            await conn.commit()

    async def query(self, vector: list[float], top_k: int = 5) -> list[Record]:
        async with await psycopg.AsyncConnection.connect(
            self.url
        ) as conn, conn.cursor() as cur:
            await cur.execute(self.SEARCH, (str(vector), top_k))
            result = await cur.fetchall()
            await conn.commit()
        return [Record(id=row[0], vector=row[1], metadata=row[2]) for row in result]
