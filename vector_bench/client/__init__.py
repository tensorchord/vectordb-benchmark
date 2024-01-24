from vector_bench.client.pgvecto_rs import PgVectorsClient
from vector_bench.client.pgvector import PgvectorClient
from vector_bench.spec import EnumSelector


class DataBaseClient(EnumSelector):
    PGVECTO_RS = PgVectorsClient
    PGVECTOR = PgvectorClient
