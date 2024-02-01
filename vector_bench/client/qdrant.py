from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance as QdrantDistance
from qdrant_client.models import PointStruct, ScoredPoint, VectorParams

from vector_bench.client.base import BaseClient
from vector_bench.spec import DatabaseConfig, Distance, Record

DISTANCE_TO_QDRANT = {
    Distance.COSINE: QdrantDistance.COSINE,
    Distance.EUCLIDEAN: QdrantDistance.EUCLID,
    Distance.DOT_PRODUCT: QdrantDistance.DOT,
}


class QdrantVectorClient(BaseClient):
    dim: int
    url: str
    table: str
    distance: Distance

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> QdrantVectorClient:
        cls.dim = config.vector_dim
        cls.url = config.url
        cls.table = f"{config.table}_qdrant"
        cls.distance = config.distance

        cls = QdrantVectorClient()
        cls.init_db()
        return cls

    def init_db(self):
        self.client = QdrantClient(url=self.url)
        collections_response = self.client.get_collections()
        for collection in collections_response.collections:
            if collection.name == self.table:
                # already exists, return
                return

        self.client.create_collection(
            collection_name=self.table,
            vectors_config=VectorParams(
                size=self.dim,
                distance=DISTANCE_TO_QDRANT[self.distance.__func__],
            ),
        )

    def insert_batch(self, records: list[Record]):
        self.client.upsert(
            collection_name=self.table,
            points=[
                PointStruct(
                    id=record.id, vector=record.vector.tolist(), payload=record.metadata
                )
                for record in records
            ],
        )

    def query(self, vector: list[float], top_k: int = 10) -> list[Record]:
        points: list[ScoredPoint] = self.client.search(
            collection_name=self.table,
            query_vector=vector,
            limit=top_k,
        )
        return [
            Record(id=point.id, vector=point.vector, metadata=point.payload)
            for point in points
        ]
