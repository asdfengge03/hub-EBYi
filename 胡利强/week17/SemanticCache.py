from redisvl import RedisVectorIndex
from redisvl.schema import IndexSchema
from typing import Optional, Any
import hashlib

class SemanticCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "semantic_cache",
        vector_dim: int = 1536,
        threshold: float = 0.9
    ):
        self.schema = IndexSchema.from_dict({
            "index": {"name": index_name, "prefix": f"{index_name}:"},
            "fields": [
                {"name": "query", "type": "text"},
                {"name": "response", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {
                    "dims": vector_dim,
                    "distance_metric": "cosine",
                    "algorithm": "flat"
                }}
            ]
        })
        self.index = RedisVectorIndex(self.schema, redis_url=redis_url)
        self.threshold = threshold

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query_vector: list) -> Optional[Any]:
        results = self.index.search(
            query_vector=query_vector,
            k=1,
            distance_threshold=1 - self.threshold
        )
        if not results:
            return None
        return results[0]["response"]

    def set(self, query: str, query_vector: list, response: str):
        key = self._key(query)
        self.index.store(
            key=key,
            properties={
                "query": query,
                "vector": query_vector,
                "response": response
            }
        )

    def clear(self):
        self.index.flush()
