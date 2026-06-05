from redisvl import RedisVectorIndex
from redisvl.schema import IndexSchema
from typing import Optional, Any
import hashlib
import json

class EmbeddingsCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "embeddings_cache",
        vector_dim: int = 1536
    ):
        self.schema = IndexSchema.from_dict({
            "index": {"name": index_name, "prefix": f"{index_name}:"},
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "embedding", "type": "vector", "attrs": {
                    "dims": vector_dim,
                    "distance_metric": "cosine",
                    "algorithm": "flat"
                }}
            ]
        })
        self.index = RedisVectorIndex(self.schema, redis_url=redis_url)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        key = self._hash(text)
        res = self.index.fetch(key)
        return res.get("embedding") if res else None

    def set(self, text: str, embedding: list):
        key = self._hash(text)
        self.index.store(
            key=key,
            properties={
                "text": text,
                "embedding": embedding
            }
        )

    def exists(self, text: str) -> bool:
        return self.get(text) is not None
