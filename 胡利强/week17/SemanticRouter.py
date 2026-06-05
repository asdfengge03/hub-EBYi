from redisvl import RedisVectorIndex
from redisvl.schema import IndexSchema
from typing import Optional, dict

class SemanticRouter:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "semantic_router",
        vector_dim: int = 1536,
        threshold: float = 0.8
    ):
        self.schema = IndexSchema.from_dict({
            "index": {"name": index_name, "prefix": f"{index_name}:"},
            "fields": [
                {"name": "name", "type": "tag"},
                {"name": "description", "type": "text"},
                {"name": "handler", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {
                    "dims": vector_dim,
                    "distance_metric": "cosine",
                    "algorithm": "flat"
                }}
            ]
        })
        self.index = RedisVectorIndex(self.schema, redis_url=redis_url)
        self.threshold = threshold

    def add_route(
        self,
        name: str,
        description: str,
        handler: str,
        vector: list
    ):
        self.index.store(
            key=name,
            properties={
                "name": name,
                "description": description,
                "handler": handler,
                "vector": vector
            }
        )

    def route(
        self,
        query_vector: list
    ) -> Optional[dict]:
        res = self.index.search(
            query_vector=query_vector,
            k=1,
            distance_threshold=1 - self.threshold
        )
        return res[0] if res else None
