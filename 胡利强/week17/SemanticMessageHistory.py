from redisvl import RedisVectorIndex
from redisvl.schema import IndexSchema
from typing import list, dict
import uuid

class SemanticMessageHistory:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "semantic_history",
        session_id: str = None,
        vector_dim: int = 1536
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.schema = IndexSchema.from_dict({
            "index": {"name": index_name, "prefix": f"{index_name}:"},
            "fields": [
                {"name": "session_id", "type": "tag"},
                {"name": "role", "type": "tag"},
                {"name": "content", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {
                    "dims": vector_dim,
                    "distance_metric": "cosine",
                    "algorithm": "flat"
                }}
            ]
        })
        self.index = RedisVectorIndex(self.schema, redis_url=redis_url)

    def add_message(
        self,
        role: str,
        content: str,
        vector: list
    ):
        key = str(uuid.uuid4())
        self.index.store(
            key=key,
            properties={
                "session_id": self.session_id,
                "role": role,
                "content": content,
                "vector": vector
            }
        )

    def get_relevant_messages(
        self,
        query_vector: list,
        k: int = 5
    ) -> list[dict]:
        filters = f"@session_id:{{{self.session_id}}}"
        return self.index.search(
            query_vector=query_vector,
            k=k,
            filter=filters
        )

    def clear(self):
        self.index.delete_by_tags(
            tags={"session_id": self.session_id}
        )
