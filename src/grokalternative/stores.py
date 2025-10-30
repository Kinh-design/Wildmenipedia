from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient

from .settings import get_settings


@dataclass
class KG:
    uri: str
    user: str
    password: str
    _driver: Optional[Driver] = None

    @classmethod
    def from_env(cls) -> "KG":
        s = get_settings()
        return cls(uri=s.NEO4J_URL, user=s.NEO4J_USER, password=s.NEO4J_PASSWORD)

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def upsert_triple(self, s: str, p: str, o: str, meta: Dict[str, Any] | None = None) -> None:
        query = (
            "MERGE (s:Entity {id:$s}) MERGE (o:Entity {id:$o}) "
            "MERGE (s)-[r:REL {pred:$p}]->(o) SET r += $meta"
        )
        with self.driver.session() as sess:
            sess.run(query, s=s, p=p, o=o, meta=meta or {})

    def neighbors(self, node_id: str, limit: int = 10) -> list[dict[str, Any]]:
        query = (
            "MATCH (s:Entity {id:$id})-[r:REL]->(o:Entity) "
            "RETURN s.id AS s, r.pred AS p, o.id AS o, r AS meta LIMIT $limit"
        )
        with self.driver.session() as sess:
            res = sess.run(query, id=node_id, limit=limit)
            return [dict(record) for record in res]


@dataclass
class VS:
    host: str
    port: int
    _client: Optional[QdrantClient] = None

    @classmethod
    def from_env(cls) -> "VS":
        s = get_settings()
        return cls(host=s.QDRANT_HOST, port=s.QDRANT_PORT)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client
