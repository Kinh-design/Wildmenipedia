from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

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

    def neighbors(self, node_id: str, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        query = (
            "MATCH (s:Entity {id:$id})-[r:REL]->(o:Entity) "
            "RETURN s.id AS s, r.pred AS p, o.id AS o, r AS meta "
            "SKIP $offset LIMIT $limit"
        )
        with self.driver.session() as sess:
            res = sess.run(query, id=node_id, limit=limit, offset=offset)
            return [dict(record) for record in res]

    def count_neighbors(self, node_id: str) -> int:
        query = (
            "MATCH (s:Entity {id:$id})-[r:REL]->(o:Entity) "
            "RETURN count(r) AS cnt"
        )
        try:
            with self.driver.session() as sess:
                rec = sess.run(query, id=node_id).single()
                if not rec:
                    return 0
                data = dict(rec)
                return int(data.get("cnt", 0))
        except Exception:
            return 0

    def get_entity_props(self, entity_id: str) -> Dict[str, Any]:
        """Return basic properties for an entity node: name and aliases.

        If the node does not exist, returns an empty dict.
        """
        query = "MATCH (e:Entity {id:$id}) RETURN e.name AS name, e.aliases AS aliases, e.types AS types LIMIT 1"
        try:
            with self.driver.session() as sess:
                rec = sess.run(query, id=entity_id).single()
                if not rec:
                    return {}
                name = rec.get("name")
                aliases = rec.get("aliases") or []
                types = rec.get("types") or []
                return {"name": name, "aliases": aliases, "types": types}
        except Exception:
            return {}

    def ensure_schema(self) -> None:
        """Create minimal schema: unique id on :Entity and index on REL.pred.

        Safe to call multiple times; uses IF NOT EXISTS and swallows errors
        to avoid failing when permissions are limited.
        """
        stmts = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX rel_pred_idx IF NOT EXISTS FOR ()-[r:REL]-() ON (r.pred)",
        ]
        try:
            with self.driver.session() as sess:
                for s in stmts:
                    sess.run(s)
        except Exception:
            # best-effort; ignore if not permitted
            pass

    def set_label(self, entity_id: str, label: str) -> None:
        query = "MERGE (e:Entity {id:$id}) SET e.name = $label"
        with self.driver.session() as sess:
            sess.run(query, id=entity_id, label=label)

    def add_alias(self, entity_id: str, alias: str) -> None:
        query = (
            "MERGE (e:Entity {id:$id}) "
            "SET e.aliases = CASE "
            "WHEN e.aliases IS NULL THEN [$alias] "
            "WHEN NOT $alias IN e.aliases THEN e.aliases + $alias "
            "ELSE e.aliases END"
        )
        with self.driver.session() as sess:
            sess.run(query, id=entity_id, alias=alias)

    def add_type(self, entity_id: str, type_uri: str) -> None:
        query = (
            "MERGE (e:Entity {id:$id}) "
            "SET e.types = CASE "
            "WHEN e.types IS NULL THEN [$type] "
            "WHEN NOT $type IN e.types THEN e.types + $type "
            "ELSE e.types END"
        )
        with self.driver.session() as sess:
            sess.run(query, id=entity_id, type=type_uri)


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

    # Vector utilities
    def ensure_collection(self, name: str = "default", dim: int = 256, distance: str = "Cosine") -> None:
        try:
            self.client.get_collection(name)
            return
        except Exception:
            pass
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=getattr(qm.Distance, distance)),
        )

    def upsert_point(self, collection: str, point_id: str, vector: list[float], payload: Optional[dict[str, Any]] = None) -> None:
        self.ensure_collection(collection, dim=len(vector))
        self.client.upsert(
            collection_name=collection,
            points=[qm.PointStruct(id=point_id, vector=vector, payload=payload or {})],
        )

    def search(self, collection: str, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.ensure_collection(collection, dim=len(vector))
        res = self.client.search(collection_name=collection, query_vector=vector, limit=top_k)
        out: list[dict[str, Any]] = []
        for p in res:
            out.append({
                "id": getattr(p, "id", None),
                "score": getattr(p, "score", None),
                "payload": getattr(p, "payload", {}),
            })
        return out
