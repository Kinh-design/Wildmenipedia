from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    LLM_PROVIDER: str = "local"
    LLM_MODEL: str = "llama3"

    ENV: str = "dev"

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
