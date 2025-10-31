from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    LLM_PROVIDER: str = "local"
    LLM_MODEL: str = "llama3"

    ENV: str = "dev"

    # Scraper toggles (optional dependencies are not required if disabled)
    SCRAPER_ENABLE_HTTPX: bool = True
    SCRAPER_ENABLE_CLOUDSCRAPER: bool = False
    SCRAPER_ENABLE_SCRAPY: bool = False
    SCRAPER_ENABLE_CRAWL4AI: bool = False
    SCRAPER_ENABLE_CRAWLEE: bool = False
    SCRAPER_ENABLE_FIRECRAWL: bool = False

    # Pydantic v2 config provided via model_config above


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
