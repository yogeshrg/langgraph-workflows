"""LangGraph Workflows configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "LangGraph Workflows"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8004

    # LangGraph
    checkpoint_dir: Path = Field(default=Path("./data/checkpoints"))
    thread_pool_size: int = 10

    # LLM
    default_model: str = "gpt-4o-mini"
    default_provider: str = "openai"  # openai, anthropic
    default_temperature: float = 0.7

    # Streaming
    stream_tokens: bool = True
    stream_mode: str = "messages"  # messages, values, updates

    # Visualization
    enable_visualizer: bool = True
    visualizer_port: int = 8080

    # Debug
    debug_checkpoints: bool = True
    max_debug_history: int = 1000


settings = Settings()