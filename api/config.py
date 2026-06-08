"""
api/config.py
Centralised settings loaded from .env
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_secret_key: str = "dev-secret-key"
    api_port: int = 8000
    debug: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()