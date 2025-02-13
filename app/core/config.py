from pydantic_settings import BaseSettings

from pydantic import Field


class Settings(BaseSettings):
    environment: str = "production"
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    classifier_path: str = Field("/app/data/models/query_classifier", env="CLASSIFIER_PATH")
    api_port: int = Field(8000, env="API_PORT")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    debug: bool = Field(False, env="DEBUG")
    rate_limit: int = Field(100, env="RATE_LIMIT")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_ignore_empty = True
        extra = "ignore"


settings = Settings()
