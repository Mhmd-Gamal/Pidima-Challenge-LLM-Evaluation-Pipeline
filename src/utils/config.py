"""
Configuration management using environment variables.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model settings
    MODEL_NAME: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="HuggingFace model identifier"
    )
    MODEL_CACHE_DIR: str = Field(
        default="./models",
        description="Directory to cache model weights"
    )
    DEVICE: str = Field(
        default="cpu",
        description="Device to run model on (cpu/cuda)"
    )
    USE_4BIT: bool = Field(
        default=False,
        description="Use 4-bit quantization (GPU only)"
    )
    
    # Generation settings
    MAX_LENGTH: int = Field(
        default=512,
        description="Maximum sequence length"
    )
    DEFAULT_MAX_TOKENS: int = Field(
        default=100,
        description="Default max tokens for generation"
    )
    
    # API settings
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    API_PORT: int = Field(
        default=8000,
        description="API port"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Evaluation settings
    EVAL_TEMPERATURE: float = Field(
        default=0.0,
        description="Temperature for evaluation (0 for deterministic)"
    )
    EVAL_BATCH_SIZE: int = Field(
        default=1,
        description="Batch size for evaluation"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()