import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loguru import logger


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"


@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider


QWEN = ModelConfig("qwen2.5", 0.0, ModelProvider.OLLAMA)
GEMMA_3 = ModelConfig("gemma3:12b", 0.0, ModelProvider.OLLAMA)
LLAMA_3_3 = ModelConfig("llama-3.3-70b-versatile", 0.0, ModelProvider.GROQ)


class Config:
    SEED = 42
    CHAT_MODEL = GEMMA_3
    TOOL_MODEL = QWEN

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME / "data"
        DATABASE_PATH = DATA_DIR / "mindora.sqlite"

    class Memory:
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        MAX_RECALL_COUNT = 5


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)


def configure_logging():
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": "<green>{time:YYYY-MM-DD - HH:mm:ss}</green> | <level>{level}</level> | {message}",
            },
        ]
    }
    logger.configure(**config)
