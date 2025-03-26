import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loguru import logger


class ModelProvider(str, Enum):
    GROQ = "groq"


@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider


DEEPSEEK_70B = ModelConfig("deepseek-r1-distill-llama-70b", 0.0, ModelProvider.GROQ)
LLAMA_3_3 = ModelConfig("llama-3.3-70b-versatile", 0.0, ModelProvider.GROQ)


class Config:
    SEED = 42
    CHAT_MODEL = LLAMA_3_3  # For general tasks
    TOOL_MODEL = DEEPSEEK_70B  # For complex tasks

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
