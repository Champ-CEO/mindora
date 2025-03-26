from functools import lru_cache

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from mindora.config import Config, ModelConfig, ModelProvider


@lru_cache(maxsize=1)
def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Memory.EMBEDDING_MODEL)


def create_llm(model_config: ModelConfig) -> BaseChatModel:
    if model_config.provider == ModelProvider.GROQ:
        return ChatGroq(model=model_config.name, temperature=model_config.temperature)
