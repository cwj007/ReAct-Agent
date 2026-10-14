#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：practice_problems 
@File    ：factory.py
@IDE     ：PyCharm 
@Author  ：JJ超人
@Date    ：2026/8/5 16:51 
'''
from abc import ABC, abstractmethod
from typing import Optional

from langchain.embeddings import init_embeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

from utils.config_handler import rag_config


class BaseModelFactory(ABC):
    @abstractmethod
    def create(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    """Chat 模型工厂"""

    def create(self) -> BaseChatModel:
        model_name = rag_config["chat_model_name"]

        if model_name.startswith("deepseek"):
            return ChatDeepSeek(
                model=model_name,
                api_key=rag_config.get("chat_api_key"),
                base_url=rag_config.get("chat_base_url"),
            )

        if model_name.startswith("qwen"):
            return ChatTongyi(
                model=model_name,
                dashscope_api_key=rag_config.get("chat_api_key"),
            )

        raise ValueError(f"Unsupported chat model: {model_name}")


class EmbeddingModelFactory(BaseModelFactory):
    """Embedding 模型工厂"""

    def create(self) -> Embeddings:
        provider = rag_config["embedding_model_provider"]

        if provider == "dashscope":
            return DashScopeEmbeddings(
                model=rag_config["embedding_model_name"],
                dashscope_api_key=rag_config["embedding_api_key"],
            )

        if provider == "openai":
            return OpenAIEmbeddings(
                model=rag_config["embedding_model_name"],
                api_key=rag_config.get("embedding_api_key"),
                base_url=rag_config.get("embedding_base_url"),
            )

        # 以后可以加 OpenAI / HuggingFace / Ollama
        raise ValueError(f"Unsupported embedding provider: {provider}")


chat_model = ChatModelFactory().create()

embed_model = EmbeddingModelFactory().create()

if __name__ == '__main__':
    print(embed_model)