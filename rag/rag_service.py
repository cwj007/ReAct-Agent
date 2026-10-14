#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：practice_problems 
@File    ：rag_service.py
@IDE     ：PyCharm 
@Author  ：JJ超人
@Date    ：2026/8/7 14:11
总结服务类：用户提问：搜索参考资料，将提问和参考资料提交给模型，让模型总结回复

惰性单例设计：
- 构造实例是轻量操作，不连接 Milvus、不建链
- 首次真正调用检索/总结时才初始化，进程内只初始化一次（类级缓存 + 锁保证线程安全）
- 常驻服务场景下，initialize() 整个进程生命周期只执行一次
'''
import threading

from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizerService():
    _init_lock = threading.Lock()   # 常驻服务并发请求下保证只初始化一次
    _shared_vector_store = None     # 进程级共享，避免每个实例都重复连接 Milvus
    _shared_retriever = None        # 进程级共享检索器
    _shared_chain = None            # 进程级共享总结链

    def __init__(self):
        # 惰性：构造时不连接 Milvus、不建链，首次调用才触发初始化
        pass

    def _ensure_initialized(self):
        """首次调用时初始化：连接 Milvus、创建检索器、构建总结链（进程内仅一次）"""
        if RagSummarizerService._shared_vector_store is None:
            with RagSummarizerService._init_lock:
                # 双重检查，避免多个线程同时进入初始化
                if RagSummarizerService._shared_vector_store is None:
                    vector_store = VectorStoreService()
                    vector_store.initialize()
                    RagSummarizerService._shared_vector_store = vector_store
                    RagSummarizerService._shared_retriever = vector_store.get_retriever()
                    RagSummarizerService._shared_chain = self._build_chain()

        return (
            RagSummarizerService._shared_vector_store,
            RagSummarizerService._shared_retriever,
            RagSummarizerService._shared_chain,
        )

    @staticmethod
    def _build_chain():
        # 提示词要求纯文本输出，直接用 prompt -> model 即可，不需要结构化解析器
        prompt_text = load_rag_prompts()
        prompt_template = PromptTemplate.from_template(prompt_text)
        return prompt_template | chat_model

    def retrieve_docs(self, query: str):
        _, retriever, _ = self._ensure_initialized()
        return retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        _, retriever, chain = self._ensure_initialized()

        context_docs = retriever.invoke(query)

        if not context_docs:
            return "未检索到相关参考资料，无法回答。"

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        result = chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        # chain.invoke 返回 AIMessage 对象，只取 .content 纯文本，
        # 避免把 response_metadata / usage_metadata 等包装信息一并带出
        return result.content

if __name__ == '__main__':
    rag = RagSummarizerService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
