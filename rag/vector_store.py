#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
LangChain + Milvus 官方最佳实践版本
✅ 不混用原生 Client
✅ schema 完全由 LangChain 管理
✅ 初始化幂等
✅ 不卡死、日志清晰
"""

import os
import urllib.request
import time
from typing import Set

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import milvus_config
from utils.file_handler import (
    txt_loader,
    pdf_loader,
    listdir_with_allowed_type,
    get_file_md5_hex,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    """
    向量库服务（LangChain 官方推荐用法）
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=milvus_config["chunk_size"],
            chunk_overlap=milvus_config["chunk_overlap"],
            separators=milvus_config["separators"],
            length_function=len,
        )

        self.top_k = milvus_config.get("k", 3)
        self.vector_store: Milvus | None = None

        logger.info("✅ VectorStoreService 实例创建完成（未初始化 Milvus Collection）")

    # ============================================================
    # 初始化（⚠️ 必须显式调用）
    # ============================================================
    def initialize(self):
        # 幂等保护：已初始化则直接返回，避免重复连接 Milvus（进程内只初始化一次）
        if self.vector_store is not None:
            logger.debug("✅ Milvus VectorStore 已初始化，跳过重复初始化")
            return

        logger.info("🚀 开始初始化 Milvus VectorStore...")

        from pymilvus import (
            MilvusClient,
            connections,
            DataType,
            FieldSchema,
            CollectionSchema,
        )
        from pymilvus.milvus_client import IndexParams

        uri = f"http://{milvus_config['host']}:{milvus_config['port']}"
        db_name = milvus_config["db_name"]
        collection_name = milvus_config["collection_name"]
        dim = int(milvus_config["embedding_dim"])

        # ============================================================
        # 1. 幂等初始化：数据库不存在则创建，存在则跳过
        # ============================================================
        admin_client = MilvusClient(uri=uri)
        if db_name in admin_client.list_databases():
            logger.info(f"✅ 数据库已存在，跳过创建：{db_name}")
        else:
            admin_client.create_database(db_name)
            logger.info(f"✅ 数据库不存在，已创建：{db_name}")

        # ============================================================
        # 2. 幂等初始化：集合不存在则创建，存在则跳过
        #    schema 与下方 Milvus(...) 的 primary_field/text_field 保持一致
        # ============================================================
        probe = MilvusClient(uri=uri, db_name=db_name)
        if probe.has_collection(collection_name):
            logger.info(f"✅ 集合已存在，跳过创建：{collection_name}")
        else:
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=2048),
                    FieldSchema(name="chunk_id", dtype=DataType.INT32),
                ],
                description=f"LangChain RAG collection: {collection_name}",
            )
            index_params = IndexParams()
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                index_name="vector_idx",
                metric_type="L2",
                nlist=128,
            )
            probe.create_collection(
                collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )
            logger.info(f"✅ 集合不存在，已创建：{collection_name}")

        # ============================================================
        # 3. 兼容性规避：langchain-milvus 0.3.3 + pymilvus 2.6.x 官方 bug
        #    参考：https://github.com/langchain-ai/langchain-milvus/issues/130
        #
        #    原因：pymilvus 2.6 的 MilvusClient 改用新的 ConnectionManager，
        #         自生成别名 "cm-{id}" 不会注册进旧版 connections 注册表。
        #         而 langchain-milvus 内部仍通过 Collection(using=cm-{id})
        #         访问集合，因此抛 ConnectionNotExistException（表现为卡住）。
        #    方案：用同配置 probe client 拿到共享 handler，手动注册进旧版
        #         注册表，langchain-milvus 内部的 ORM 调用即可找到连接。
        # ============================================================
        connections._alias_handlers[probe._using] = probe._handler
        connections._alias_config[probe._using] = {
            "address": f"{milvus_config['host']}:{milvus_config['port']}",
            "db_name": db_name,
            "user": "",
        }

        self.vector_store = Milvus(
            embedding_function=embed_model,
            collection_name=milvus_config["collection_name"],
            connection_args={
                "uri": uri,
                "db_name": db_name,
            },
            # 适配已存在的旧集合 schema（id 主键 + content 文本字段）
            # langchain-milvus 默认是 pk/text，不匹配会导致写入失败
            primary_field="id",
            text_field="content",
            auto_id=True,
            drop_old=False,
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            },
            consistency_level="Strong",
        )

        logger.info(
            f"✅ Milvus VectorStore 初始化完成 | "
            f"collection={milvus_config['collection_name']} | "
            f"db={milvus_config['db_name']}"
        )

    # ============================================================
    # 对外接口
    # ============================================================
    def get_retriever(self):
        if self.vector_store is None:
            raise RuntimeError("请先调用 initialize()")

        return self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )

    def load_document(self):
        if self.vector_store is None:
            raise RuntimeError("请先调用 initialize()")

        md5_path = get_abs_path(milvus_config["md5_hex_store"])
        os.makedirs(os.path.dirname(md5_path), exist_ok=True)

        def _load_md5_set() -> Set[str]:
            if not os.path.exists(md5_path):
                return set()
            with open(md5_path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f}

        def _save_md5(md5_hex: str):
            with open(md5_path, "a", encoding="utf-8") as f:
                f.write(md5_hex + "\n")

        md5_set = _load_md5_set()
        data_path = get_abs_path(milvus_config["data_path"])

        files = listdir_with_allowed_type(
            data_path,
            tuple(milvus_config["allow_knowledge_file_type"]),
        )

        if not files:
            logger.warning("[加载知识库] 未找到任何可加载文件")
            return

        for path in files:
            md5_hex = get_file_md5_hex(path)
            if md5_hex in md5_set:
                logger.info(f"[加载知识库] 已存在，跳过：{path}")
                continue

            try:
                if path.endswith(".txt"):
                    docs = txt_loader(path)
                elif path.endswith(".pdf"):
                    docs = pdf_loader(path)
                else:
                    continue

                if not docs:
                    logger.warning(f"[加载知识库] 无有效文本：{path}")
                    continue

                split_docs = self.splitter.split_documents(docs)
                split_docs = [
                    d for d in split_docs
                    if d.page_content and d.page_content.strip()
                ]

                if not split_docs:
                    logger.warning(f"[加载知识库] 分片后为空：{path}")
                    continue

                # 旧集合存在必填字段 chunk_id，langchain-milvus 不会写入它，
                # 这里按分片顺序注入，避免插入时报 DataNotMatchException
                for idx, d in enumerate(split_docs):
                    d.metadata["chunk_id"] = idx

                logger.info(
                    f"[加载知识库] 开始写入向量：{path} ({len(split_docs)} chunks)"
                )
                self.vector_store.add_documents(split_docs)
                _save_md5(md5_hex)

                logger.info(f"[加载知识库] 加载成功：{path}")

            except Exception as e:
                logger.error(f"[加载知识库] 加载失败：{path}", exc_info=True)

    def drop_collection(self):
        if self.vector_store is None:
            raise RuntimeError("请先调用 initialize()")

        self.vector_store.drop()
        logger.warning(
            f"[Collection] {milvus_config['collection_name']} 已删除"
        )

    def close(self):
        logger.info("✅ VectorStoreService 关闭（Milvus 连接由 LangChain 管理）")


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.initialize()     # ✅ 唯一入口
    vs.load_document()  # ✅ 安全

    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")

    for r in res:
        print(r.page_content)
        print("-" * 20)