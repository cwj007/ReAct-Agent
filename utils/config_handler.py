#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：practice_problems 
@File    ：config_handler.py
@IDE     ：PyCharm 
@Author  ：JJ超人
@Date    ：2026/8/4 16:03 
'''
import yaml
from utils.path_tool import get_abs_path

def load_rag_config(config_path: str=get_abs_path("config/rag.yml")):
    with open(config_path, "r", encoding='utf-8') as f:
        # 全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

def load_milvusconfig(config_path: str=get_abs_path("config/milvus.yml")):
    with open(config_path, "r", encoding='utf-8') as f:
        # 全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml")):
    with open(config_path, "r", encoding='utf-8') as f:
        # 全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(config_path: str=get_abs_path("config/agent.yml")):
    with open(config_path, "r", encoding='utf-8') as f:
        # 全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config = load_rag_config()
milvus_config = load_milvusconfig()
prompts_config = load_prompts_config()
agent_config = load_agent_config()


if __name__ == '__main__':
    print(rag_config['chat_model_name'])
    print(milvus_config.get('embedding_dim'))