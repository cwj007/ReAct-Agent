#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：practice_problems 
@File    ：prompt_loader.py
@IDE     ：PyCharm 
@Author  ：JJ超人
@Date    ：2026/8/4 16:30 
'''
from functools import wraps

from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# 装饰器
def prompt_loader(config_key: str):
    """
    装饰器：根据 prompts_config 中的 key 加载 prompt 文件
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 从配置中取路径
            try:
                relative_path = prompts_config[config_key]
            except KeyError as e:
                logger.error(
                    f"[prompt_loader] prompts.yaml 缺少配置项: {config_key}"
                )
                raise KeyError(f"prompts.yaml 缺少配置项: {config_key}") from e

            # 2. 转为绝对路径
            abs_path = get_abs_path(relative_path)

            # 3. 读取 prompt 文件
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            except Exception as e:
                logger.error(
                    f"[prompt_loader] 读取 prompt 文件失败: {abs_path}, error={e}"
                )
                raise RuntimeError(f"读取 prompt 失败: {abs_path}") from e

            logger.debug(
                f"[prompt_loader] 成功加载 prompt({config_key}), length={len(prompt_text)}"
            )

            # 4. 交给原函数（方便以后做后处理）
            return func(prompt_text, *args, **kwargs)

        return wrapper
    return decorator

@prompt_loader("main_prompt_path")
def load_system_prompts(prompt: str) -> str:
    return prompt


@prompt_loader("rag_summarize_prompt_path")
def load_rag_prompts(prompt: str) -> str:
    return prompt


@prompt_loader("report_prompt_path")
def load_report_prompts(prompt: str) -> str:
    return prompt