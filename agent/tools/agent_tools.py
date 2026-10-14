#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：practice_problems 
@File    ：agent_tools.py
@IDE     ：PyCharm 
@Author  ：JJ超人
@Date    ：2026/8/7 14:45 
'''
import json
import os
import random
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from rag.rag_service import RagSummarizerService
from utils.config_handler import agent_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizerService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010", ]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="查询指定城市的实时天气信息，支持中文（如：北京）或英文（如：Tokyo）城市名")
def get_weather(city: str) -> str:
    """调用 uapis.cn 天气接口，返回当前天气、温度、风向风力、湿度及预警信息。"""
    params = {"city": city, "lang": "zh"}
    url = f"{agent_config['weather_api_url']}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"[get_weather] 请求失败: {url}, error={e}")
        return f"查询天气失败：{e}"

    lines = [
        f"城市：{data.get('city', '') or data.get('province', '')}",
        f"天气：{data.get('weather', '')}",
        f"温度：{data.get('temperature', '')}°C",
        f"风向：{data.get('wind_direction', '')}",
        f"风力：{data.get('wind_power', '')}",
        f"湿度：{data.get('humidity', '')}%",
        f"更新时间：{data.get('report_time', '')}",
    ]

    if data.get("alerts"):
        alerts = "；".join(
            f"{a.get('title', '')}（{a.get('level', '')}）" for a in data["alerts"]
        )
        lines.append(f"气象预警：{alerts}")

    return "\n".join(lines)


@tool(description="获取用户的公网 IP 及所在省市信息，以纯字符串形式返回")
def get_user_location() -> str:
    """获取用户公网 IP 和所在省市县信息（按 IP 自动定位）。"""
    result_lines = []

    # 1) 获取公网 IP
    try:
        with urllib.request.urlopen(agent_config["ip_api_url"], timeout=10) as resp:
            ip_data = json.loads(resp.read().decode("utf-8"))
        result_lines.append(f"IP：{ip_data.get('ip', '')}")
    except Exception as e:
        logger.error(f"[get_user_location] 获取 IP 失败: {e}")

    # 2) 天气接口不传城市，按客户端 IP 自动定位省市县
    try:
        with urllib.request.urlopen(agent_config["weather_api_url"], timeout=10) as resp:
            loc = json.loads(resp.read().decode("utf-8"))
        location = "".join(
            [
                loc.get("province", ""),
                loc.get("city", ""),
                loc.get("district", ""),
            ]
        )
        result_lines.append(f"位置：{location}")
    except Exception as e:
        logger.error(f"[get_user_location] 获取位置失败: {e}")

    if not result_lines:
        return "获取用户位置失败"

    return "\n".join(result_lines)


@tool(description="获取用户的ID, 以存字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以存字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

# if __name__ == '__main__':
#     print(fetch_external_data('1001', '2025-01'))
