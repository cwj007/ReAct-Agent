# 智扫通 · 扫地机器人智能客服 Agent

> 基于 **LangChain ReAct Agent + RAG 检索增强生成** 打造的家用清洁机器人智能客服系统。
> 一个具备 **自主思考、动态工具调用、按场景切换提示词、多轮检索增强** 能力的可落地 Agent 工程实践。

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-1.x-1C3C3C">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Agent%20Runtime-FF6F61">
  <img alt="Milvus" src="https://img.shields.io/badge/Milvus-Vector%20DB-00A4EF">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white">
</p>

---

## 一、项目简介

**智扫通** 是面向「扫地机器人 / 扫拖一体机器人」场景的智能客服 Agent。它不是简单的「提问 → 大模型 → 回答」链路，而是围绕 **Agent 自主决策** 构建的完整系统：

- 由 LLM 自主判断 **是否需要调用工具、调用哪个工具、以什么参数调用**；
- 通过 **ReAct（Reasoning + Acting）循环**，完成「思考 → 行动 → 观察 → 再思考」的多轮工具编排；
- 结合 **RAG 检索增强**，让回答严格基于企业私有知识库（选购指南、故障排除、维护保养、100 问等），降低幻觉；
- 面向实际业务场景，支持 **个性化使用报告生成**，并在报告场景下 **动态切换系统提示词**。

典型可回答的问题：

| 用户提问 | Agent 行为 |
| --- | --- |
| 小户型适合哪些扫地机器人？ | 调用 `rag_summarize` 检索选购指南 → 总结回答 |
| 我在北京，这个天气下机器人怎么保养？ | `get_user_location` → `get_weather` → `rag_summarize` → 综合建议 |
| 给我生成我的 6 月使用报告 | `get_user_id` → `get_current_month` → `fill_context_for_report` → `fetch_external_data` → 切换报告提示词 → 输出 Markdown 报告 |

---

## 二、核心亮点（Agent 开发向）

### 1. ReAct 自主决策 Agent
基于 LangChain 新版 `create_agent` 构建，通过系统提示词强约束 Agent 遵循 ReAct 范式，实现 **工具的自主动态编排**：
- 先判断「现有信息是否足够回答」，不足则自主决策调用工具；
- 工具返回后 **再次评估信息完整性**，不足则发起二次/多次工具调用（最多 5 次）；
- 5 次调用后仍无法回答则诚实回复「我不知道」，避免编造。

### 2. 中间件（Middleware）驱动的 Agent 运行时增强
项目充分利用 LangChain Middleware 机制，对 Agent 执行过程做 **无侵入式切面增强**，这是本项目最具工程价值的亮点：

| 中间件 | 装饰器 | 作用 |
| --- | --- | --- |
| `monitor_tool` | `@wrap_tool_call` | 工具调用全链路监控：记录工具名、入参、成功/失败；**并作为场景标记的注入点** |
| `log_before_model` | `@before_model` | 模型调用前记录消息数量与最新消息，提供可观测性 |
| `report_prompt_switch` | `@dynamic_prompt` | 根据运行时上下文 **动态切换系统提示词** |

### 3. 运行时上下文注入 + 提示词动态路由（核心设计）
这是本项目的关键设计：**无需硬编码判断，由 Agent 自身的行为触发上下文变更，进而驱动提示词路由。**

```
用户："生成我的使用报告"
        │
        ▼
 Agent 调用工具 fill_context_for_report()
        │
        ▼
 中间件 monitor_tool 捕获到该工具调用
        │
        ▼
 request.runtime.context['report'] = True   ← 注入运行时上下文标记
        │
        ▼
 下一轮模型调用前，@dynamic_prompt 读取上下文
        │
        ├─ report=True  → load_report_prompts()   报告写手提示词
        └─ report=False → load_system_prompts()   客服提示词
```

- 报告场景提示词只保留「用户 ID / 月份 / 外部数据 / RAG」四类工具，**缩小工具空间、提升报告生成稳定性**；
- 普通客服场景则使用包含天气、定位等全部工具的提示词；
- 通过 `context={"report": False}` 在 `stream` 时初始化上下文，全程由 Agent 行为驱动切换。

### 4. RAG 作为 Agent 的一项能力（Tool）
将 RAG 检索问答封装为 Agent 的标准工具 `rag_summarize`，使「检索增强」不再是固定链路，而是 **由 Agent 按需自主触发** 的能力。

### 5. 面向业务的多工具协同
Agent 可协同 7 个工具完成复杂任务编排：

| 工具 | 能力 | 说明 |
| --- | --- | --- |
| `rag_summarize` | 知识库检索问答 | 向量库语义检索 + LLM 总结 |
| `get_weather` | 实时天气查询 | 对接 uapis.cn，支持中文/英文城市名 |
| `get_user_location` | 用户定位 | 公网 IP 定位 + 按 IP 自动定位省市县 |
| `get_user_id` | 获取用户 ID | 为个性化报告提供身份标识 |
| `get_current_month` | 获取当前月份 | 标准 `YYYY-MM` 格式 |
| `fetch_external_data` | 查询使用记录 | 从外部 CSV 系统读取指定用户/月份记录 |
| `fill_context_for_report` | 触发上下文注入 | 仅用于报告场景的「信号工具」 |

---

## 三、技术栈

| 层次 | 技术选型 |
| --- | --- |
| 语言环境 | Python 3.11 |
| Agent 框架 | LangChain 1.x（`create_agent`）、LangGraph（Agent Runtime / Middleware） |
| 大语言模型 | DeepSeek（`langchain-deepseek`）/ 通义千问（`ChatTongyi`），工厂模式可切换 |
| Embedding | DashScope / OpenAI `text-embedding-3-large`（3072 维） |
| 向量数据库 | Milvus（`langchain-milvus`，IVF_FLAT 索引，L2 距离） |
| 文本处理 | `RecursiveCharacterTextSplitter` 递归分块 |
| 前端交互 | Streamlit（流式逐字输出 + 会话状态管理） |
| 工程化 | PyYAML 配置驱动、工厂模式、单例、结构化日志 |

---

## 四、系统架构

```mermaid
flowchart TD
    U[用户提问] --> UI[Streamlit 前端]
    UI --> RA[ReactAgent.create_agent]

    RA --> MW1["@before_model<br/>log_before_model"]
    MW1 --> MW2["@dynamic_prompt<br/>report_prompt_switch"]
    MW2 --> LLM[Chat Model<br/>DeepSeek / Qwen]

    LLM -->|决定调用工具| MW3["@wrap_tool_call<br/>monitor_tool"]
    MW3 --> T{工具路由}

    T --> T1[rag_summarize]
    T --> T2[get_weather]
    T --> T3[get_user_location]
    T --> T4[get_user_id]
    T --> T5[get_current_month]
    T --> T6[fetch_external_data]
    T --> T7[fill_context_for_report]

    T1 --> RAG[RagSummarizerService]
    RAG --> VS[(Milvus 向量库)]

    T7 -.注入上下文 report=True.-> MW3
    MW3 -.runtime.context.-> MW2

    LLM -->|信息足够, 生成最终回答| UI
    UI -->|流式输出| U
```

**请求执行流程（报告生成场景）：**

1. Streamlit 接收用户输入，调用 `ReactAgent.execute_stream(query)`，传入 `context={"report": False}`；
2. `@dynamic_prompt` 读取 `report=False`，加载客服提示词；
3. 模型思考后决定调用 `get_user_id`、`get_current_month`；
4. `@wrap_tool_call` 监控工具执行并记录日志；
5. 模型调用 `fill_context_for_report` 信号工具 → 中间件将 `runtime.context['report']` 置为 `True`；
6. 下一轮模型调用前，`@dynamic_prompt` 感知到 `report=True`，加载「报告写手」提示词；
7. 模型调用 `fetch_external_data` 获取使用记录，并以 Markdown 生成个性化报告；
8. 前端以字符流形式逐字输出结果。

---

## 五、目录结构

```text
LangChain-ReAct_Agent/
├── app.py                      # Streamlit 前端入口（流式输出 + 会话状态）
├── agent/
│   ├── react_agent.py          # ReactAgent：create_agent 装配模型/工具/中间件
│   └── tools/
│       ├── agent_tools.py      # 7 个 Agent 工具定义
│       └── middleware.py       # 3 个中间件（监控 / 日志 / 动态提示词）
├── rag/
│   ├── rag_service.py          # RAG 总结服务（惰性单例 + 线程安全）
│   └── vector_store.py         # Milvus 向量库服务（幂等初始化 + 增量入库）
├── model/
│   └── factory.py              # 模型工厂（Chat / Embedding 可插拔）
├── utils/
│   ├── config_handler.py       # YAML 配置加载
│   ├── file_handler.py         # 文件读取（TXT/PDF）、MD5、目录扫描
│   ├── logger_handler.py       # 结构化日志（控制台 + 文件双通道）
│   ├── path_tool.py            # 统一绝对路径工具
│   └── prompt_loader.py        # 提示词加载装饰器
├── config/
│   ├── agent.yml               # Agent 工具相关配置（API 地址、数据路径）
│   ├── rag.yml                 # 模型与 Embedding 配置
│   ├── milvus.yml              # 向量库配置（连接、分块、检索参数）
│   └── prompts.yml             # 提示词文件路径映射
├── prompts/
│   ├── main_prompt.txt         # 客服场景系统提示词（ReAct 约束）
│   ├── report_prompt.txt       # 报告生成场景系统提示词
│   └── rag_summarize.txt       # RAG 总结提示词模板
├── data/                       # 知识库（100 问、选购、故障排除、维护保养等）
│   └── external/records.csv    # 外部用户使用记录数据
└── logs/                       # 运行日志
```

---

## 六、核心模块详解

### 1. Agent 装配 —— [agent/react_agent.py](agent/react_agent.py)
使用 LangChain 新版 `create_agent`，以声明式方式完成「模型 + 系统提示词 + 工具集 + 中间件」装配，并通过 `stream(stream_mode="values")` 实现流式输出。

### 2. 中间件 —— [agent/tools/middleware.py](agent/tools/middleware.py)
- `monitor_tool`：工具执行的前置/后置切面，统一日志 + 场景标记注入；
- `log_before_model`：模型调用前埋点；
- `report_prompt_switch`：基于 `runtime.context` 的提示词动态路由。

### 3. 工具集 —— [agent/tools/agent_tools.py](agent/tools/agent_tools.py)
统一使用 `@tool` 装饰器声明工具，工具的 `description` 直接参与 LLM 的工具选择决策，因此每个工具的说明都精确描述了 **能力边界、入参格式、使用场景**。

### 4. RAG 服务 —— [rag/rag_service.py](rag/rag_service.py)
- **惰性单例**：构造时零成本，首次真正检索时才连接 Milvus、构建总结链；
- **线程安全**：类级缓存 + 锁 + 双重检查，保证常驻服务并发场景下 `initialize()` 进程内仅执行一次；
- **进程级共享**：向量库、检索器、总结链全局共享，避免重复连接。

### 5. 向量库服务 —— [rag/vector_store.py](rag/vector_store.py)
- **幂等初始化**：数据库 / 集合不存在则创建，存在则跳过；
- **增量入库**：以文件 MD5 去重，已入库文件自动跳过，避免重复 Embedding 与写入；
- **健壮分块**：递归分块后过滤空分片。

---

## 七、快速开始

### 1. 环境准备

```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install langchain langchain-community langchain-deepseek langchain-openai \
            langchain-milvus langchain-text-splitters langgraph streamlit pymilvus pyyaml
```

### 2. 启动 Milvus

准备好 Milvus 服务（默认端口 `19530`），并在 `config/milvus.yml` 中配置 `host` / `port`。

### 3. 配置说明

| 配置文件 | 关键项 |
| --- | --- |
| `config/rag.yml` | `chat_model_name`、`embedding_model_name`、`embedding_model_provider` 及各 API Key |
| `config/milvus.yml` | 连接信息、`collection_name`、`chunk_size`、`k`、`embedding_dim` |
| `config/agent.yml` | 天气 / IP 接口地址、外部数据路径 |
| `config/prompts.yml` | 三个提示词文件的路径映射 |

> ⚠️ 切换模型只需修改 `rag.yml` 中的 `chat_model_name`（`deepseek-*` / `qwen-*`）或 `embedding_model_provider`（`dashscope` / `openai`），无需改动业务代码。

### 4. 知识库入库

```bash
python rag/vector_store.py
```

首次运行会创建数据库与集合、加载 `data/` 下的 TXT/PDF 知识文件、分块并写入 Milvus。

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器打开后即可与智能客服对话。

### 6. 命令行调试 Agent

```bash
python agent/react_agent.py
```

---

## 八、关键实现难点

| 难点 | 解决方案 |
| --- | --- |
| **Agent 场景切换** | 用「信号工具 + 中间件上下文注入 + 动态提示词」实现无硬编码的场景路由 |
| **RAG 服务重复初始化** | 惰性单例 + 双重检查锁，保证常驻服务下进程内只初始化一次 |
| **知识库重复入库** | 基于文件 MD5 的增量去重，节省 Embedding 成本 |
| **Milvus 连接兼容性** | 规避 `langchain-milvus 0.3.3` 与 `pymilvus 2.6.x` 的 `ConnectionNotExistException` 官方 Bug（见 [issue #130](https://github.com/langchain-ai/langchain-milvus/issues/130)） |
| **自定义 Schema 适配** | `langchain-milvus` 默认主键/文本字段为 `pk`/`text`，与已有集合不匹配会导致写入失败，需显式指定 `primary_field`/`text_field` 并补齐必填字段 |
| **Agent 幻觉控制** | 提示词强约束「仅基于参考资料回答、不编造、5 次工具调用后仍不足则说不知道」 |

---

## 九、安全说明

本项目 `config/*.yml` 中的 API Key 为演示用明文配置。**若将本项目上传至公开仓库，请务必：**

1. 将 `config/rag.yml` 加入 `.gitignore`，改用环境变量或本地私密配置文件注入密钥；
2. 检查并移除历史提交中已泄露的 Key；
3. 对已暴露的 Key 立即在对应平台 **撤销并重新生成**。

---

## 十、可扩展方向

- **多轮记忆**：引入 LangGraph Checkpointer 实现跨会话长期记忆；
- **工具扩展**：接入订单查询、售后工单、设备 OTA 状态等企业系统；
- **可观测性**：接入 LangSmith / Langfuse 做 Agent 全链路追踪与评估；
- **评测体系**：构建 Agent 工具选择准确率、RAG 召回率、回答忠实度等自动化评测；
- **部署形态**：以 FastAPI 暴露 Agent 服务，前端与 Agent 解耦。
