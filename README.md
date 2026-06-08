# 无人机巡检 Agentic RAG 系统

> **Agentic RAG** 架构：LLM 自主决策调用哪些检索工具，多轮推理后给出带引用的结构化回答。
>
> 面向电力巡检场景，覆盖"智能问答 → 缺陷诊断 → 报告生成"全链路，支持多模态图像输入。

## 核心亮点

| 特性 | 说明 |
|---|---|
| **Agentic RAG** | LLM 通过 tool use 自主编排 4 个检索工具（规程/案例/资产/历史），而非固定 retrieve→generate |
| **多模态诊断** | 上传绝缘子等巡检图像，VLM 看图 + Agent 检索规程案例 → 自动诊断 |
| **ReAct 推理链** | 最多 8 轮 Agent 循环，前端完整展示思考→工具调用→结果→回答过程 |
| **双编排实现** | 同一套工具，提供手写 ReAct 循环与 LangGraph StateGraph（router 路由 + grade 质检 + reflect 反思重试）两种编排，可对比 |
| **本地 Embedding** | sentence-transformers（BAAI/bge-small-zh-v1.5），无需额外 API |
| **基础 RAG 对比** | 内置传统 RAG Tab，可直观对比 Agentic RAG 的优势 |

## 技术架构

```text
用户问题 / 巡检图像
        |
        v
  ┌─────────────────────────────┐
  │   Agentic RAG (ReAct Loop)  │
  │   LLM 自主决策工具调用       │
  └──────────┬──────────────────┘
             |  tool_use
    ┌────────┼────────┬──────────────┐
    v        v        v              v
 search_  search_  lookup_     lookup_asset_
 regulations cases  asset       history
    |        |        |              |
    v        v        v              v
 ChromaDB  ChromaDB  JSON         JSONL
 (49 chunks) (12 cases) (资产档案)  (巡检历史)
    └────────┴────────┴──────────────┘
             |  tool_result
             v
  ┌─────────────────────────────┐
  │  LLM 综合推理 → 结构化回答   │
  │  (带引用来源)                │
  └─────────────────────────────┘
```

### 模型配置

| 用途 | 模型 | 协议 |
|---|---|---|
| 文本推理 | mimo-v2.5-pro | Anthropic |
| 多模态（图像） | mimo-v2.5 | Anthropic |
| 文本嵌入 | BAAI/bge-small-zh-v1.5 | 本地 sentence-transformers |
| 向量存储 | ChromaDB | 本地持久化 |

### Agent 工具

| 工具 | 功能 |
|---|---|
| `search_regulations` | 语义检索行业规程条款（DL/T 741 等） |
| `search_cases` | 语义检索历史缺陷案例，支持缺陷类型过滤 |
| `lookup_asset` | 查询资产档案（线路、型号、投运年份等） |
| `lookup_asset_history` | 查询指定资产的巡检历史记录 |

## 目录结构

```text
drone-inspection-rag/
├── app/
│   └── streamlit_app.py         # Streamlit Demo（6 个 Tab）
├── src/
│   ├── agent/
│   │   ├── agent.py             # ReAct Agent 核心循环（手写）
│   │   ├── graph.py             # LangGraph StateGraph：router→ReAct→grade→reflect 纠错式编排
│   │   └── tools.py             # 4 个工具定义 + 执行分发
│   ├── config.py                # 配置加载
│   ├── ingestion/               # 文档解析 + 切片
│   ├── retrieval/               # 多路召回 + RRF 融合
│   ├── generation/              # LLM 客户端 + Prompt + 报告生成
│   ├── detection/               # YOLO 缺陷检测 stub
│   └── router/                  # 意图路由
├── data/
│   ├── regulations/             # 行业规程（3 份 Markdown）
│   ├── defects_history/         # 历史缺陷案例（12 条 JSONL）
│   └── assets/                  # 资产档案 + 巡检历史
├── demo/
│   └── sample_images/           # 绝缘子样本图像（5 张）
├── eval/                        # 评测脚本
├── docs/                        # PRD / 架构文档
├── config.yaml                  # 模型 / 检索 / 路径配置
├── requirements.txt
└── .env.example                 # API Key 模板
```

## 快速开始

### 1. 环境准备

```bash
cd drone-inspection-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API 密钥

复制 `.env.example` 为 `.env`，填入 MiMo API Key：

```bash
cp .env.example .env
```

```env
MIMO_API_KEY=your-api-key-here
```

### 3. 构建向量索引

```bash
python scripts/build_index.py
```

首次运行会加载本地 embedding 模型并构建 ChromaDB 索引（49 个规程 chunk + 12 条案例）。

### 4. 启动 Demo

```bash
streamlit run app/streamlit_app.py --server.headless true
```

打开 http://localhost:8501 ，包含 6 个功能 Tab：

| Tab | 功能 |
|---|---|
| 智能问答（Agent） | Agentic RAG 问答（手写 ReAct 循环），可上传图像，展示完整推理链 |
| 缺陷诊断（Agent） | 上传巡检图像，Agent 自动看图 + 检索 + 诊断 |
| LangGraph Agent | LangGraph 纠错式编排（router 路由 + grade 质检 + reflect 反思重试），支持上传图像，附图结构可视化 |
| 巡检报告 | 基于诊断结果生成结构化报告草稿 |
| 基础 RAG 对比 | 传统 RAG 流程，对比 Agent 模式效果 |
| 系统信息 | 配置状态 / 数据规模 / 工具列表 |

### 5. 运行评测（可选）

```bash
python eval/business_kpi.py      # 业务 KPI：耗时/引用率/等级准确率
python eval/ragas_eval.py        # RAGAS 评测（需额外安装 ragas 等依赖）
```

## Agentic RAG vs 基础 RAG

| 维度 | 基础 RAG | Agentic RAG（本项目） |
|---|---|---|
| 检索策略 | 固定管线，每次都检索 | LLM 自主判断是否需要检索、检索什么 |
| 工具编排 | 无 | 4 个工具，LLM 自主组合调用 |
| 多轮推理 | 单轮 | 最多 8 轮，可换关键词再搜 |
| 多模态 | 不支持 | VLM 看图 + Agent 检索协同 |
| 可解释性 | 仅返回答案 | 展示完整思考链和工具调用过程 |

## 数据说明

所有数据均为合成 / 公开来源，不涉及任何真实企业数据：

- **行业规程**：参考 DL/T 741、Q/GDW 1799 等公开标准的写作风格自行撰写
- **缺陷案例**：基于公开数据集标注思路自行合成
- **资产档案 / 巡检历史**：虚拟杆塔与巡检记录
- **样本图像**：来自 FINet SFID 公开绝缘子数据集
