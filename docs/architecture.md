# 技术架构说明

## 一、整体架构

```text
                    +------------------+
                    | Streamlit 前端    |
                    | 6 Tab: Agent问答  |
                    | /Agent诊断/LangGraph|
                    | /报告/基础RAG/信息 |
                    +--------+---------+
                             |
                             v
+----------------------------+----------------------------+
|                  Agent Layer (src/agent/)                 |
|                                                          |
|  +--------------------------------------------------+   |
|  | 编排（两种实现，同一套工具）：                      |   |
|  |  a) agent.py  手写 ReAct Loop (最多 8 轮)          |   |
|  |  b) graph.py  LangGraph：router → ReAct →          |   |
|  |               grade → (不足) reflect → ReAct       |   |
|  | LLM 自主决策 → tool_use → 执行 → 质检 → 反思重试   |   |
|  +--------------------------------------------------+   |
|  | 工具: search_regulations | search_cases            |   |
|  |       lookup_asset      | lookup_asset_history     |   |
|  +--------------------------------------------------+   |
|                                                          |
+----------------------------+----------------------------+
                             |
                             v
+----------------------------+----------------------------+
|                  Application Layer (src/)                |
|                                                          |
|  detection         retrieval          generation         |
|  +--------+        +-----------+      +--------------+   |
|  | yolo_  |        | retriever |      | llm_client   |   |
|  | stub   |        +-----+-----+      | (Anthropic)  |   |
|  +---+----+              |            | report_      |   |
|      |                   |            |  generator   |   |
|      v                   v            +------+-------+   |
|  +--------+        +-----------+             v           |
|  |  VLM   |        | Chroma    |       +----------+      |
|  | 视觉    |        | 双 collec |       | Prompts  |      |
|  | 描述    |        +-----+-----+       +----------+      |
|  +--------+              ^                                |
|                          |                                |
|                  +-------+--------+                       |
|                  | ingestion      |                       |
|                  | text_loader    |                       |
|                  +-------+--------+                       |
+--------------------------+---------------------------------+
                           |
                           v
+----------------------------+----------------------------+
|                       Data Layer                         |
|                                                          |
|  data/regulations/*.md  data/defects_history/cases.jsonl |
|  data/assets/asset_registry.json                         |
|  data/assets/inspection_history.jsonl                    |
|                                                          |
|  chroma/                 (向量索引持久化)                 |
+----------------------------------------------------------+
```

## 二、关键设计决策

### 2.1 模型服务选型

- 当前使用 MiMo API（Anthropic 协议），mimo-v2.5-pro（文本）+ mimo-v2.5（多模态）
- Embedding 使用本地 sentence-transformers（BAAI/bge-small-zh-v1.5），无需额外 API
- 协议标准化，切换模型服务商只需改 config.yaml

### 2.2 为什么不引入 LangChain / LlamaIndex 的检索抽象

- 检索与生成直接用原生 Anthropic SDK + Chroma，代码透明、好讲解
- Agent 工具调用直接基于 Anthropic tool use 协议实现，无需框架封装其 RAG 管线
- 整体代码量可控，便于理解 Agentic RAG 的底层原理
- 注：编排层另行引入了 LangGraph（见 2.6），但仅用其图编排能力，模型调用仍走原生 SDK

### 2.6 为什么编排层引入 LangGraph（与手写循环并存）

`agent.py` 的手写 `for turn in range(MAX_TURNS)` 循环已经能跑，但控制流是隐式的——
"何时继续调工具、何时收尾"散落在循环体的 if 分支里，且加新阶段就得改循环骨架。
`graph.py` 用 LangGraph `StateGraph` 把控制流显式建模成图，并借此加入手写循环里没有的
**入口路由** 与 **质检-反思重试**：

```text
         ┌─(资产历史查询)─→ [direct_lookup] ─┐
start →[router]                              ├→ [agent] ─(含 tool_use)─→ [tools] ─┐
         └─(规程 / 通用)─────────────────────┘     │                              │
                                                   │ (无 tool_use / 到上限)        └→ 回到 [agent]
                                                   ▼
                                                [grade] ─(充分 / 反思达上限)─→ END
                                                   │
                                                   └─(不足)─→ [reflect] → 回到 [agent]
```

- **router（真条件分支，规则分流）**：复用 `src/router/intent_router` 识别意图。**含资产编号的历史/
  档案查询**所需工具是确定的，直接走 `direct_lookup` 快路径；其余走 ReAct `agent`。规则判断不调模型。
- **direct_lookup（确定性快路径）**：直接调 `lookup_asset` + `lookup_asset_history`（无 LLM 选工具的
  来回），结果注入 system 供 agent 综述——省掉 2 个工具选择轮次，更快更稳。
- **grade（LLM-as-Judge 质检）**：复用 `llm_client.chat`（`max_tokens=2048`，符合 MiMo 推理模型规则），
  按 `prompts.FAITHFULNESS_RUBRIC` 给忠实度打 1-5 分，低于 `FAITHFULNESS_PASS_THRESHOLD`（=4）判为不足。
  **该 rubric 与离线 `eval/ragas_eval` 的 Faithfulness 维度是同一把尺**——在线门控（二元代理）与离线
  指标（聚合真值）口径统一，改判据只改 `prompts.py` 一处。
- **reflect（反思重试）**：质检不足时把批评意见注入对话、回到 agent 重检索，最多 `MAX_REFLECTIONS=2`
  次；与 agent 的 `MAX_TURNS` 共同保证终止。到达 `MAX_TURNS` 时 agent 不再带工具定义，强制产出
  文本答案，确保进入 grade 的状态无悬空 tool_use（协议合法）。
- **状态显式**：`AgentState`（messages / steps / turn / reflections / grade_*）+ reducer
  （`operator.add`）把"追加消息、累计思考链"写进类型，而非藏在循环变量里。
- **控制流即数据**：两个决策点收敛为两个条件边（`_should_continue` / `_after_grade`），便于审查与单测；
  这也正是 LangGraph 相对手写循环的核心价值——分支与循环是图的一等公民。
- **演进友好**：呼应 PRD_v2——后续再加"检索打分过滤、混合召回切换"等阶段，只是往图里加节点与边。
- **不绑死框架**：节点内直接调原生 Anthropic 客户端，不用 `langchain-anthropic` /
  `create_react_agent`，规避 MiMo 推理模型 ThinkingBlock 与 `bind_tools` 的兼容问题。
- **代价**：grade + 最多 2 次反思会显著增加耗时与调用次数，故作为可选编排与手写 ReAct 并存、按需取用。

### 2.3 为什么用 RRF 而不是分数加权

- 不同召回路（向量 / 关键词 / 元数据）的分数量纲不一致
- RRF 只看排名不看绝对分，工业上更稳健
- 实现简单，参数 k=60 是 RRF 论文推荐的鲁棒值

### 2.4 为什么按 Markdown 标题切分而不是固定 chunk size

- 规程文档结构性强，标题层级即语义边界
- 切片同时携带 `section_path`，引用溯源能精确到章节
- 超长段落再做二次滑窗切分作为兜底

### 2.5 为什么 v1 只走 VLM 描述不接 YOLO

- 实习级别 PoC，做完整闭环优先于做硬技术
- VLM 已能给出足够的"视觉关键词"驱动文本召回
- v1.5 接 YOLO 只需替换 `src/detection/yolo_stub.py:detect()` 的实现，签名稳定

## 三、模块详解

### 3.1 ingestion

| 文件 | 职责 |
|---|---|
| `text_loader.py` | Markdown 解析 + 切片；案例 / 资产 / 巡检历史的统一加载 |

### 3.2 retrieval

| 文件 | 职责 |
|---|---|
| `vector_store.py` | Chroma 客户端 + collection 管理 + 元数据清洗 |
| `retriever.py` | 多路召回 API + RRF 融合 |

### 3.3 generation

| 文件 | 职责 |
|---|---|
| `llm_client.py` | Anthropic SDK 客户端 + 重试 + 文本/图像调用；本地 sentence-transformers embedding |
| `prompts.py` | 集中管理所有 system / user prompt 模板 |
| `report_generator.py` | 端到端管线：诊断 / 报告 / 问答三个入口 |

### 3.4 detection

| 文件 | 职责 |
|---|---|
| `yolo_stub.py` | v1 用 VLM 充当检测器返回结构化描述；v1.5 替换为真实 YOLO |

### 3.5 router

| 文件 | 职责 |
|---|---|
| `intent_router.py` | 轻量规则分类（v2 替换为 LLM 分类器）|

## 四、检索质量调优手段

| 手段 | 当前是否启用 | v2 计划 |
|---|---|---|
| 多路召回（规程 + 案例） | 启用 | 增加图像路 |
| 元数据过滤（asset_type / defect_type） | 启用 | 增加 severity 过滤 |
| RRF 融合 | 启用 | 加入业务权重 |
| Rerank（重排模型） | 暂未启用 | 引入 BGE-Reranker-v2-m3 |
| 关键词召回（BM25） | 暂未启用 | 与向量召回组成 hybrid |

## 五、可观测性

DEBUG=1 时：
- `report_generator.diagnose_image` 在 `result.debug` 中带 query 文本
- 便于在 Streamlit 系统信息 Tab 中检查"为什么召回了这一组"

更完整的日志体系（结构化日志 + token 统计 + 链路追踪）排在 v2.5。

## 六、性能与成本

| 操作 | 平均耗时 | 单次 API 成本 |
|---|---|---|
| 单图诊断（VLM + 召回 + LLM） | 25-30s | 约 0.05 元 |
| 报告生成（10 张图聚合） | 30-50s | 约 0.20 元 |
| 知识问答 | 5-8s | 约 0.005 元 |
| 索引构建（首次） | 60-120s | 约 0.05 元 |

## 七、扩展性预留

- 替换向量库：将 `vector_store.py` 内的 chroma 调用替换为 Milvus / Weaviate 即可
- 替换 LLM：在 `config.yaml` 改 `provider.base_url` 与 `model` 名
- 增加召回路：在 `retriever.py` 新增 `retrieve_xxx`，并在调用方 `fuse_rrf` 多入参
- 接入 YOLO：替换 `detection/yolo_stub.py:detect` 实现，输入输出契约保持不变
