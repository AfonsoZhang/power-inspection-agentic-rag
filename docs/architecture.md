# 技术架构说明

## 一、整体架构

```text
                    +------------------+
                    | Streamlit 前端    |
                    | 5 Tab: Agent问答  |
                    | /Agent诊断/报告  |
                    | /基础RAG/系统信息 |
                    +--------+---------+
                             |
                             v
+----------------------------+----------------------------+
|                  Agent Layer (src/agent/)                 |
|                                                          |
|  +--------------------------------------------------+   |
|  | ReAct Agent Loop (最多 8 轮)                       |   |
|  | LLM 自主决策 → tool_use → 执行 → 结果反馈          |   |
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

### 2.2 为什么不引入 LangChain / LlamaIndex

- 直接用原生 Anthropic SDK + Chroma，代码透明、好讲解
- Agent 工具调用直接基于 Anthropic tool use 协议实现，无需框架封装
- 整体代码量可控，便于理解 Agentic RAG 的底层原理

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
