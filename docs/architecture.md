# 技术架构说明

## 一、整体架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Streamlit 前端（8 Tab）                                              │
│  Agent问答 / Agent诊断 / LangGraph / 低空作业台 / 多智能体             │
│  / 巡检报告 / 基础RAG对比 / 系统信息                                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│  编排层 src/agent/                                                    │
│                                                                       │
│   agent.py    手写 ReAct 循环（基线：agent ⇄ tools）                   │
│   graph.py    LangGraph：router → [direct_lookup] → agent ⇄ tools     │
│                          → grade →(不足) reflect → agent              │
│   crew.py     多智能体：planner → 合规闸门 → diagnostician → dispatcher│
│                                                                       │
│   tools.py    6 个工具，分两类 ─────────────────────────────────────┐ │
│               检索类（语义，结果需模型综合）                          │ │
│                 search_regulations / search_cases                    │ │
│                 lookup_asset / lookup_asset_history                  │ │
│               确定性作业类（可验证计算，零幻觉、可单测）               │ │
│                 plan_inspection_mission / check_flight_clearance     │ │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
┌───────────────┐   ┌────────────────────┐   ┌────────────────────────┐
│ retrieval/    │   │ generation/        │   │ mission/  ← 低空作业    │
│  vector_store │   │  llm_types  中立   │   │  geo      距离/2-opt    │
│  retriever    │   │  providers  适配   │   │  planner  时效/架次     │
│  (RRF 融合)   │   │  llm_client 门面   │   │  airspace 空域/气象     │
└──────┬────────┘   │  prompts           │   └───────────┬────────────┘
       │            │  report_generator  │               │
       │            └─────────┬──────────┘               │
       ▼                      ▼                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  数据层 data/                                                         │
│   regulations/*.md          49 chunks（Markdown 标题切分）            │
│   defects_history/cases     12 条缺陷案例                             │
│   assets/asset_registry     52 基杆塔（含经纬度）                      │
│   assets/inspection_history 75 条巡检记录                             │
│   airspace/constraints      5 个空域限制区 + 气象限值 + 安全距离       │
│   .chroma/                  向量索引持久化                            │
└──────────────────────────────────────────────────────────────────────┘
```

## 二、关键设计决策

### 2.1 provider 中立：为什么不直接用某一家的 SDK 对象

v1 的 Agent 循环直接操作 Anthropic 的 content block（`block.type == "tool_use"`），
换模型服务商就要重写循环。v2 在 `generation/` 下拆了三层：

| 层 | 文件 | 职责 |
|---|---|---|
| 中立数据结构 | `llm_types.py` | `ToolCall` / `LLMResponse` + 中立消息格式；图片 base64 编码只此一处 |
| 协议适配 | `providers.py` | 中立 ⇄ Anthropic / OpenAI 的双向翻译，**纯函数、可单测、不发网络请求** |
| 门面 | `llm_client.py` | 文本 / 多模态选路、重试、本地 embedding |

结果：`agent.py` / `graph.py` / `crew.py` 里没有任何一家 SDK 的类型；换 provider 只改
`config.yaml`。适配层的 17 个单测覆盖了工具调用往返、图像分片、连续 tool_result 合并
（Anthropic 要求合并进一条 user 消息，OpenAI 要求拆成多条 tool 消息）等易错点。

默认 provider 是 DeepSeek（OpenAI 兼容）。任何 OpenAI 兼容端点（通义、vLLM、Ollama）
都用 `provider: openai` + 自定义 `base_url`。多模态单独配 `vlm:` 段，未配置时图像入口
自动禁用，纯文本链路不受影响。

### 2.2 工具分两类：检索 vs 确定性计算

新增的两个低空作业工具（`plan_inspection_mission` / `check_flight_clearance`）刻意做成
**不含任何模型调用**的纯计算：

- 复飞时效 = 缺陷发现日 + 该等级的规程时限
- 架次切分 = 电池有效作业时长 ÷（悬停时长 + 段间转场时间）
- 空域裁决 = 塔位经纬度与限制区的距离比对，取最严的一条

这类问题有唯一正确答案，交给模型只会引入不可控误差，而且没法回归。放进工具后：
用例可以断言"每个架次不超电池续航""高优先级不会被航线优化排到低优先级之后"，
CI 里不用 API Key 就能守住。模型只决定**何时调用**和**如何向人解释结果**。

### 2.3 LangGraph 编排：router / grade / reflect

```text
         ┌─(工具已确定的场景)→ [direct_lookup] ─┐
start →[router]                                 ├→ [agent] ─(含 tool_call)→ [tools] ─┐
         └─(需自主编排)──────────────────────────┘     │                             │
                                                       │ (无 tool_call / 到轮次上限)  └→ 回 [agent]
                                                       ▼
                                                    [grade] ─(通过 / 反思用尽)→ END
                                                       │
                                                       └─(不足)→ [reflect] → 回 [agent]
```

- **router（真条件分支，零模型调用）**：规则识别意图。三类场景所需工具是确定的——
  含资产编号的历史/档案查询、含资产编号的合规校验、指名线路的排期——直接走
  `direct_lookup` 预取，省掉 LLM 试探选工具的来回。误判的代价只是多绕一次 ReAct，
  不会给出错误答案，所以这里刻意不上 LLM 分类器（那会给每个问题都加一次模型调用）。
- **grade（LLM-as-Judge 质检）**：按 `prompts.FAITHFULNESS_RUBRIC` 给忠实度打 1-5 分，
  低于 `FAITHFULNESS_PASS_THRESHOLD`（=4）判为不足。**该 rubric 与离线
  `eval/ragas_eval` 的 Faithfulness 维度是同一把尺**——在线门控与离线指标口径统一，
  改判据只改 `prompts.py` 一处。
- **reflect（反思重试）**：把批评意见注入对话、回到 agent 重检索，最多
  `agent.max_reflections` 次；与 `agent.max_turns` 共同保证终止。到达轮次上限时
  agent 不再带工具定义、并被明确告知不得输出工具调用语法，确保进入 grade 的状态干净。
- **状态显式**：`AgentState` + reducer（`operator.add`）把"追加消息、累计思考链"
  写进类型，而非藏在循环变量里；两个决策点收敛为两条条件边，便于审查与单测。
- **代价**：grade + 最多 2 次反思会显著增加耗时与调用次数，故与手写 ReAct 并存、按需取用。

### 2.4 多智能体：为什么拆角色而不是继续加工具

单个 ReAct Agent 拿到 6 个工具时，"排架次"和"查规程"两类任务会互相干扰——提示词要
同时约束两套输出格式，工具选择也更容易跑偏。`crew.py` 拆成三个角色，每个角色只看得见
自己该用的工具子集（tool scoping），上一棒的产物写进共享状态传给下一棒：

| 角色 | 可见工具 | 产物 |
|---|---|---|
| 规划员 planner | `plan_inspection_mission`, `lookup_asset` | 待飞塔位与架次编排 |
| **合规闸门 clearance** | **无（确定性计算）** | 逐基放行结论 |
| 诊断员 diagnostician | `search_regulations`, `search_cases`, `lookup_asset_history` | 逐基定性与处置 |
| 调度员 dispatcher | 无（只做综合） | 班组派工单 |

闸门夹在两个模型角色之间、且是确定性的，是这条链的关键：能不能飞是硬约束，不能让模型
"觉得可以"就往下走。全员禁飞时条件边直接跳过诊断员，省掉一整轮模型调用。

另外，待飞塔位列表取自确定性规划结果，不从规划员的自由文本里正则抠——模型漏写或写错
编号，下游就会对着不存在的塔做诊断。

### 2.5 为什么用 RRF 而不是分数加权

- 不同召回路（规程 / 案例）的分数量纲不一致
- RRF 只看排名不看绝对分，工业上更稳健；`k=60` 是 RRF 论文推荐的鲁棒值
- 参数收敛到 `config.yaml` 的 `retrieval.rrf_k` / `retrieval.rerank_top_k`

### 2.6 为什么按 Markdown 标题切分而不是固定 chunk size

- 规程文档结构性强，标题层级即语义边界
- 切片同时携带 `section_path`，引用溯源能精确到章节
- 超长段落再做二次滑窗切分作为兜底

### 2.7 为什么检测层只走 VLM 描述不接 YOLO

- 优先做完整闭环而非单点硬技术
- VLM 已能给出足够的"视觉关键词"驱动文本召回
- 接 YOLO 只需替换 `src/detection/yolo_stub.py:detect()` 的实现，签名稳定

## 三、模块详解

| 目录 | 文件 | 职责 |
|---|---|---|
| `ingestion/` | `text_loader.py` | Markdown 解析 + 切片；案例 / 资产 / 巡检历史统一加载 |
| `retrieval/` | `vector_store.py` | Chroma 客户端 + collection 管理 + 元数据清洗（chromadb 懒加载） |
| | `retriever.py` | 多路召回 API + RRF 融合 |
| `generation/` | `llm_types.py` | provider 中立的消息与工具调用结构 |
| | `providers.py` | Anthropic / OpenAI 协议双向翻译（纯函数） |
| | `llm_client.py` | 调用门面 + 重试 + 本地 embedding（sentence-transformers 懒加载） |
| | `prompts.py` | 集中管理 prompt 模板与共享 rubric |
| | `report_generator.py` | 基础 RAG 管线：诊断 / 报告 / 问答三个入口 |
| `mission/` | `geo.py` | haversine 距离、最近邻 + 2-opt 航线优化（纯函数） |
| | `planner.py` | 缺陷闭环判定 → 时效排序 → 航线优化 → 架次切分 |
| | `airspace.py` | 空域限制区 / 气象限值 / 带电体安全距离，取最严裁决 |
| `agent/` | `tools.py` | 6 个工具的中立 schema + 执行分发（异常转文本回灌） |
| | `agent.py` | 手写 ReAct 循环 |
| | `graph.py` | LangGraph 纠错式编排 |
| | `crew.py` | 多智能体协作链 |
| `router/` | `intent_router.py` | 规则意图分类 + 资产编号 / 线路名抽取 |
| `detection/` | `yolo_stub.py` | VLM 充当检测器返回结构化描述；后续可替换为真实 YOLO |

## 四、依赖分层与 CI

重依赖（`chromadb` / `sentence-transformers` / `openai` / `anthropic` / `streamlit`）
一律在函数内 import。带来的直接好处是 **CI 只装 `requirements-dev.txt` 就能跑完 90 个单测，
不下模型、不联网、不需要任何 API Key** —— 这本身就是对"懒加载 + 确定性核心"这个设计的回归验证。

## 五、检索质量调优手段

| 手段 | 当前是否启用 | 后续 |
|---|---|---|
| 多路召回（规程 + 案例） | 启用 | 增加图像路 |
| 元数据过滤（asset_type / defect_type） | 启用 | 增加 severity 过滤 |
| RRF 融合 | 启用 | 加入业务权重 |
| Rerank（重排模型） | 未启用 | 引入 BGE-Reranker-v2-m3 |
| 关键词召回（BM25） | 未启用 | 与向量召回组成 hybrid |

`eval/retrieval_eval.py` 的实测显示 `k` 从 6 提到 8 指标不变（Recall 0.911 / 全召回 0.800），
说明当前瓶颈不在召回条数，而在切分粒度与 query 表述——这正是上表把 rerank 与 BM25
排在前面的依据。

## 六、可观测性

`DEBUG=1` 时 `report_generator.diagnose_image` 会在 `result.debug` 中带上 query 文本，
便于检查"为什么召回了这一组"。更完整的日志体系（结构化日志 + token 统计 + 链路追踪）尚未做。

## 七、扩展点

- **换模型服务商**：改 `config.yaml` 的 `llm:` / `vlm:` 段；若是新协议族，在
  `providers.py` 加一组 `to_* / from_*` 翻译函数即可，Agent 层不动
- **换向量库**：替换 `vector_store.py` 内的 chroma 调用
- **加召回路**：在 `retriever.py` 新增 `retrieve_xxx`，调用方 `fuse_rrf` 多传一个入参
- **换空域数据源**：`data/airspace/constraints.json` 换成真实 UOM 数据，`airspace.py` 判定逻辑不变
- **接 YOLO**：替换 `detection/yolo_stub.py:detect` 实现，输入输出契约保持不变
