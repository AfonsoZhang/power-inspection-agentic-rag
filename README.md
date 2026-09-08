# 无人机巡检 Agentic RAG 系统

> LLM 自主编排工具完成「智能问答 → 缺陷诊断 → 复飞任务规划 → 飞行前合规校验 → 作业仿真 → 报告生成」全链路。
>
> 三种编排（手写 ReAct / LangGraph 纠错式 / 多智能体协作）跑在同一套工具上，可直接对比；
> 再往上有一层 **Graph-as-Policy**：作业流程被编译成可静态校验的计算图，在内置仿真里搜参数，
> 最后固化成一份 JSON，脱离模型也能执行。

[![CI](https://github.com/AfonsoZhang/power-inspection-agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AfonsoZhang/power-inspection-agentic-rag/actions/workflows/ci.yml)

## 核心亮点

| 特性 | 说明 |
|---|---|
| **Agentic RAG** | LLM 通过 tool use 自主决定调用哪些工具、检索什么，而非固定 retrieve→generate 管线 |
| **工具分两类** | 4 个语义检索工具 + 4 个**确定性作业工具**（复飞规划 / 合规校验 / 作业仿真 / 策略自学习）。后者不含模型调用，答案唯一且可单测 |
| **三种编排对比** | 手写 ReAct 循环 · LangGraph（router 路由 + grade 质检 + reflect 反思）· 多智能体（规划员→合规闸门→诊断员→调度员） |
| **低空作业闭环** | 缺陷时效 → 待飞塔位 → 航线优化（最近邻 + 2-opt）→ 电池架次切分 → 空域/气象/带电体安全距离裁决 |
| **Graph-as-Policy** | 作业流程 = 类型化计算图。模型只在编译期拼图（带"校验→打回重修"闭环），运行期按图执行；图能在蒙特卡洛仿真里被搜索、被安全判据否决，也能存成 JSON 脱机执行 |
| **provider 中立** | 默认 DeepSeek，一行配置切到任何 OpenAI 兼容端点或 Anthropic 协议；Agent 层不出现任何 SDK 类型 |
| **本地 Embedding** | sentence-transformers（BAAI/bge-small-zh-v1.5），检索侧零外部 API |
| **142 个单测 + CI** | 不下模型、不联网、不需要 API Key 就能全绿 |

## 快速开始

```bash
git clone https://github.com/AfonsoZhang/power-inspection-agentic-rag.git
cd power-inspection-agentic-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) 单测：不需要 API Key，不联网
pip install -r requirements-dev.txt && pytest -q

# 2) 完整运行
pip install -r requirements.txt
cp .env.example .env        # 填入 DEEPSEEK_API_KEY
python scripts/build_index.py
streamlit run app/streamlit_app.py --server.headless true
```

打开 http://localhost:8501 。常用命令都在 `Makefile` 里（`make test` / `make index` / `make app` / `make eval-retrieval`）。

### 模型配置

`config.yaml` 分 `llm:` / `vlm:` / `embedding:` 三段：

```yaml
llm:                          # 文本推理（Agent 主干，必须支持 function calling）
  provider: deepseek          # deepseek | openai | anthropic
  model: deepseek-chat
  base_url: ""                # 留空用 provider 默认
  api_key_env: DEEPSEEK_API_KEY

vlm:                          # 多模态（可选）。DeepSeek 无视觉模型，故单独配置
  enabled: false              # false 时前端自动隐藏图像入口，纯文本功能不受影响
```

任何 OpenAI 兼容端点（通义千问、vLLM、Ollama、OpenAI 本体）都用 `provider: openai` + 自定义 `base_url`。
启用多模态的例子（通义千问 VL）：

```yaml
vlm:
  enabled: true
  provider: openai
  model: qwen-vl-max
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY
```

## Agent 工具

| 工具 | 类型 | 功能 |
|---|---|---|
| `search_regulations` | 语义检索 | 行业规程条款（DL/T 741 等风格） |
| `search_cases` | 语义检索 | 历史缺陷案例，支持缺陷/资产类型过滤 |
| `lookup_asset` | 结构化查表 | 资产档案（线路、型号、投运年份、经纬度） |
| `lookup_asset_history` | 结构化查表 | 指定资产的巡检历史 |
| `plan_inspection_mission` | **确定性计算** | 复飞任务规划：时效筛选 → 优先级排序 → 航线优化 → 架次切分 |
| `check_flight_clearance` | **确定性计算** | 飞行前合规：空域限制区 / 气象限值 / 带电体安全距离 |
| `simulate_flight_policy` | **确定性计算** | 蒙特卡洛仿真当前策略图：完成率 / 架次中断率 / 备降风险 |
| `optimize_flight_policy` | **确定性计算** | 在仿真里搜更优的策略图，按「安全 → 完成率 → 不中断率 → 吞吐」排序 |

后四个工具刻意不含模型调用。时效判定、航程与续航测算、空域限高比对都有唯一正确答案，
交给模型只会引入不可控误差且无法回归；做成工具后可以断言"每个架次不超电池续航"
"高优先级不会被航线优化排到低优先级之后"，CI 里不用 API Key 就能守住。
模型只决定何时调用、以及如何向人解释结果。

## 三种编排

```text
① 手写 ReAct（agent.py）        agent ⇄ tools，最小可用基线

② LangGraph（graph.py）
         ┌─(工具已确定)→ [direct_lookup] ─┐
   start →[router]                        ├→ [agent] ⇄ [tools]
         └─(需自主编排)───────────────────┘        │
                                                   ▼
                                                [grade] ─(通过/反思用尽)→ END
                                                   └─(不足)→ [reflect] → [agent]

③ 多智能体（crew.py）
   [规划员] → [合规闸门(确定性)] ─┬→ [诊断员] → [调度员] → 派工单
                                  └─(全员禁飞)────────↑
```

- **router 是真条件分支**：含资产编号的历史查询、含资产编号的合规校验、指名线路的排期
  这三类「所需工具已确定」的问题直接确定性预取，跳过 LLM 试探选工具的来回；其余走 ReAct。
- **grade 与离线评测同尺**：在线质检和 `eval/ragas_eval` 的 Faithfulness 用同一份
  `prompts.FAITHFULNESS_RUBRIC`，改判据只改一处。
- **多智能体的价值在闸门**：能不能飞是硬约束，闸门做成确定性节点夹在两个模型角色之间，
  不让模型"觉得可以"就往下走；全员禁飞时直接跳过诊断，省一整轮调用。

## Graph-as-Policy：把作业流程变成一张可校验、可仿真、可脱机执行的图

思路来自 [GaP (arXiv:2607.05369)](https://arxiv.org/abs/2607.05369)——agent 把指令编译成
带类型的技能图，在内置仿真里迭代改图，最后脱离 agent 在边缘端执行。这里做的是它的电力巡检版本。

```text
作业意图 ──▶ [compiler] ──▶ 图 JSON ──▶ validate()  ──失败──┐
   (LLM 唯一参与的一步)          ▲                          │
                                └────── 错误清单打回重修 ◀──┘
                                             │ 通过
                                             ▼
                     ┌──────────────── MissionPolicy ────────────────┐
                     ▼                                               ▼
              [simulator] 蒙特卡洛推演                        [execute] 按图作业
                     │  风况/电池健康/悬停分散/临时缺陷              （不再需要模型）
                     ▼
              [selflearn] 240 张候选图择优 ──▶ learned_policy.json
```

**技能库（图节点只能取自这里）**：`sense_assets` · `sense_weather` · `collect_due_towers` ·
`screen_airspace` · `order_route_nn` / `order_route_2opt` · `split_sorties` · `fly_sorties`。

**静态校验拦四类错**：技能不存在 / 参数越出取值域、端口类型对不上、图成环、
**control 节点上游没有 `screen_airspace`**（未经合规筛查不得进入执行）。
校验器是确定性的，所以"图合不合法"从来不由模型说了算——模型输出非法图会被带着错误清单打回重修，
重修用尽则降级到基线图，而不是把非法的图放出去。

### 自学习结果（`make eval-policy`，**不需要 API Key，任何人可复现**）

240 张候选图 × 30 次蒙特卡洛，公共随机数（同一 trial、同一基塔用同一组扰动），种子固定：

| 指标 | 基线图 | 最优图 |
|---|---|---|
| 备降/迫降率 | 0.4% | **0.0%** |
| 完成率 coverage | 43.5% | **44.6%** |
| 吞吐（基/飞行小时） | 6.25 | **6.31** |
| 平均完成塔位 | 14.8 | **15.2** |

**最有信息量的不是这张表，而是被否决的那张图**：`nn|res0|cap3|rth0|la7` 的 coverage 达到
51.5%，比最优图高 6.9 个百分点——但它的备降/迫降率是 7.5%。排序把安全放在第一位且不可交换，
所以它连同另外 154 张图一起被否决。只按吞吐挑图，挑中的就是它。

> **仿真是什么**：作业时序与续航的随机模型（四个随机源：风况、电池健康、悬停时长分散、
> 临时发现缺陷），**不是**飞行动力学仿真。系数是工程假设值而非实测标定，且转场只统计架次内
> 塔间航段（起降点往返未建模），所以绝对数字没有外部效度，只能用于策略之间的横向比较。
> 完整口径写在 [`src/policy/simulator.py`](src/policy/simulator.py) 顶部。

## 评测

### 检索质量（`make eval-retrieval`，**不需要 API Key，任何人可复现**）

在 20 条 golden QA 上评规程库 + 案例库的召回。剔除 5 条多模态题（输入是图像，
用题干做文本检索衡量不到真实链路），`asset::` 参考走确定性查表、不混入检索指标。

| 指标 | k=6 | k=8 |
|---|---|---|
| Recall@k | **0.911** | 0.911 |
| MRR | **0.943** | 0.941 |
| 命中率（至少命中一条参考章节） | **1.000** | 1.000 |
| 全召回率（参考章节全部召回） | **0.800** | 0.800 |
| 资产查表成功率 | 1.000 | 1.000 |

k 从 6 提到 8 指标不变 → 瓶颈不在召回条数，而在切分粒度与 query 表述。
逐题明细（含漏召回的 3 题）见 [`eval/results/`](eval/results/)。

### 生成质量（需要 API Key）

- `make eval-kpi` — 业务 KPI：平均耗时 / 引用命中率 / 等级判定准确率
- `make eval-judge` — LLM-as-Judge 三模式对比（基础 RAG / Agent / Agent+图像），
  三个维度 Faithfulness、Answer Relevancy、Context Precision

> 这两项依赖模型服务商，不同 provider 结果会变，所以本 README 不预先写死数字——
> 跑完后 `eval/results/` 里就是可核对的原始记录。

## 目录结构

```text
├── app/streamlit_app.py       # Streamlit Demo（9 Tab）
├── src/
│   ├── agent/
│   │   ├── agent.py           # 手写 ReAct 循环
│   │   ├── graph.py           # LangGraph 纠错式编排
│   │   ├── crew.py            # 多智能体协作链
│   │   └── tools.py           # 8 个工具的中立 schema + 执行分发
│   ├── mission/               # 低空作业（确定性）
│   │   ├── geo.py             #   haversine + 最近邻 + 2-opt
│   │   ├── planner.py         #   缺陷闭环判定 → 时效 → 航线 → 架次
│   │   └── airspace.py        #   空域 / 气象 / 带电体安全距离
│   ├── policy/                # Graph-as-Policy（确定性）
│   │   ├── skills.py          #   技能库：类型签名 + 参数取值域
│   │   ├── policy_graph.py    #   图结构 / 静态校验 / 序列化 / 执行
│   │   ├── compiler.py        #   LLM 编译作业意图 → 图（含重修闭环）
│   │   ├── simulator.py       #   蒙特卡洛作业仿真
│   │   └── selflearn.py       #   候选图搜索与择优
│   ├── generation/
│   │   ├── llm_types.py       # provider 中立的消息与工具调用结构
│   │   ├── providers.py       # Anthropic / OpenAI 协议双向翻译（纯函数）
│   │   ├── llm_client.py      # 调用门面 + 重试 + 本地 embedding
│   │   ├── prompts.py         # prompt 模板 + 共享 rubric
│   │   └── report_generator.py
│   ├── retrieval/             # Chroma 向量库 + 多路召回 + RRF
│   ├── ingestion/             # Markdown 切分 + 数据加载
│   ├── router/                # 规则意图分类
│   ├── detection/             # VLM 缺陷描述（YOLO stub）
│   └── config.py
├── data/
│   ├── regulations/           # 行业规程（3 份 Markdown → 49 chunks）
│   ├── defects_history/       # 历史缺陷案例（12 条）
│   ├── assets/                # 52 基杆塔档案 + 75 条巡检历史
│   ├── airspace/              # 空域限制区 / 气象限值 / 安全距离
│   └── policies/              # 自学习固化的策略图（可脱机执行）
├── eval/                      # 检索评测 + 策略自学习（均免 Key）+ 业务 KPI + LLM-as-Judge
├── tests/                     # 142 个单测
├── docs/                      # 架构说明 / PRD / ROI
└── config.yaml
```

## Agentic RAG vs 基础 RAG

| 维度 | 基础 RAG | Agentic RAG（本项目） |
|---|---|---|
| 检索策略 | 固定管线，每次都检索 | LLM 自主判断是否需要检索、检索什么 |
| 工具编排 | 无 | 8 个工具，含确定性作业计算与仿真 |
| 多轮推理 | 单轮 | 可换关键词再搜；LangGraph 版还有质检-反思重试 |
| 多模态 | 不支持 | VLM 看图 + Agent 检索协同（需配置 vlm） |
| 可解释性 | 仅返回答案 | 展示路由 / 思考 / 工具调用 / 质检 / 反思全过程 |

## 数据说明

所有数据均为合成 / 公开来源，不涉及任何真实企业数据：

- **行业规程**：参考 DL/T 741、Q/GDW 1799 等公开标准的写作风格自行撰写
- **缺陷案例**：基于公开数据集标注思路自行合成
- **资产档案 / 巡检历史**：虚拟杆塔，沿真实城市走向生成经纬度，用于航线与空域计算
- **空域限制区**：合成演示数据，仅用于展示判定逻辑。**真实作业必须以民航局 UOM 系统与属地管理部门发布的空域信息为准**
- **样本图像**：来自 FINet SFID 公开绝缘子数据集

## 文档

- [技术架构与设计决策](docs/architecture.md)
- [PRD v1](docs/PRD_v1.md) · [v1.1 迭代记录](docs/PRD_v2_iteration.md) · [用户访谈](docs/user_interviews.md) · [ROI 测算](docs/ROI_one_pager.md)

## License

[MIT](LICENSE)
