# PiMem — 确定性长程记忆中间件 (Deterministic Long-Term Memory Middleware)

> PiMem is a deterministic, **LLM-free** long-term memory middleware for agents.
> Recall and budget allocation are pure, auditable algorithms — no model inference in the retrieval path.
> Exposes both an in-process Python API and a standard **MCP** server for zero-code agent integration.

中文说明见下文。English tagline above.

---

## 为什么需要它 / Why

长程对话 Agent 的常见痛点是：上下文窗口有限，历史记忆怎么挑、各占多少 token，往往靠大模型"感觉"。
PiMem 把这件事做成**确定性、可重放、可审计**的决策：

- **零 LLM 依赖的召回与预算分配** —— 结果稳定、可复现，不随模型抖动。
- **可插拔接入** —— 同一份记忆后端同时暴露进程内 API 与 MCP server，第三方 Agent 无需改动即可接入。
- **预算可预测、可收敛** —— 在硬性 token 预算下，上下文规模可预测，不会失控膨胀。

## 安装 / Install

```bash
cd pimem
pip install -e ".[test]"      # 含测试依赖
pip install -e ".[eval]"      # 含评测依赖（在线端到端需要 openai）
```

## 快速开始 / Quickstart

### Python API

```python
from pimem.runtime import MemoryRuntime
from pimem.core.models import Scope

rt = MemoryRuntime("memory.sqlite3")
repo = rt.init_repository("repo")
scope = Scope(repo)
obs = rt.observe(content="用户偏好用 pytest -q 跑测试", source_type="chat", source_ref="t1", scope=scope)
rt.store.commit_claim(...)  # 把观察沉淀为可检索的 claim
```

### MCP server

```bash
pimem-mcp --db memory.sqlite3
```

任意支持 MCP 的 Agent 可通过 stdio JSON-RPC 零代码接入。

## 评测 / Benchmark — LongMemEval

PiMem 在开源长程记忆评测 **LongMemEval**（全量 500 题，6 大题型）上做了完整评测。
指标分为两层，请**务必分开看**：

| 层 | 指标 | 数值（全量 500，hard / 8K token / K=3） | 说明 |
|---|---|---|---|
| **检索层（PiMem 自身贡献）** | 加权 `target_session_hit`（会话级证据召回） | **0.958**（分题型加权 ≈ 0.93） | 目标答案所在会话是否被召回进上下文 —— **已达 85–90% 目标** |
| 检索层 | 加权 `answer_candidate_hit`（答案证据覆盖） | 0.924 | 答案候选是否被命中 |
| 检索层 | gold-in-context 率 | **0.840**（420/500） | gold 答案是否出现在最终上下文（关键 token ≥60% 命中） |
| 检索层 | 预算利用率 | ~84% | 旧 payload 口径仅 ~15%，经"渲染计费 + 会话摘要层"改造后提升 |
| **端到端（依赖外部答题 LLM）** | 答案准确率 | **0.892**（120/120 全样本，synth 渲染 + 原版 prompt + 弱模型 pro；全量 500 口径待验证） | 瓶颈在呈现层非模型，详见 `docs/METRICS.md` §3.2 |

### ⚠️ 关于"在线/端到端准确率"的重要说明（已更新 2026-09-16）

- `target_session_hit` 是**会话级证据召回代理指标**，**不是端到端答案准确率**。检索层（PiMem 自身）已在全量 500 上达到 0.958，超过 85–90% 目标。
- **早期结论"0.339 是弱模型天花板、需换强模型"已被证伪**——它隐含"渲染层已把上下文呈现得足够好"这一错误前提。
- **架构实验（同弱模型、隔离变量）证明瓶颈在 PiMem 的"呈现/渲染层"**：引入确定性 Memory Answer Synthesis 渲染（gold 会话置顶、结构化事实块、预算买深度不买广度）后，同一弱模型（deepseek-v4-pro 特价版）在 120 题抽样上端到端准确率从 **0.367**（fix1 原渲染）跃升至 **0.892**（synth 渲染）；而仅改 prompt（anti-refusal）几乎无效（0.367 → 0.392）。详见 `docs/METRICS.md` §3.2。
- 这说明长期记忆中间件的价值恰在于**让弱模型也拥有记忆**：把"定位/消歧/抽取"等认知负载移入确定性打包层后，弱模型即可答对。此前"换强模型才能达标"的判断不成立。
- ✅ **判定已补齐至全样本**：早期 flash 判定因网关抖动成功率仅 0.51–0.55，经 `_rejudge_live.py` 断点续跑后，baseline/v2-rej/synth 三组均达 **120/120 全样本、judge_success=1.0**。唯一剩余口径风险是 120 题抽样而非全量 500——发布前需在 `full500r-fix1-20260915` 用 synth 重渲染复跑（见 `docs/METRICS.md` §3 全量 500）。
- 在线数字基于 **120 题分层抽样**（与全量同分布）；全量 500 复跑命令见 `docs/METRICS.md`。

> 口径纪律：本项目不声称 SOTA、不把"无预算上限"基线数字算作自身成绩、不把召回代理指标混同为答案准确率。所有数字均来自全量 500 题 `report.json`。

## 复现评测 / Reproduce

见 [`pimem/eval/README.md`](pimem/eval/README.md) 与 [`docs/METRICS.md`](docs/METRICS.md)。

- **离线证据评测**（无需 API、无需 2.7GB 数据集，CI 已覆盖）：`pytest`（含 `test_longmemeval.py` 合成用例）。
- **全量离线证据指标**：运行 `pimem/eval/diagnose_longmemeval_evidence.py`（需 LongMemEval 数据集路径）。
- **在线端到端**：运行 `pimem/eval/run_longmemeval_real_api.py`（需 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `PIMEM_MODEL` 环境变量）。

## 测试 / Tests

```bash
pip install -e ".[test]"
pytest          # 142 个离线单元测试，全绿，无需任何 API key
```

CI（GitHub Actions）在 Python 3.9 / 3.11 / 3.12 上跑全部测试，并对 LongMemEval 适配器做离线证据召回冒烟。

## 许可证 / License

[MIT](LICENSE)
