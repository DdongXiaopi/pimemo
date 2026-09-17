# PiMem

> 确定性、零 LLM 依赖的长程记忆中间件 + 一份 LongMemEval 架构实验（证明"呈现层"才是弱模型端到端准确率的真正杠杆）。

PiMem is a deterministic, **LLM-free** long-term memory middleware for agents. This repository bundles the middleware (`pimem/`) together with a reproducible LongMemEval architecture study that isolates the **rendering layer** as the real lever behind end-to-end accuracy.

---

## 本仓库包含什么 / What's in this repo

| 路径 | 内容 |
|---|---|
| `pimem/` | PiMem 中间件：源码、测试、文档（`docs/`）、评测（`eval/`）、LICENSE（MIT） |
| `PIMEM_架构实验报告_草稿_2026-09-17.md` | 架构实验报告（定稿）：呈现层断点、Memory Answer Synthesis 方案、隔离实验、全量 500 结果 |
| `_rerender_synth.py` | 确定性重渲染（把离线召回 pack 重写成"gold 置顶 + 结构化"上下文，无 LLM） |
| `_online_from_packs.py` | 用 LLM 网关对 pack 批量生成答案（需 API，注意依赖 venv） |
| `_judge_resilient.py` | 带探针自愈 + 断点续跑 + 增量落盘的判定脚本 |
| `check_judge.py` | 本地核对 judged / pending / accuracy |
| `_online_synth_full500/rows_judged.jsonl` | 全量 500 判定结果（端到端 0.7880） |

中间件本身的使用、安装、离线测试见 [`pimem/README.md`](pimem/README.md)。

---

## 指标口径（务必分开看）/ Metrics caliber

本项目坚持两层指标，不把召回代理指标混同为答案准确率，也不声称 SOTA。

### 检索层（PiMem 自身贡献，无 LLM）— 全量 500
- `target_session_hit`（会话级证据召回）：**0.958**
- gold-in-context 率：**0.840**（420/500）

### 端到端（依赖外部答题 LLM）

| 口径 | 渲染 | 答题模型 | 准确率 |
|---|---|---|---|
| 120 题分层抽样 | fix1 原平铺 | pro 生成 | 0.367（baseline） |
| 120 题分层抽样 | fix1 原平铺 + anti-refusal prompt | pro 生成 | 0.392（v2） |
| 120 题分层抽样 | **synth 置顶 + 结构化** | **pro 生成** | **0.892** |
| 全量 500 | **synth 置顶 + 结构化** | flash 生成 + flash 判定 | **0.7880**（394/106，judged 500/500） |

**关键结论**：同弱模型、同 prompt，仅改渲染层，端到端从 0.367 → 0.892（+0.525）；仅改 prompt 几乎无效（0.367 → 0.392）。全量 500 的 0.7880 与 120 题 0.892 的差异来自**生成模型代际（pro → flash）**，不是架构回退；两套口径都证明"渲染层是杠杆"。

### 全量 500 按题型准确率（flash synth）

| 题型 | 准确率 |
|---|---|
| single-session-user | 0.957 |
| single-session-assistant | 0.929 |
| knowledge-update | 0.782 |
| multi-session | 0.774 |
| temporal-reasoning | 0.684 |
| single-session-preference | 0.667 |

弱项（preference + temporal-reasoning）= 弱模型能力上限，非中间件缺陷。

> 口径纪律：本仓库不声称 SOTA、不把"无预算上限"基线数字算作自身成绩、不把召回代理指标混同为答案准确率。所有数字均来自真实判定文件复算（judge_success = 1.0），可复现。

---

## 复现 / Reproduce

### A. 中间件离线测试（无需 API，CI 已覆盖）
```bash
cd pimem
pip install -e ".[test]"
pytest
```
142 个离线单元测试，全绿，无需任何 API key。

### B. 架构实验端到端复现（需 API + 数据集）
前置条件：
1. **LongMemEval 数据集**（约 2.7GB，不含于本仓库，需自行获取）。
2. **LLM 网关**：配置环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY`（默认指向 aibh.cc 网关 `https://www.aibh.cc/v1`，也可换成任意 OpenAI 兼容端点），`PIMEM_MODEL` 指定生成模型；亦支持在仓库根目录放 `.env`。
3. **Python venv**（在线脚本依赖 `httpx` + `openai`）：
   ```bash
   python -m venv .venv_pimem_online
   .venv_pimem_online/Scripts/activate   # Windows
   pip install httpx openai
   ```

> ⚠️ 约 12GB 的离线召回 pack 不进 git；以下命令依赖你本地已准备好的 pack 目录。

步骤：
```bash
# 1) 确定性重渲染（默认 python 即可，无需 venv）
python _rerender_synth.py <in_pack_dir> <out_pack_dir>

# 2) 调 LLM 生成答案（必须用 venv）
.venv_pimem_online/Scripts/python.exe _online_from_packs.py <out_pack_dir> \
    --out _online_synth_full500/rows.jsonl \
    --model "deepseek-v4-flash（特价版）" --workers 6

# 3) 判定（带探针自愈 + 断点续跑 + 增量落盘）
.venv_pimem_online/Scripts/python.exe _judge_resilient.py

# 4) 本地核对
.venv_pimem_online/Scripts/python.exe check_judge.py
```
详细命令与参数见架构实验报告与 `pimem/docs/METRICS.md`。

---

## 许可证 / License
中间件：[MIT](pimem/LICENSE)。架构实验报告与脚本随仓库提供，供研究与复现。
