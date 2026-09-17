# PiMem 评测复现指南 (Evaluation Reproducibility)

本目录下的脚本构成 PiMem 在 **LongMemEval** 上的完整评测链路。
所有数字均来自全量 500 题 `report.json` / `rows*.jsonl`，可复现。

## 0. 数据准备

从 [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 获取数据集
（本项目使用的清洗版约 2.7GB，记为 `DATA_JSON`）。
评测脚本通过 `--input DATA_JSON` 流式加载，无需一次性读入内存。

## 1. 离线证据指标（无需 API key、无需联网）

衡量 PiMem 自身的**召回与预算分配**能力（确定性、可审计）。

```bash
pip install -e ".[eval]"
python pimem/eval/diagnose_longmemeval_evidence.py \
    --input DATA_JSON \
    --output runs/full500 \
    --budget-mode hard \
    --token-budget 8192 \
    --max-blocks 3 \
    --limit 500 \
    --chain-artifacts all
```

产出 `runs/full500/report.json`，含每题 metrics：
`target_session_hit` / `answer_candidate_hit` / `model_context_coverage` /
`answer_turn_hit` / 失败阶段分级。聚合见 `docs/METRICS.md`。

> CI 冒烟：无需 2.7GB 数据集，`pytest pimem/tests/test_longmemeval.py` 用合成 fixture 覆盖适配器与证据召回逻辑。

## 2. 在线端到端（需要答题模型 API）

衡量"召回上下文 + 外部答题 LLM"的端到端答案准确率。**该指标取决于你用哪个答题模型**。

### 2.1 生成回答

```bash
export OPENAI_API_KEY=...      # 答题模型 key
export OPENAI_BASE_URL=...     # 兼容 OpenAI 的 endpoint
export PIMEM_MODEL=...         # 例如 gpt-4o / claude-... / deepseek-v4-pro
python pimem/eval/run_longmemeval_real_api.py \
    --input DATA_JSON \
    --output runs/online500/rows.jsonl \
    --context-token-budget 8192 \
    --max-blocks 3
```

### 2.2 判定对错

```bash
python pimem/eval/evaluate_qa_aibh.py \
    runs/online500/rows.jsonl \
    <REFERENCES_JSONL> \
    --output runs/online500/judged.jsonl \
    --env-file pimem/.env
```

`<REFERENCES_JSONL>` 为数据集中的 gold 答案（与 `rows.jsonl` 按 `question_id` 对齐）。
判定结果 `judged.jsonl` 含每题 `correct` 字段，聚合得端到端准确率。

## 3. 口径提醒

- `target_session_hit` 是**会话级证据召回代理指标**，不是答案准确率。
- 在线准确率 = `召回率 × 答题模型质量`，瓶颈常在外层答题模型而非 PiMem 召回。
- 不声称 SOTA、不把无预算上限基线算作自身成绩。详见 `docs/METRICS.md`。
