# 文档敏感性筛查与安全多向量 VLM-RAG

这是一个面向页面图像的研究型工程：它先判断页面是否需要进入脱敏或人工复核流程，再根据用户问题检索证据页面，最后只把安全策略允许的页面交给外部视觉大模型回答。项目以 DocVQA 为实验数据，正式检索器基于 ColPali/PaliGemma，多模态生成端接入 Doubao-Seed-2.1-pro。

> 本项目不是一个单独的“敏感信息模型”。Sensitivity、Retrieval 和 Generation 是三个监督目标、输入输出和部署职责都不同的模块。

## 目录

- [当前状态](#当前状态)
- [系统边界与端到端流程](#系统边界与端到端流程)
- [数据语义与无泄漏划分](#数据语义与无泄漏划分)
- [模型设计](#模型设计)
- [代码结构](#代码结构)
- [快速开始](#快速开始)
- [训练与评估](#训练与评估)
- [安全问答](#安全问答)
- [实验记录与验证边界](#实验记录与验证边界)
- [性能与故障排查](#性能与故障排查)
- [测试、限制与后续工作](#测试限制与后续工作)

## 当前状态

| 模块 | 状态 | 证据边界 |
| --- | --- | --- |
| 数据 manifest | 已完成 | 本地全量数据结构验收通过，两个任务均按 `doc_id` 划分且文档零交叉 |
| Sensitivity 两阶段训练 | 已完成于服务器 | checkpoint 与 test 结果尚未同步回本地仓库；README 仅列出已提供的验证集校准结果 |
| Global Retrieval | 已完成于服务器 | 训练日志数值来自用户提供的服务器运行记录，产物尚未归档到本地仓库 |
| Global hard negatives | 已生成于服务器 | 本地没有当前 JSONL，正式 Late 训练前仍需审计每条 query 的负样本数与碰撞 |
| Late Interaction smoke | 已完成于服务器 | 4,096/512 样本的 smoke 只证明训练链路和显存可运行，不代表正式检索效果 |
| Late Interaction 全量训练 | 进行中/待重新核验 | 最近一次运行使用慢速 `chunked` MaxSim；本地没有完整训练产物 |
| Retrieval test 与全量索引 | 未验证 | 只有真实 test 指标和指纹匹配索引存在后才能标为完成 |
| 真实 Doubao 评估 | 未执行 | 客户端与安全门控已通过 mock HTTP 测试，但没有使用真实 Key 调用 API |

仓库中的 `outputs/metrics_report.csv`、`outputs/retrieval_results.json`、`models/` 和 `indexes/` 是早期 18 页、24 问答模拟数据的演示产物，不能作为完整 DocVQA 的正式成绩。

## 系统边界与端到端流程

| 模块 | 输入 | 输出 | 不负责什么 |
| --- | --- | --- | --- |
| Sensitivity Classification | 单张页面图像 | 页面敏感概率、二分类结果 | 不定位字段、不涂黑图片、不检索问题、不生成答案 |
| Retrieval | 问题与完整页面语料 | 相关页面及粗排/MaxSim 分数 | 不使用 `desensitized` 作为独立语料，不决定页面能否外发 |
| Generation | 问题与安全门允许的页面 | 答案、证据页、置信度、错误记录 | 不在本地训练；不绕过安全策略读取敏感原图 |

```text
用户问题
   │
   ├─ Global Retrieval：768 维全局向量粗排 Top-128
   │
   ├─ Late Interaction：Query tokens × 1,024 Page tokens 精确 MaxSim
   │                     重排 Top-20
   │
   ├─ Sensitivity catalog + redaction policy（fail closed）
   │    ├─ 非敏感页：允许原图
   │    ├─ 敏感页：只允许已批准且哈希匹配的脱敏副本
   │    └─ 缺预测/推理错误/元数据不一致：阻断
   │
   └─ 最多 3 张允许外发的页面
        → Doubao-Seed-2.1-pro 逐页视觉问答
        → 答案融合、证据页、置信度与 JSONL 审计记录
```

正式入口是 [`scripts/answer_query.py`](scripts/answer_query.py) 和 [`scripts/evaluate_secure_rag.py`](scripts/evaluate_secure_rag.py)。`train_colpali.py`、`evaluate_generator.py`、`src/vlm_rag/generator.py` 与 `src/vlm_rag/baselines.py` 是旧单向量、拼图或兼容基线，不是生产安全链路。

## 数据语义与无泄漏划分

### 物理目录

原始数据不需要移动、复制或重命名；两个任务通过 manifest 建立逻辑数据结构。

```text
data/
├── docvqa_images/                 # 完整 DocVQA 页面
├── ocr/                           # 完整 OCR
├── docvqa_extracted/              # 完整 QA JSON
└── desensitized/
    ├── docvqa_images/             # 需脱敏正样本页面集合
    ├── ocr/
    └── docvqa_extracted/
```

标签定义只有一条：

```text
page_id ∈ desensitized/docvqa_images  → is_sensitive = 1
否则                                     → is_sensitive = 0
```

`desensitized` 是完整 DocVQA 的正标签子集，不是独立 RAG 数据集。Retrieval 始终使用完整 `docvqa_images + docvqa_extracted`；OCR 只参与数据审计和未来独立 baseline，不进入 Sensitivity 主分类器。原因是正标签本身由 OCR 规则辅助构建，直接输入 OCR 容易复现标签构造规则并形成标签泄漏。

### 全量统计

| 项目 | 数量 |
| --- | ---: |
| 页面 | 12,767 |
| 需脱敏正样本 | 3,538 |
| 负样本 | 9,229 |
| 文档 | 6,071 |
| QA queries | 50,000 |

官方 QA split 中存在相同文档跨集合的问题。本项目以固定种子按 `doc_id` 生成近似分层的 70/15/15 划分，Sensitivity 与 Retrieval 复用同一份文档归属，保证 train/val/test 的文档交集为 0。

Sensitivity 页面划分：

| split | 页面 | 正样本 | 负样本 | 文档 |
| --- | ---: | ---: | ---: | ---: |
| train | 8,937 | 2,477 | 6,460 | 4,256 |
| val | 1,925 | 533 | 1,392 | 895 |
| test | 1,905 | 528 | 1,377 | 920 |

Retrieval 查询划分：

| split | queries | pages | documents |
| --- | ---: | ---: | ---: |
| train | 35,153 | 8,937 | 4,256 |
| val | 7,291 | 1,925 | 895 |
| test | 7,556 | 1,905 | 920 |

manifest 中的文件路径相对于 `data-root` 保存为 POSIX 风格，因此可以在 Windows 生成后随数据迁移到 Linux。构建过程不会复制图片。

## 模型设计

### 基础 checkpoint

本地 checkpoint 的元数据对应 `vidore/colpaligemma-3b-pt-448-base`：

| 组件 | 配置 |
| --- | --- |
| 架构 | ColPali / PaliGemma |
| 页面输入 | 448×448 |
| Vision tower | SigLIP，27 个 encoder blocks，hidden size 1,152 |
| 页面表示 | 1,024 image tokens |
| Language model | Gemma，18 层，hidden size 2,048 |
| ColPali token embedding | 128 维 |

### Sensitivity Classification

Sensitivity 只接收页面图像。模型通过 forward hooks 保留 SigLIP 的 `[0, 8, 16, 23, 27]` 层，而不是让全部 hidden states 常驻显存；5090 配置将每层 32×32 patch 网格池化为 16×16，即每层保留 256 个区域 token。随后依次执行：

```text
可学习多层加权
→ regional projection
→ 2 层 Transformer Encoder + CLS
→ 单个页面敏感 logit
```

训练使用 `BCEWithLogitsLoss`，`pos_weight` 从实际 train split 计算。阶段 1 冻结全部 ViT，只训练区域分类头和层权重；阶段 2 从阶段 1 的 checkpoint 初始化，只解冻 ViT 最后 4 层并使用较小视觉学习率。阈值只允许在 val 上校准，test 使用固定阈值做一次最终评估。

Sensitivity 只是页面级风险分类器。它不会定位“美元金额”等具体字段，也不会生成涂黑后的图片。

### Global Retrieval

Global 阶段为大 batch 粗排训练：Query 经过 Gemma 和 rank-64 rsLoRA，页面经过视觉编码、多层融合与投影，二者都被池化到 768 维全局向量。损失同时计算 query→page 与 page→query 的对称 InfoNCE；batch sampler 避免同一 batch 内重复 `doc_id` 或正例页面，降低假负例风险。

该阶段用于快速候选召回和第一轮 hard-negative 挖掘，不是最终精排模型。

### Late Interaction Retrieval

Late 阶段不再把页面压缩成单个向量：Query 保留有效文本 token，Page 保留 1,024×128 的 ColPali image tokens，通过精确 MaxSim 学习局部文字、金额、表格单元格和版式区域之间的匹配。

默认混合损失为：

| 损失 | 权重 | 作用 |
| --- | ---: | --- |
| global InfoNCE | 0.25 | 保留稳定的粗排语义空间 |
| late interaction | 0.55 | 学习 token 级精确匹配 |
| hard-negative ranking | 0.20 | 拉开正确页与相似错误页 |

5090 快速对称配置保留全部 35,153 条 query 和可训练的 query/page 编码路径，但按页面组织 batch：默认每批选择 8 个不同文档的页面，每页最多组合 4 条 query。Query→Page 使用单目标交叉熵，Page→Query 将同页的多条 query 全部视为正例；同一 batch 内的正页和 hard-negative 页按 `page_id` 只编码一次。这不是把训练集缩减为 8,937 条，而是共享同一 optimizer step 内的重复页面前向。

每条 query 每个 epoch 使用 1 个显式 hard negative，并按 epoch 在已有的 4 个候选间确定性轮换；in-batch negatives 和容量 256 的 detached page-token queue 仍然保留。LIK 与 chunked 均使用按有效 query token 数归一化的 mean MaxSim，避免后端切换改变训练损失尺度。

训练输出包含逐 batch `tqdm` 进度条、`training.log`、逐 epoch `metrics.jsonl`、`epoch-NNN/`、`last/` 和 `best/`。`best/` 默认先保存初始化模型的 epoch-0 基线，再按固定 1,000 条 val query 的 Recall@5、MRR 选择；新 epoch 不优于基线时不会覆盖旧模型。

### Secure Generation

安全门只允许两种字节离开进程：

1. catalog 明确判定为非敏感的原图；
2. 敏感页对应的、已批准且 source/redacted SHA-256 均匹配的脱敏副本。

页面缺少 catalog、检索元数据不一致、分类失败、文件缺失、哈希变化或敏感原图没有批准副本时均默认阻断。多向量索引还会校验 adapter 与基础模型元数据指纹，避免错配权重和索引。

Doubao 客户端按页面请求结构化 JSON，验证 `relevant/answer/evidence_quote/confidence`，实现响应缓存、429/5xx 重试、指数退避和显式错误记录，再对有效页面答案进行融合。

## 代码结构

```text
configs/                         # 两个任务、显存 profile 与安全链路配置
scripts/
├── build_*manifest.py           # 数据逻辑结构
├── train_sensitivity.py         # Sensitivity 两阶段训练
├── evaluate_sensitivity.py      # val 校准与固定阈值 test
├── train_retrieval_global.py    # 全局粗排训练
├── mine_global_hard_negatives.py
├── train_retrieval_late.py      # 原生 ColPali 多向量训练
├── build_multivector_index.py
├── evaluate_retrieval_multivector.py
├── build_sensitivity_catalog.py
├── build_redaction_manifest.py
├── answer_query.py              # 正式单次安全问答入口
└── evaluate_secure_rag.py       # 正式安全链路评估入口
src/vlm_rag/
├── sensitivity/                 # schema、split、model、training、inference、catalog
├── retrieval/                   # dataset、model、losses、MaxSim、index、mining、training
├── pipeline/                    # fail-closed policy、provenance、audit、engine
└── generation/                  # Ark client、cache、fusion、schema
tests/                            # 数据、训练、推理、多向量与安全策略单元测试
```

更长的部署顺序见 [`SERVER_GUIDE.md`](SERVER_GUIDE.md)，方法背景见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，研究性说明见 [`docs/technical_report.md`](docs/technical_report.md)。技术报告中的“目标效果参考”不是本仓库已经达到的实验结果。

## 快速开始

### 1. Linux 路径与缓存

```bash
export PROJECT_ROOT=/path/to/aiproject
export DOCVQA_DATA_ROOT=/path/to/data
export COLPALI_MODEL_PATH=/path/to/checkpoint
export VLM_CACHE_ROOT=/path/to/data-disk/vlm-cache

export HF_HOME="$VLM_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$VLM_CACHE_ROOT/torch"
export PIP_CACHE_DIR="$VLM_CACHE_ROOT/pip"
export TMPDIR="$VLM_CACHE_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
cd "$PROJECT_ROOT"
```

`setup.sh` 不下载 checkpoint，但默认会尝试安装 torch 2.7.1/torchvision 0.22.1。对于已经能够训练的较新服务器环境，先运行结构检查，不要盲目覆盖 PyTorch：

```bash
export VLM_PYTHON="$(command -v python)"
bash setup.sh --no-install
```

### 2. 环境版本

本地精确锁定环境已验证为 Python 3.11、torch 2.7.1+cu128、torchvision 0.22.1+cu128、RTX 4080 Laptop 12GB；其余版本见 [`requirements.txt`](requirements.txt)。全新环境可按锁定组合安装：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

服务器曾运行 torch 2.11.0+cu130 等更新组合，这只能视为兼容运行记录，不是仓库锁定版本。结构验收可使用 `--dependency-policy compatible`；不要在训练进程运行时修改当前 Python 环境。

Late Interaction 建议额外安装 LIK：

```bash
python -m pip install --no-cache-dir "colpali-engine[lik]==0.3.17"

PYTHONPATH=src python - <<'PY'
from vlm_rag.retrieval.maxsim import (
    late_interaction_kernel_available,
    resolve_maxsim_backend,
)
print("LIK available:", late_interaction_kernel_available())
print("auto resolves to:", resolve_maxsim_backend("auto"))
PY
```

理想输出是 `True / lik`。若输出 `False / chunked`，计算仍是精确 MaxSim，但会明显更慢；必须先做短反向 smoke，不能假设 LIK 与当前 Python/PyTorch/CUDA 组合必然兼容。

正式快速对称训练还要求 LIK 与 chunked 的归一化值及梯度一致：

```bash
python scripts/verify_maxsim_backends.py
```

### 3. Manifest 与结构验收

```bash
python scripts/build_sensitivity_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT"

python scripts/build_retrieval_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT"

python scripts/validate_project.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --skip-gpu-smoke \
  --dependency-policy compatible
```

Windows 本地结构验证：

```powershell
E:\envs\vlm\python.exe scripts\validate_project.py `
  --data-root E:\实习\aiproject\data `
  --model E:\实习\aiproject\checkpoint `
  --skip-gpu-smoke `
  --dependency-policy compatible
```

## 训练与评估

### 1. Sensitivity 两阶段训练

```bash
export SENSITIVITY_HEAD_DIR="$PROJECT_ROOT/outputs/sensitivity_head"
export SENSITIVITY_FINETUNE_DIR="$PROJECT_ROOT/outputs/sensitivity_unfreeze4"
export SENSITIVITY_EVAL_DIR="$PROJECT_ROOT/outputs/sensitivity_evaluation"
export SENSITIVITY_PRED_DIR="$PROJECT_ROOT/outputs/sensitivity_predictions"

python scripts/train_sensitivity.py \
  --config configs/sensitivity_head_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --output-dir "$SENSITIVITY_HEAD_DIR"

python scripts/train_sensitivity.py \
  --config configs/sensitivity_unfreeze4_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --init-checkpoint "$SENSITIVITY_HEAD_DIR/best.pt" \
  --output-dir "$SENSITIVITY_FINETUNE_DIR"
```

阶段切换使用 `--init-checkpoint`；同一阶段中断后才使用 `--resume last.pt`。

### 2. Val 校准与固定阈值 Test

```bash
python scripts/evaluate_sensitivity.py \
  --mode calibrate \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/val.jsonl" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --target-recall 0.90 \
  --output-dir "$SENSITIVITY_EVAL_DIR"

python scripts/evaluate_sensitivity.py \
  --mode evaluate \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/test.jsonl" \
  --calibration "$SENSITIVITY_EVAL_DIR/calibration.json" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --output-dir "$SENSITIVITY_EVAL_DIR/test"
```

禁止利用 test 标签重新选择阈值。

### 3. 全量分类与 Catalog

```bash
python scripts/classify_pages.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/all.jsonl" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --calibration "$SENSITIVITY_EVAL_DIR/calibration.json" \
  --batch-size 16 \
  --format both \
  --output-dir "$SENSITIVITY_PRED_DIR"

python scripts/build_sensitivity_catalog.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/all.jsonl" \
  --predictions "$SENSITIVITY_PRED_DIR/predictions.jsonl" \
  --errors "$SENSITIVITY_PRED_DIR/errors.jsonl" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --calibration "$SENSITIVITY_EVAL_DIR/calibration.json" \
  --output "$SENSITIVITY_PRED_DIR/sensitivity_catalog.jsonl"
```

catalog 必须覆盖全部 12,767 页。损坏图片会被写入 `errors.jsonl`，构建安全 catalog 时不会被静默当成非敏感页。

### 4. Global Retrieval

先用 [`scripts/profile_training_memory.py`](scripts/profile_training_memory.py) 在实际 GPU 上标定。服务器已完成的一次稳定运行使用 physical batch 64、梯度累积 2；仓库配置仍是 profile 起点 128/1。需要复现该设置时使用临时配置，避免覆盖版本控制中的基准配置：

```bash
cp configs/retrieval_global_5090.yaml /tmp/retrieval_global_bs64.yaml
sed -i 's/^batch_size: 128$/batch_size: 64/' /tmp/retrieval_global_bs64.yaml
sed -i 's/^gradient_accumulation_steps: 1$/gradient_accumulation_steps: 2/' \
  /tmp/retrieval_global_bs64.yaml

export RETRIEVAL_GLOBAL_DIR="$PROJECT_ROOT/outputs/retrieval_global_bs64"

python -u scripts/train_retrieval_global.py \
  --config /tmp/retrieval_global_bs64.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/train.jsonl" \
  --output-dir "$RETRIEVAL_GLOBAL_DIR" \
  --epochs 10 \
  --num-workers 8 \
  2>&1 | tee logs/retrieval_global.log
```

### 5. Hard-negative 挖掘与审计

```bash
python -u scripts/mine_global_hard_negatives.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/train.jsonl" \
  --checkpoint "$RETRIEVAL_GLOBAL_DIR" \
  --base-model "$COLPALI_MODEL_PATH" \
  --output "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --page-batch-size 64 \
  --query-batch-size 128 \
  --candidate-top-k 512 \
  --negatives-per-query 4 \
  2>&1 | tee logs/mine_global_hard_negatives.log
```

同文档页面和正页面会被排除。正式 Late 训练前审计当前文件，而不是只检查脚本退出码：

```bash
python - <<'PY'
import json
from collections import Counter

path = "data/manifests/retrieval/hard_negatives.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
counts = Counter(len(row["negative_page_ids"]) for row in rows)
duplicates = len(rows) - len({row["query_id"] for row in rows})
collisions = sum(
    row["positive_page_id"] in row["negative_page_ids"] for row in rows
)
print("rows:", len(rows))
print("negative counts:", dict(sorted(counts.items())))
print("duplicate queries:", duplicates)
print("positive/negative collisions:", collisions)
PY
```

目标是 35,153 行、每行 4 个负样本、无重复 query、无正负碰撞。若候选经过同文档过滤后不足，应提高 `candidate-top-k`，不能静默接受空负样本。

### 6. Late Interaction smoke 与正式训练

先确认 LIK 后端，再执行短反向 smoke：

```bash
python -u scripts/train_retrieval_late.py \
  --config configs/retrieval_late_symmetric_fast_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --hard-negatives "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --init-adapter "$PROJECT_ROOT/outputs/retrieval_late/best" \
  --output-dir "$PROJECT_ROOT/outputs/retrieval_late_symmetric_smoke" \
  --pages-per-batch 4 \
  --queries-per-page 4 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --num-workers 8 \
  --max-train-samples 256 \
  --max-val-samples 32
```

确认训练/验证损失有限、反向正常且 `best/` 存在后再跑全量：

```bash
export RETRIEVAL_LATE_DIR="$PROJECT_ROOT/outputs/retrieval_late_symmetric_fast"

python -u scripts/train_retrieval_late.py \
  --config configs/retrieval_late_symmetric_fast_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --hard-negatives "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --init-adapter "$PROJECT_ROOT/outputs/retrieval_late/best" \
  --output-dir "$RETRIEVAL_LATE_DIR" \
  --pages-per-batch 8 \
  --queries-per-page 4 \
  --gradient-accumulation-steps 4 \
  --epochs 3 \
  --num-workers 8 \
  2>&1 | tee logs/retrieval_late.log
```

旧 adapter 只有模型权重，使用 `--init-adapter`；新的 `last/` 同时保存 optimizer、scheduler、epoch、global step 和 memory queue，使用 `--resume` 可以严格恢复。`--epochs` 在 resume 时表示本次实验的总 epoch 上限。`--init-global`、`--init-adapter` 和 `--resume` 三者互斥。

### 7. Test 索引、检索评估与生产索引

正式指标先在 test 页面语料上计算：

```bash
export RETRIEVAL_TEST_INDEX_DIR="$PROJECT_ROOT/indexes/retrieval_test"

python scripts/build_multivector_index.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --output-dir "$RETRIEVAL_TEST_INDEX_DIR" \
  --batch-size 4 \
  --pages-per-shard 128

python scripts/evaluate_retrieval_multivector.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --index-dir "$RETRIEVAL_TEST_INDEX_DIR" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --coarse-top-k 128 \
  --maxsim-backend lik \
  --maxsim-normalization mean \
  --output "$RETRIEVAL_LATE_DIR/test_metrics.json"
```

通过后再建立全量生产索引：

```bash
export RETRIEVAL_INDEX_DIR="$PROJECT_ROOT/indexes/retrieval_all"

python scripts/build_multivector_index.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/all.jsonl" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --output-dir "$RETRIEVAL_INDEX_DIR" \
  --batch-size 4 \
  --pages-per-shard 128
```

不要使用 all-page 索引评估 test，除非评估协议明确允许额外语料；README 的正式 test 结果应来自 test-only index。

## 安全问答

### 1. 可选的脱敏副本批准

Sensitivity 不生成脱敏图片。如果已有独立工具或人工产出的副本，先准备相对于 data-root 的映射：

```json
{"page_id":"example_page","redacted_path":"redacted/example_page.png"}
```

然后显式批准并记录原图/副本哈希：

```bash
python scripts/build_redaction_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --catalog "$SENSITIVITY_PRED_DIR/sensitivity_catalog.jsonl" \
  --mappings redaction_mappings.jsonl \
  --output "$SENSITIVITY_PRED_DIR/redaction_manifest.jsonl" \
  --approve
```

没有 redaction manifest 时，非敏感原图仍可回答；敏感原图会被阻断，系统继续尝试 Top-20 中排名更低的安全证据。如果没有任何安全页，返回 `blocked_sensitive_evidence`，不会调用外部 API。

### 2. Doubao-Seed-2.1-pro

任何暴露在聊天、日志或命令历史中的 Key 都应立即吊销。新 Key 只通过进程环境变量输入：

```bash
read -s -p "ARK_API_KEY: " ARK_API_KEY; echo
export ARK_API_KEY
export DOUBAO_MODEL=doubao-seed-2-1-pro
# 若方舟控制台要求 Endpoint ID：export DOUBAO_MODEL=ep-xxxxxxxx
```

先完成索引和 sensitivity catalog，再显式允许真实调用：

```bash
python scripts/answer_query.py \
  --query "What was the total amount?" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --index-dir "$RETRIEVAL_INDEX_DIR" \
  --sensitivity-catalog "$SENSITIVITY_PRED_DIR/sensitivity_catalog.jsonl" \
  --output "$PROJECT_ROOT/outputs/secure_rag/answers.jsonl" \
  --audit "$PROJECT_ROOT/outputs/secure_rag/audit.jsonl" \
  --allow-real-api
```

有批准副本时增加：

```bash
--redaction-manifest "$SENSITIVITY_PRED_DIR/redaction_manifest.jsonl"
```

小规模端到端评估：

```bash
python scripts/evaluate_secure_rag.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --index-dir "$RETRIEVAL_INDEX_DIR" \
  --sensitivity-catalog "$SENSITIVITY_PRED_DIR/sensitivity_catalog.jsonl" \
  --output-dir "$PROJECT_ROOT/outputs/secure_rag/eval" \
  --max-queries 10 \
  --allow-real-api
```

安全验收要求 `sensitive_original_exposure = 0` 且 `missing_catalog_exposure = 0`。

## 实验记录与验证边界

### 仓库内可复核结果

截至当前版本，本地重新运行了 24 个单元测试，全部通过；结构验收确认依赖、CUDA/BF16、数据目录、两个 manifest 和基础 checkpoint 均有效。真实 GPU forward 在本次 README 验收中主动跳过，没有重新消耗显存。

### Sensitivity 验证集校准

以下数值来自用户提供的服务器 val 校准结果，不是 test 指标：

| 指标 | 数值 |
| --- | ---: |
| ROC-AUC | 0.91366 |
| PR-AUC | 0.82675 |
| target-recall threshold | 0.291225 |
| Recall | 0.90056 |
| Precision | 0.55046 |
| F1 | 0.68327 |
| False-negative rate | 0.09944 |

用户报告固定阈值 test 已运行，但 test JSON 尚未同步到本地，所以这里不预填 test 指标。

### 服务器训练运行记录

下表来自用户提供的服务器终端输出，尚未作为仓库 artifact 归档：

| 阶段 | 设置 | 结果 | 解释 |
| --- | --- | --- | --- |
| Global Retrieval | batch 64，accumulation 2，10 epochs | final train loss 0.298547；peak allocated 22.91 GB；peak reserved 23.32 GB | 训练损失与显存记录，不是 test 检索指标 |
| Late smoke | batch 8，accumulation 12，4,096/512，1 epoch | train loss 1.0932；val loss 0.8857；peak reserved 13.82 GB | 只证明短链路可反向，不代表正式模型效果 |
| Late full | 最近使用 `chunked` 后端 | 约 9 小时/epoch 的运行观察 | 当前未确认完成，不能给出最终 best 或 test 指标 |

在 `outputs/retrieval_late/test_metrics.json`、test-only index 和对应 adapter 被同步并通过指纹校验前，项目不声称已经取得正式 Retrieval test 成绩。真实 Doubao EM/Accuracy、成本和延迟也仍待小规模 API 验收。

## 性能与故障排查

| 现象 | 原因与处理 |
| --- | --- |
| 交互式 Python 报 `No module named vlm_rag` | 仓库采用 `src/` layout；使用 `PYTHONPATH=src python ...`，脚本入口已自动加入该路径 |
| LIK 检查为 `False / chunked` | `requirements.txt` 不安装 extra；停止训练进程后安装 `colpali-engine[lik]`，再做短反向 smoke |
| Late 一个 epoch 很慢 | 使用 `retrieval_late_symmetric_fast_5090.yaml`；确认 LIK、页面分组、单个轮换 hard negative 和同 batch 页面去重均已生效 |
| 显存低但训练不快 | 显存容量与吞吐不是同一指标；检查训练中 GPU Util、数据解码和 MaxSim backend |
| Global profile 通过但正式训练 OOM | query 长度、完整反向、allocator 峰值和 profile 样本可能不同；采用稳定 batch 64/accumulation 2，并保留余量 |
| hard negatives 少于 4 | 同文档过滤后候选不足；提高 `candidate-top-k` 并重新审计，不要静默训练空负样本 |
| `--resume` 报缺少 `training_state.pt` | 历史 adapter 使用 `--init-adapter`；只有新版 `last/` 支持严格 `--resume` |
| index provenance mismatch | adapter、基础模型或脚本已变化；用当前 adapter 和 manifest 重建多向量索引 |
| dependency exact 校验失败 | 新服务器环境使用 `--dependency-policy compatible`；只有复现本地锁定环境时才使用 exact |
| 12GB 上 Late backward OOM | 本地 12GB 只适合结构/前向和小 smoke；全量 Late 使用 32GB GPU 并先 profile |
| 豆包 401/403 | 停止重试，检查新 Key、服务开通状态、模型名或 Endpoint ID；不要把 Key 写进文件 |
| 没有脱敏副本 | 敏感页会被 fail-closed 阻断；系统不会假装完成字段级脱敏 |

训练时可在第二个终端观察吞吐：

```bash
nvidia-smi dmon -s pucm -d 2
```

如果 GPU `sm` 长期很低，优先检查图片 I/O、DataLoader、LIK 是否启用以及 hard-negative 的多次页面 forward；不要单纯继续增大 batch。

## 测试、限制与后续工作

### 验收命令

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/test_integration.py
python scripts/test_integration_generator.py
python -m pip check
python scripts/validate_project.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --skip-gpu-smoke \
  --dependency-policy compatible
git diff --check
```

### 当前限制

- Sensitivity 是页面级风险筛查，不是字段检测、OCR 规则解释器或自动涂黑系统。
- 完整 Retrieval test、全量生产索引和真实 Doubao 评估尚未在本地归档。
- 新版 Late 的 `last/training_state.pt` 支持 optimizer、scheduler、epoch、global step 与 queue 严格恢复；历史旧 adapter 只能作为 `--init-adapter`。
- 页面分组会在同一 batch 内共享正页/负页前向，但不会跨 optimizer step 缓存最终页面向量；query/page 对称训练仍然保留。
- LIK 是否可用取决于服务器的 Python、PyTorch、CUDA 和已安装 wheel；`chunked` 正确但速度较慢。
- Doubao 融合是逐页回答后融合，不等同于一次性多图联合推理。
- 早期 OCR-RAG、SigLIP、单向量 ColPali 和图像拼接结果只用于研究基线。

### 后续优先级

1. 安装并验证 LIK 的训练反向，记录同一数据子集的端到端加速比。
2. 在 5090 上分别 profile gradient checkpointing 开/关，选择 reserved 不超过 28GB 的最快配置。
3. 将服务器 Sensitivity test、Global/Late checkpoint、日志与 retrieval test JSON 归档到可审计实验目录。
4. 完成 test-only 多向量索引评估，再构建全量生产索引。
5. 在严格外发门控下完成 1–10 条真实 Doubao 请求，记录费用、延迟、EM/Accuracy 和失败类型。
6. 可选地评估冻结视觉前缀缓存；只有端到端 profiler 证明 SigLIP 占比足够高时再引入。

## 安全说明

- 不要提交 `data/`、checkpoint、adapter、index、缓存、日志、`.env` 或任何 API Key。
- `--allow-real-api` 是真实外部传输的显式开关；测试和 smoke 不应携带它。
- catalog 缺失、推理错误和敏感原图无副本均必须 fail closed。
- 如果页面内容或脱敏副本发生变化，应重新分类、重建 catalog/redaction manifest，并重新生成索引指纹。
- 本项目当前是研究工程，不应在没有权限控制、人工复核和组织合规评估的情况下直接处理生产敏感文档。
