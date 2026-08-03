# 文档敏感性筛查与安全多向量 VLM-RAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7+cu128-red.svg)](https://pytorch.org/)
[![ColPali](https://img.shields.io/badge/retriever-ColPali%20v1.3-green)](https://huggingface.co/vidore/colpali-v1.3-merged)
[![License MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

面向页面图像的研究型工程：先判断页面是否需要脱敏或人工复核，再根据用户问题检索证据页面，最后只把安全策略允许的页面交给外部视觉大模型回答。项目以 DocVQA 为实验数据，检索器基于 ColPali/PaliGemma，生成端接入 Doubao-Seed-2.1-pro。

> **三层模块，一条安全管线。** Sensitivity 判断页面是否含敏感信息，Retrieval 用 Global 粗排 + Late Interaction 精排找到证据页，Generation 通过 fail-closed 安全门后调用外部 VLM 生成答案。三个模块独立训练、独立评估、独立部署。

## 目录

- [当前状态](#当前状态)
- [系统边界与端到端流程](#系统边界与端到端流程)
- [关键设计决策](#关键设计决策)
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

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 数据 manifest | ✅ 已完成 | 两个任务均按 `doc_id` 划分，文档零交叉 |
| Sensitivity 两阶段训练 | ✅ 已完成 (5090 32GB) | val Recall@0.90 阈值已校准；test 结果待同步 |
| Global Retrieval | ✅ 已完成 (5090 32GB) | batch=64, symmetric InfoNCE, rank=64 rsLoRA |
| Global hard negatives | ✅ 已生成 | 每 query 4 个，35,153 条；正式 Late 前需审计 |
| Late Interaction smoke | ✅ 已完成 | 4,096 样本，证明训练链路和显存可运行 |
| Late Interaction 全量训练 | 🔄 进行中 | symmetric fast 配置；最近使用 chunked MaxSim |
| Retrieval test 与全量索引 | ⏳ 未验证 | 需 test-only index + adapter 指纹匹配 |
| 真实 Doubao 评估 | ⏳ 未执行 | 客户端已通过 mock 测试 |

仓库中 `outputs/metrics_report.csv` 和 `models/` 是早期 18 页 24 问答模拟数据的演示产物，**不能**作为正式成绩。

## 系统边界与端到端流程

| 模块 | 输入 | 输出 | 不负责什么 |
| --- | --- | --- | --- |
| Sensitivity Classification | 单张页面图像 | 页面敏感概率、二分类结果 | 不定位字段、不涂黑图片、不检索、不生成答案 |
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

正式入口是 [`scripts/answer_query.py`](scripts/answer_query.py) 和 [`scripts/evaluate_secure_rag.py`](scripts/evaluate_secure_rag.py)。`train_colpali.py`、`evaluate_generator.py`、`src/vlm_rag/generator.py` 与 `src/vlm_rag/baselines.py` 是旧单向量/拼图基线，不是生产安全链路。

## 关键设计决策

| 决策 | 说明 |
| --- | --- |
| OCR 不进入主分类器 | 正标签由 OCR 规则构建；直接输入 OCR 会导致标签泄漏 |
| 页面不压缩为单向量 | Late 阶段保留 1,024×128 image tokens，用 MaxSim 做 token 级匹配 |
| 按文档划分而非按 QA | 同一 `doc_id` 的页面/query 完整进入同一 split，零交叉 |
| 安全门 fail-closed | 缺 catalog、哈希不匹配、无脱敏副本 → 阻断，不静默降级 |
| manifest 逻辑视图 | 不复制图片；POSIX 路径，Windows 生成可随数据迁移 Linux |

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
| Vision tower | SigLIP，27 encoder blocks，hidden size 1,152 |
| 页面表示 | 1,024 image tokens |
| Language model | Gemma，18 层，hidden size 2,048 |
| ColPali token embedding | 128 维 |

### Sensitivity Classification

Sensitivity 只接收页面图像。模型通过 **forward hooks** 保留 SigLIP 的 `[0, 8, 16, 23, 27]` 层——不是 `output_hidden_states=True`（会让全部 28 层常驻显存）。随后：

```text
可学习多层加权 → regional projection → 2 层 Transformer Encoder + CLS → 单个页面敏感 logit
```

| 配置项 | 本地 12GB | 5090 32GB |
| --- | ---: | ---: |
| spatial_pool_size | 8 (64 regions) | 16 (256 regions) |
| batch_size | 2 | 16–24 |
| 阶段 1 峰值 reserved | 1.17 GB | ~8 GB |
| 阶段 2 峰值 reserved | 1.19 GB | ~12 GB |

**两阶段训练：**

| | 阶段 1 (head) | 阶段 2 (unfreeze4) |
| --- | --- | --- |
| ViT | 冻结 | 解冻最后 4 层 |
| 训练内容 | 分类头 + 层权重 + Transformer | 同上 + ViT 尾层 |
| learning_rate | head 1e-3, vision N/A | head 1e-3, vision 1e-5 |
| 损失 | BCEWithLogitsLoss(pos_weight) | 同 |
| 早停 | patience=3, 监控 val loss | 同 |

阈值只在 val 上校准（target_recall=0.90），test 使用固定阈值跑一次。

Sensitivity 只是页面级风险分类器。它不会定位"美元金额"等具体字段，也不会生成涂黑后的图片。

### Global Retrieval

Global 阶段为大 batch 粗排训练：Query 经过 Gemma 和 rank-64 rsLoRA，页面经过视觉编码、多层融合与投影，二者通过 **TokenAttentionPool**（可学习的 softmax 门控加权，而非简单 mean pool）池化到 768 维全局向量。损失同时计算 query→page 与 page→query 的**对称 InfoNCE**；batch sampler 避免同一 batch 内重复 `doc_id` 或正例页面，降低假负例风险。

该阶段用于快速候选召回和第一轮 hard-negative 挖掘，不是最终精排模型。

### Late Interaction Retrieval

Late 阶段不再把页面压缩成单个向量：Query 保留有效文本 token，Page 保留 1,024×128 的 ColPali image tokens，通过精确 MaxSim 学习局部文字、金额、表格单元格和版式区域之间的匹配。

**混合损失：**

| 损失分量 | 权重 | 作用 |
| --- | ---: | --- |
| global InfoNCE (对称) | 0.25 | 保持粗排语义空间稳定 |
| late interaction (对称) | 0.55 | token 级精确匹配 |
| hard-negative ranking | 0.20 | 拉开正例与相似错误页的距离 |

Hard negatives 来自三个渠道：每 query 4 个显式 hard negatives（Global 阶段挖掘，每 epoch 确定性轮换 1 个）、容量 256 的 detached page-token queue（不参与反向传播）、以及 in-batch negatives。

**Batch 组织（page-grouped）：** 每步选 8 个不同文档的页面，每页最多组合 4 条 query。同页 query 共享一次 page forward，消除同一 optimizer step 内的重复编码。同页对应多条 query 时使用 `multi_positive_symmetric_cross_entropy`，将同页的所有 query 视为正例。

**MaxSim 双后端：** `auto` 模式下优先使用 LIK CUDA kernel（极致吞吐）；不可用时降级为 chunked PyTorch 分块计算（结果完全一致，慢一些）。两个后端均使用按有效 query token 数归一化的 mean MaxSim，切换后端不会改变训练损失尺度。

训练输出包含逐 batch `tqdm` 进度条、`training.log`、逐 epoch `metrics.jsonl`、`epoch-NNN/`、`last/` 和 `best/`。`best/` 默认先保存 epoch-0 基线，再按固定 1,000 条 val query 的 Recall@5 + MRR 选择；新 epoch 不优于基线时不会覆盖。

### Secure Generation

安全门只允许两种字节离开进程：

1. catalog 明确判定为非敏感的原图；
2. 敏感页对应的、已批准且 source/redacted SHA-256 均匹配的脱敏副本。

页面缺少 catalog、检索元数据不一致、分类失败、文件缺失、哈希变化或敏感原图没有批准副本时均默认阻断。多向量索引还会校验 adapter 与基础模型元数据指纹，避免错配权重和索引。

Doubao 客户端按页面请求结构化 JSON，验证 `relevant/answer/evidence_quote/confidence`，实现响应缓存、429/5xx 重试、指数退避和显式错误记录，再对有效页面答案进行融合。

## 代码结构

```text
configs/                             # 任务配置、显存 profile、安全链路
scripts/
├── build_*manifest.py               # 数据逻辑结构（不复制图片）
├── train_sensitivity.py             # Sensitivity 两阶段训练
├── evaluate_sensitivity.py          # val 校准 + 固定阈值 test
├── classify_pages.py                # 全量批量推理
├── build_sensitivity_catalog.py     # 持久化预测目录
├── build_redaction_manifest.py      # 脱敏副本批准 manifest
├── train_retrieval_global.py        # Global 粗排训练
├── mine_global_hard_negatives.py    # Hard negative 挖掘
├── train_retrieval_late.py          # Late Interaction 精排训练
├── build_multivector_index.py       # 多向量分片索引构建
├── evaluate_retrieval_multivector.py# 检索评估 (MRR/Recall)
├── profile_training_memory.py       # 5090 显存标定
├── answer_query.py                  # 正式单次安全问答入口
├── evaluate_secure_rag.py           # 正式安全链路评估入口
├── validate_project.py              # 结构与依赖验收
├── smoke_*.py                       # 本地 12GB 快速冒烟测试
└── verify_maxsim_backends.py        # MaxSim 后端一致性验证

src/vlm_rag/
├── sensitivity/           # 页面敏感度分类
│   ├── model.py           → SensitivityClassifier (hooks + 区域池化 + Transformer)
│   ├── training.py        → 两阶段训练 + 早停 + pos_weight
│   ├── inference.py       → 批量推理 + 阈值校准
│   ├── schema.py          → SensitivityRecord (JSONL manifest)
│   ├── catalog.py         → 持久化预测目录 (概率/阈值/SHA-256)
│   └── split.py           → 按 doc_id 分层划分
├── retrieval/             # 多向量 ColPali 检索
│   ├── model.py           → LateInteractionRetriever (LoRA r=64/rsLoRA + TokenAttentionPool)
│   ├── losses.py          → 对称 InfoNCE + 混合三损失 + memory queue
│   ├── maxsim.py          → MaxSim (LIK / chunked 双后端，分块精确计算)
│   ├── dataset.py         → PageGroupedBatchSampler (共享 page forward)
│   ├── index.py           → 分片多向量持久化索引 (指纹校验)
│   └── training.py        → Global + Late 双训练配置
├── pipeline/              # 安全管线
│   ├── engine.py          → SecureRAGEngine (粗排→重排→门控→生成→审计)
│   ├── policy.py          → SensitivityPolicy (fail-closed 安全门)
│   ├── contracts.py       → PipelineAnswer / RetrievalHit / PageAccessDecision
│   ├── provenance.py      → SHA-256 文件/适配器/基础模型元数据指纹
│   └── audit.py           → JSONL 审计日志
└── generation/            # 安全生成
    ├── client.py          → DoubaoVisionClient (fail-explicit 错误 + 重试 + 限速)
    ├── schema.py          → GenerationAnswer / GenerationError (6 种类型)
    ├── fusion.py          → retrieval_score × confidence 加权融合
    └── cache.py           → 线程安全 JSON 响应缓存

tests/                      # 数据、训练、推理、多向量与安全策略单元测试
```

更长的部署顺序见 [`SERVER_GUIDE.md`](SERVER_GUIDE.md)，方法背景见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，研究性说明见 [`docs/technical_report.md`](docs/technical_report.md)。技术报告中的"目标效果参考"不是本仓库已经达到的实验结果。

## 快速开始

### 1. 环境与路径

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

### 2. 安装依赖

```bash
# PyTorch (CUDA 12.8)
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt

# Late Interaction 建议安装 LIK kernel
python -m pip install --no-cache-dir "colpali-engine[lik]==0.3.17"
```

**一键结构验收（不需要 GPU）：**

```bash
bash setup.sh --no-install
python scripts/validate_project.py \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --skip-gpu-smoke --dependency-policy compatible
```

### 3. 构建 Manifest + 运行测试

```bash
python scripts/build_sensitivity_manifest.py --data-root "$DOCVQA_DATA_ROOT"
python scripts/build_retrieval_manifest.py   --data-root "$DOCVQA_DATA_ROOT"
python -m pytest tests/ -x -q
```

### 4. 本地烟雾测试

```bash
# Sensitivity（4080 12GB 可跑）
python scripts/smoke_sensitivity.py --data-root ../data --model ../checkpoint --batch-size 2
```

## 训练与评估

完整训练流程 7 步，详见 [`SERVER_GUIDE.md`](SERVER_GUIDE.md)。下面是关键命令概要。

### 1. Sensitivity 两阶段训练

```bash
# 阶段 1：冻结 ViT，训练分类头
python scripts/train_sensitivity.py \
  --config configs/sensitivity_head_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --output-dir "$PROJECT_ROOT/outputs/sensitivity_head"

# 阶段 2：解冻 ViT 最后 4 层微调
python scripts/train_sensitivity.py \
  --config configs/sensitivity_unfreeze4_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --init-checkpoint "$PROJECT_ROOT/outputs/sensitivity_head/best.pt" \
  --output-dir "$PROJECT_ROOT/outputs/sensitivity_unfreeze4"
```

阶段切换用 `--init-checkpoint`；同一阶段中断后用 `--resume last.pt`。

### 2. Val 校准与 Test 评估

```bash
# val 校准阈值（target_recall=0.90）
python scripts/evaluate_sensitivity.py --mode calibrate \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/val.jsonl" \
  --checkpoint outputs/sensitivity_unfreeze4/best.pt \
  --model "$COLPALI_MODEL_PATH" --data-root "$DOCVQA_DATA_ROOT" \
  --target-recall 0.90 --output-dir outputs/sensitivity_eval

# test 固定阈值评估（禁止利用 test 标签重新选阈值）
python scripts/evaluate_sensitivity.py --mode evaluate \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/test.jsonl" \
  --calibration outputs/sensitivity_eval/calibration.json \
  --checkpoint outputs/sensitivity_unfreeze4/best.pt \
  --model "$COLPALI_MODEL_PATH" --data-root "$DOCVQA_DATA_ROOT"
```

### 3. Sensitivity Catalog

```bash
python scripts/classify_pages.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/all.jsonl" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --checkpoint outputs/sensitivity_unfreeze4/best.pt \
  --model "$COLPALI_MODEL_PATH" --batch-size 16 --format both

python scripts/build_sensitivity_catalog.py \
  --data-root "$DOCVQA_DATA_ROOT" --predictions predictions.jsonl \
  --calibration outputs/sensitivity_eval/calibration.json \
  --output sensitivity_catalog.jsonl
```

### 4. Global Retrieval

先 profile 显存再跑正式训练。服务器已完成 batch=64、accumulation=2、10 epochs 的稳定运行。

```bash
python scripts/profile_training_memory.py --task global \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH"

python -u scripts/train_retrieval_global.py \
  --config configs/retrieval_global_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --output-dir outputs/retrieval_global \
  --epochs 10 --num-workers 8
```

### 5. Hard Negative 挖掘

```bash
python -u scripts/mine_global_hard_negatives.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --checkpoint outputs/retrieval_global \
  --base-model "$COLPALI_MODEL_PATH" \
  --output "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --negatives-per-query 4 --candidate-top-k 512
```

目标：35,153 行、每行 4 个负样本、无重复 query、无正负碰撞。

### 6. Late Interaction

```bash
# smoke — 确认训练链路可反向
python -u scripts/train_retrieval_late.py \
  --config configs/retrieval_late_symmetric_fast_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --hard-negatives .../hard_negatives.jsonl \
  --init-adapter outputs/retrieval_global/best \
  --pages-per-batch 4 --queries-per-page 4 \
  --gradient-accumulation-steps 8 --epochs 1 \
  --max-train-samples 256 --num-workers 8

# 全量训练
python -u scripts/train_retrieval_late.py \
  --config configs/retrieval_late_symmetric_fast_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --hard-negatives .../hard_negatives.jsonl \
  --init-adapter outputs/retrieval_global/best \
  --pages-per-batch 8 --queries-per-page 4 \
  --gradient-accumulation-steps 4 --epochs 3 --num-workers 8
```

### 7. Test 索引与评估

```bash
# 构建 test-only 多向量索引
python scripts/build_multivector_index.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --model "$COLPALI_MODEL_PATH" --adapter outputs/retrieval_late/best \
  --output-dir indexes/retrieval_test --batch-size 4 --pages-per-shard 128

# 评估 MRR/Recall
python scripts/evaluate_retrieval_multivector.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --index-dir indexes/retrieval_test \
  --model "$COLPALI_MODEL_PATH" --adapter outputs/retrieval_late/best \
  --coarse-top-k 128 --maxsim-backend lik --maxsim-normalization mean
```

## 安全问答

### 安全门控

Sensitivity 不生成脱敏图片。如果已有独立工具或人工产出的副本，先准备映射文件再显式批准：

```bash
python scripts/build_redaction_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --catalog sensitivity_catalog.jsonl \
  --mappings redaction_mappings.jsonl --approve
```

没有 redaction manifest 时，非敏感原图仍可回答；敏感原图会被阻断，系统继续尝试排名更低的安全证据。如果没有任何安全页，返回 `blocked_sensitive_evidence`。

### Doubao API 调用

```bash
export ARK_API_KEY="your-key"
export DOUBAO_MODEL=doubao-seed-2-1-pro

# 单次问答（--allow-real-api 是真实外发的显式开关）
python scripts/answer_query.py \
  --query "What was the total amount?" \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --adapter outputs/retrieval_late/best --index-dir indexes/retrieval_all \
  --sensitivity-catalog sensitivity_catalog.jsonl --allow-real-api

# 小规模端到端评估
python scripts/evaluate_secure_rag.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --adapter outputs/retrieval_late/best --index-dir indexes/retrieval_all \
  --sensitivity-catalog sensitivity_catalog.jsonl \
  --max-queries 10 --allow-real-api
```

安全验收要求 `sensitive_original_exposure = 0` 且 `missing_catalog_exposure = 0`。

## 实验记录与验证边界

### Sensitivity 验证集校准（来自服务器 val 结果）

| 指标 | 数值 |
| --- | ---: |
| ROC-AUC | 0.91366 |
| PR-AUC | 0.82675 |
| target-recall threshold | 0.291225 |
| Recall | 0.90056 |
| Precision | 0.55046 |
| F1 | 0.68327 |

### 服务器训练运行记录

| 阶段 | 设置 | 结果 |
| --- | --- | --- |
| Global | batch 64, accum 2, 10 epochs | train loss 0.2985; peak reserved 23.32 GB |
| Late smoke | batch 8, accum 12, 4,096/512, 1 epoch | train loss 1.0932; val loss 0.8857; peak reserved 13.82 GB |
| Late full | chunked 后端, 约 9h/epoch | 当前未确认完成，不能给出 final best/test 指标 |

在 `test_metrics.json`、test-only index 和 adapter 被同步并通过指纹校验前，不声称取得正式 Retrieval test 成绩。

## 性能与故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `No module named vlm_rag` | 仓库采用 `src/` layout；使用 `PYTHONPATH=src python ...` |
| LIK 检查为 `False` | 安装 `colpali-engine[lik]` 后重做 smoke |
| Late 一个 epoch 很慢 | 确认 LIK、page-grouped batching、单轮换 hard negative 均已生效 |
| 显存低但训练不快 | 显存 ≠ 吞吐；检查 GPU Util、DataLoader、MaxSim backend |
| hard negatives 不足 4 个 | 同文档过滤后候选不足；提高 `candidate-top-k` 并审计 |
| `--resume` 报缺失 `training_state.pt` | 旧 adapter 用 `--init-adapter`；新版 `last/` 支持严格 `--resume` |
| index provenance mismatch | adapter/基础模型/脚本已变化；用当前 adapter 重建索引 |
| 豆包 401/403 | 检查 Key、模型名、Endpoint ID；不要把 Key 写进文件 |
| 12GB Late backward OOM | 本地 12GB 只适合前向/smoke；全量 Late 用 32GB GPU |

## 测试、限制与后续工作

### 验收命令

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/validate_project.py \
  --data-root "$DOCVQA_DATA_ROOT" --model "$COLPALI_MODEL_PATH" \
  --skip-gpu-smoke --dependency-policy compatible
git diff --check
```

### 当前限制

- Sensitivity 是页面级风险筛查，不是字段检测或自动涂黑系统。
- 完整 Retrieval test、全量索引和真实 Doubao 评估尚未在本地归档。
- 新版 Late 的 `last/training_state.pt` 支持严格 resume；旧 adapter 只能 `--init-adapter`。
- 早期 OCR-RAG、SigLIP、单向量 ColPali 和图像拼接结果只用于研究基线。

### 后续优先级

1. 验证 LIK 训练反向，记录加速比。
2. 在 5090 上分别 profile gradient checkpointing 开/关。
3. 将服务器 checkpoint/test JSON 归档到可审计实验目录。
4. 完成 test-only 多向量索引评估。
5. 在严格门控下完成真实 Doubao 请求，记录费用、延迟、EM/Accuracy。
6. 可选地评估冻结视觉前缀缓存。

## 安全说明

- 不要提交 `data/`、checkpoint、adapter、index、缓存、`.env` 或任何 API Key。
- `--allow-real-api` 是真实外部传输的显式开关；测试和 smoke 不应携带它。
- catalog 缺失、推理错误和敏感原图无副本均必须 fail closed。
- 如果页面内容或脱敏副本发生变化，应重新分类、重建 catalog/redaction manifest，并重新生成索引指纹。
- 本项目当前是研究工程，不应在没有权限控制、人工复核和组织合规评估的情况下直接处理生产敏感文档。
