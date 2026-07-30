# 文档敏感性筛查与多向量 VLM-RAG

本仓库包含两个边界清晰、可以串联但不能混用数据语义的研究任务：

1. **Sensitivity classification**：输入普通文档页面图像，判断页面是否需要进入脱敏/人工复核流程。
2. **Retrieval / RAG**：输入问题，在完整 DocVQA 页面中检索相关页面，再将候选页交给视觉问答模块。

`data/desensitized` 是从完整 DocVQA 中筛选出的分类正标签集合，不是独立语料，也不是 RAG 数据根。分类器只做页面级风险筛查，不会涂黑、删除或覆盖原图。

## 1. 数据语义

物理目录无需重排，也不需要复制图片：

```text
data/
├── docvqa_images/                 # 完整页面：12,767
├── ocr/                           # 完整 OCR
├── docvqa_extracted/              # 完整 QA JSON
└── desensitized/
    ├── docvqa_images/             # 正样本页面：3,538
    ├── ocr/
    └── docvqa_extracted/
```

标签定义：

```text
完整页面位于 desensitized/docvqa_images 中 → is_sensitive = 1
否则                                      → is_sensitive = 0
```

当前完整数据核验结果为：

| 项目 | 数量 |
| --- | ---: |
| 页面 | 12,767 |
| 正样本 | 3,538 |
| 负样本 | 9,229 |
| 文档 | 6,071 |
| QA 查询 | 50,000 |

分类和检索均复用同一套按 `doc_id` 生成的 70/15/15 划分。官方 QA split 中相同文档存在交叉，因此不能直接作为本项目的分类/检索实验划分。

## 2. 架构

### 2.1 需脱敏页面分类

```text
页面图像 448×448
      │
      ▼
SigLIP ViT（选取 0/8/16/23/27 层）
      │
      ├── 每层 32×32 patch 网格
      │
      ▼
自适应区域池化
  本地配置：8×8 = 64 tokens
  5090 配置：16×16 = 256 tokens
      │
      ▼
可学习多层融合 → 区域投影 → 2 层 Transformer + CLS
      │
      ▼
页面敏感概率
```

旧实现会在分类头之前把整个页面压成一个向量，容易丢失金额、表格单元格和局部字段。当前实现保留区域 token，只有在区域间完成注意力交互后才用 CLS 做二分类。

OCR 不进入主分类器。当前正标签本身由 OCR 规则辅助构建，直接输入 OCR 会使模型倾向于复现标签规则，形成标签泄漏。OCR 仅用于数据审计和未来独立 baseline。

5090 训练分两阶段：

- 阶段 1：冻结完整 ViT，训练区域 Transformer、分类头和层权重；
- 阶段 2：从阶段 1 权重初始化，只解冻 ViT 最后 4 层，小学习率微调。

最佳 checkpoint 优先满足目标 Recall（默认 0.90），再比较 Precision、PR-AUC 和 F1，而不是仅追求 Accuracy。

### 2.2 多向量检索

```text
阶段 A：大 batch 全局预训练
Query → Gemma + LoRA rank 64 ─┐
                              ├→ 768-d 全局向量 → 对称 InfoNCE
Page  → frozen SigLIP ────────┘   batch 96/128 候选
                    │
                    ▼
       全局模型挖掘 hard negatives
                    │
                    ▼
阶段 B：原生 ColPali late interaction
Query tokens [Lq,128] × Page tokens [1024,128]
                    │
                    ▼
 exact MaxSim + global loss + hard-negative loss + memory queue
```

正式 late-interaction 路径不再把页面压成单个 768 维向量。每页保留 1,024 个 ColPali image token，查询保留有效文本 token，并使用精确 MaxSim。全局向量仅负责粗排和辅助损失。

MaxSim 优先使用 `late-interaction-kernels`（LIK）；未安装时可使用精确的分块 PyTorch 后端。分块后端不会构造完整的 `B×B×Lq×Ld` 张量，因此显存不会随语料总页数增长。

持久化索引按 shard 保存页面 token，搜索时先用 CPU 中的全局向量粗排，再仅把候选 token 分批送入 GPU 重排。

## 3. 环境

已验证的本地环境位于 `E:\envs\vlm`。关键版本见 `requirements.txt`。Linux/RTX 5090 建议：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install "colpali-engine[lik]==0.3.17"
```

如果 LIK 暂不支持服务器上的具体 CUDA/PyTorch 组合，将配置中的 `maxsim_backend` 改为 `chunked`，结果仍是精确 MaxSim，只是速度较慢。

缓存、环境和临时目录应放在数据盘：

```bash
export VLM_CACHE_ROOT=/data/cache/vlm
export HF_HOME=$VLM_CACHE_ROOT/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$VLM_CACHE_ROOT/torch
export PIP_CACHE_DIR=$VLM_CACHE_ROOT/pip
export TMPDIR=$VLM_CACHE_ROOT/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 4. 构建 manifest

```bash
python scripts/build_sensitivity_manifest.py --data-root "$DOCVQA_DATA_ROOT"

python scripts/build_retrieval_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT"
```

输出：

```text
data/manifests/
├── sensitivity/
│   ├── all.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── summary.json
│   └── anomalies.json
└── retrieval/
    ├── all.jsonl
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    └── summary.json
```

manifest 内路径是相对于 `data-root` 的 POSIX 路径，可在 Windows 生成后原样上传 Linux。构建过程不移动、不复制、不重命名数据。

本地实际生成的 retrieval 划分：

| split | 查询 | 页面 | 文档 |
| --- | ---: | ---: | ---: |
| train | 35,153 | 8,937 | 4,256 |
| val | 7,291 | 1,925 | 895 |
| test | 7,556 | 1,905 | 920 |

三个 split 的 `doc_id` 交集均为 0。

## 5. 本地 12GB 验证

```powershell
E:\envs\vlm\python.exe -m unittest discover -s tests -v

E:\envs\vlm\python.exe scripts\smoke_sensitivity.py `
  --data-root ..\data `
  --model ..\checkpoint `
  --batch-size 2 `
  --samples-per-class 2
```

本机 RTX 4080 Laptop 实测：

| 检查 | 峰值 reserved |
| --- | ---: |
| 区域分类头，冻结 ViT，batch 2，真实前向/反向 | 1.17 GB |
| 分类器解冻最后 4 层，batch 1，真实前向/反向 | 1.19 GB |
| 原生 ColPali 多向量，rank 64，batch 1，query/page 前向 | 10.81 GB |

最后一项已接近 12GB 上限，因此本机不强行执行 late-interaction 反向；这不是 5090 正式 batch 的测量结果。

## 6. RTX 5090 显存标定

配置中的 batch 是安全起点，不是假定的实测结论。租到 5090 后先在独立进程中运行真实 optimizer step：

```bash
python scripts/profile_training_memory.py \
  --task sensitivity-head \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --work-dir "$PROFILE_ROOT"

python scripts/profile_training_memory.py \
  --task sensitivity-unfreeze4 \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --work-dir "$PROFILE_ROOT"

python scripts/profile_training_memory.py \
  --task global \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --work-dir "$PROFILE_ROOT"

python scripts/profile_training_memory.py \
  --task late \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --work-dir "$PROFILE_ROOT"
```

脚本逐候选 batch 启动新进程，记录 `max_memory_allocated/reserved`，默认硬上限为 28GB，并清理每次 profile 产生的临时 checkpoint。建议目标是 reserved 26–28GB；不要以 `nvidia-smi` 的瞬时占用代替 PyTorch 峰值。

初始候选：

| 任务 | 候选 physical batch |
| --- | --- |
| Sensitivity head，256 区域 token | 16 / 24 / 32 / 40 |
| Sensitivity 解冻最后 4 层 | 8 / 12 / 16 / 24 |
| Retrieval global | 64 / 96 / 128 |
| Retrieval late interaction | 2 / 4 / 6 / 8 |

## 7. Sensitivity 正式训练

阶段 1：

```bash
python scripts/train_sensitivity.py \
  --config configs/sensitivity_head_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --output-dir "$SENSITIVITY_HEAD_DIR"
```

阶段 2 只继承模型权重，不继承阶段 1 的 optimizer 状态：

```bash
python scripts/train_sensitivity.py \
  --config configs/sensitivity_unfreeze4_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --init-checkpoint "$SENSITIVITY_HEAD_DIR/best.pt" \
  --output-dir "$SENSITIVITY_FINETUNE_DIR"
```

同一阶段中断续训才使用：

```bash
python scripts/train_sensitivity.py ... \
  --resume "$SENSITIVITY_FINETUNE_DIR/last.pt"
```

评估与阈值校准：

```bash
python scripts/evaluate_sensitivity.py \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --target-recall 0.90 \
  --output-dir "$SENSITIVITY_EVAL_DIR"
```

批量分类：

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
```

损坏图片会写入 `errors.jsonl`；其他页面继续处理，但任务最终非零退出，避免静默漏页。

## 8. Retrieval 正式训练

### 8.1 全局阶段

```bash
python scripts/train_retrieval_global.py \
  --config configs/retrieval_global_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --output-dir "$RETRIEVAL_GLOBAL_DIR"
```

同一 batch 内不允许重复 `doc_id` 或正例页面，避免同文档页面成为假负例。损失同时计算 query→page 和 page→query。

### 8.2 挖掘 hard negatives

```bash
python scripts/mine_global_hard_negatives.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/train.jsonl" \
  --checkpoint "$RETRIEVAL_GLOBAL_DIR" \
  --base-model "$COLPALI_MODEL_PATH" \
  --output "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --page-batch-size 64 \
  --query-batch-size 128
```

负例会排除正页面以及相同 `doc_id` 的页面。

### 8.3 原生多向量阶段

```bash
python scripts/train_retrieval_late.py \
  --config configs/retrieval_late_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --hard-negatives "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --init-global "$RETRIEVAL_GLOBAL_DIR/lora" \
  --output-dir "$RETRIEVAL_LATE_DIR"
```

该命令继承全局阶段的 rank-64 LoRA，不重复套 LoRA；late 阶段保存的 `best/` 和 `last/` 只包含 LoRA、ColPali token projection、粗排池化头及元数据。

### 8.4 建索引和评估

```bash
python scripts/build_multivector_index.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/all.jsonl" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --output-dir "$RETRIEVAL_INDEX_DIR" \
  --batch-size 4 \
  --pages-per-shard 128

python scripts/evaluate_retrieval_multivector.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --index-dir "$RETRIEVAL_INDEX_DIR" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --coarse-top-k 128 \
  --maxsim-backend lik \
  --output "$RETRIEVAL_LATE_DIR/test_metrics.json"
```

可在完成一轮 late 训练后使用 `mine_hard_negatives.py` 基于精确 MaxSim 刷新更难的负例，再训练第二轮。

## 9. 测试与验收

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/test_integration.py
python scripts/test_integration_generator.py
python -m pip check
git diff --check
```

项目级结构与真实 GPU 前向：

```bash
python scripts/validate_project.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH"
```

## 10. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| manifest 数量不对 | `--data-root` 必须指向同时包含四个顶层部分的完整数据根，不能指向 `desensitized` |
| split 泄漏 | 重新运行两个 manifest 构建脚本；不要直接沿用官方 QA split |
| 12GB 上 late backward OOM | 正常；本机只验证 batch-1 前向，正式训练用 5090 profile 后的 batch |
| 5090 没用满 | 依次 profile 更大 physical batch；不要先盲目增大梯度累积 |
| LIK 加载失败 | 安装 `colpali-engine[lik]`，或切到精确 `chunked` 后端 |
| hard negatives 数量少 | 候选被相同文档过滤；提高 `candidate-top-k` |
| 阶段 2 checkpoint mismatch | 阶段切换用 `--init-checkpoint`，同阶段续训才用 `--resume` |
| 指标异常好 | 确认没有输入 OCR、没有文档交叉，并区分 smoke/tiny 与正式 test 指标 |

## 11. 当前验证边界

已在本地完成数据全量 manifest 构建、单元测试、Sensitivity 两种冻结状态的真实前向/反向、原生 ColPali rank-64 多向量前向和精确 MaxSim。尚未完成的项目是 RTX 5090 上的 batch 标定、全量训练、全量索引和正式 test 指标；这些结果不能在实际运行前预填。
