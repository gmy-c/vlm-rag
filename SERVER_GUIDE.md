# RTX 5090 服务器部署与训练顺序

本指南假定 32GB RTX 5090。所有 27–28GB batch 都必须在目标机器实测后确定，配置文件中的数值只是安全起点。

## 1. 上传内容

需要上传：

```text
<project-root>/       # 代码、configs、scripts、src、tests
<data-root>/          # 完整 data 四个顶层部分
<model-root>/         # 本地 ColPali/PaliGemma checkpoint 的全部分片和配置
```

无需上传：

```text
.git/
outputs/
__pycache__/
.pytest_cache/
本地 smoke checkpoint
Hugging Face / pip / torch cache
```

数据物理结构不需要修改。`desensitized` 必须随完整数据上传，但它只提供分类正标签。

## 2. 统一变量

```bash
export PROJECT_ROOT=/data/project/aiproject
export DOCVQA_DATA_ROOT=/data/datasets/docvqa
export COLPALI_MODEL_PATH=/data/models/colpali
export VLM_ENV_ROOT=/data/envs/vlm
export VLM_CACHE_ROOT=/data/cache/vlm
export PROFILE_ROOT=/data/outputs/memory_profiles

export SENSITIVITY_HEAD_DIR=/data/outputs/sensitivity_head
export SENSITIVITY_FINETUNE_DIR=/data/outputs/sensitivity_unfreeze4
export SENSITIVITY_EVAL_DIR=/data/outputs/sensitivity_eval
export SENSITIVITY_PRED_DIR=/data/outputs/sensitivity_predictions
export SENSITIVITY_CATALOG=$SENSITIVITY_PRED_DIR/sensitivity_catalog.jsonl

export RETRIEVAL_GLOBAL_DIR=/data/outputs/retrieval_global
export RETRIEVAL_LATE_DIR=/data/outputs/retrieval_late_symmetric_fast
export RETRIEVAL_TEST_INDEX_DIR=/data/outputs/retrieval_test_index
export RETRIEVAL_INDEX_DIR=/data/outputs/retrieval_index
export SECURE_RAG_DIR=/data/outputs/secure_rag

export HF_HOME=$VLM_CACHE_ROOT/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$VLM_CACHE_ROOT/torch
export PIP_CACHE_DIR=$VLM_CACHE_ROOT/pip
export TMPDIR=$VLM_CACHE_ROOT/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$VLM_ENV_ROOT" "$VLM_CACHE_ROOT" "$PROFILE_ROOT" \
  "$SENSITIVITY_HEAD_DIR" "$SENSITIVITY_FINETUNE_DIR" \
  "$SENSITIVITY_EVAL_DIR" "$SENSITIVITY_PRED_DIR" \
  "$RETRIEVAL_GLOBAL_DIR" "$RETRIEVAL_LATE_DIR" \
  "$RETRIEVAL_TEST_INDEX_DIR" "$RETRIEVAL_INDEX_DIR" \
  "$SECURE_RAG_DIR"

cd "$PROJECT_ROOT"
```

## 3. 环境

```bash
python3 -m venv "$VLM_ENV_ROOT"
export VLM_PYTHON="$VLM_ENV_ROOT/bin/python"

"$VLM_PYTHON" -m pip install --upgrade pip
"$VLM_PYTHON" -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$VLM_PYTHON" -m pip install -r requirements.txt
"$VLM_PYTHON" -m pip install "colpali-engine[lik]==0.3.17"
```

`setup.sh` 不下载 checkpoint。也可以使用：

```bash
export VLM_PYTHON
export DOCVQA_DATA_ROOT
export COLPALI_MODEL_PATH
export VLM_CACHE_ROOT
bash setup.sh
```

## 4. 数据与结构验收

```bash
"$VLM_PYTHON" scripts/build_sensitivity_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT"

"$VLM_PYTHON" scripts/build_retrieval_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT"

"$VLM_PYTHON" scripts/validate_project.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --skip-gpu-smoke

"$VLM_PYTHON" -m unittest discover -s tests -v
"$VLM_PYTHON" -m compileall -q src scripts tests
git diff --check
```

预期：

```text
完整页面 12,767
敏感正例  3,538
负例      9,229
文档      6,071
查询     50,000
train / val / test doc_id 交集均为 0
```

## 5. 先标定显存

分别执行：

```bash
for task in sensitivity-head sensitivity-unfreeze4 global late
do
  "$VLM_PYTHON" scripts/profile_training_memory.py \
    --task "$task" \
    --data-root "$DOCVQA_DATA_ROOT" \
    --model "$COLPALI_MODEL_PATH" \
    --work-dir "$PROFILE_ROOT" \
    --max-reserved-gb 28
done
```

查看：

```bash
find "$PROFILE_ROOT" -name '*memory-profile.json' -maxdepth 1 -print
```

选择原则：

- `peak_reserved_gb <= 28`；
- 目标区间 26–28GB；
- 选中的 batch 再连续跑至少 200 step 观察是否稳定；
- OOM 时降低 physical batch，用梯度累积保持 effective batch；
- 不把 32GB 全部作为 PyTorch 上限，需给 CUDA context、kernel workspace 和波动留余量。

把 profile 推荐值通过 CLI 覆盖配置，例如：

```bash
--batch-size 32
--micro-batch-size 4
```

## 6. Sensitivity 两阶段

阶段 1：

```bash
"$VLM_PYTHON" scripts/train_sensitivity.py \
  --config configs/sensitivity_head_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --output-dir "$SENSITIVITY_HEAD_DIR"
```

阶段 2：

```bash
"$VLM_PYTHON" scripts/train_sensitivity.py \
  --config configs/sensitivity_unfreeze4_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --init-checkpoint "$SENSITIVITY_HEAD_DIR/best.pt" \
  --output-dir "$SENSITIVITY_FINETUNE_DIR"
```

评估和推理：

```bash
"$VLM_PYTHON" scripts/evaluate_sensitivity.py \
  --mode calibrate \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/val.jsonl" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --target-recall 0.90 \
  --output-dir "$SENSITIVITY_EVAL_DIR"

"$VLM_PYTHON" scripts/evaluate_sensitivity.py \
  --mode evaluate \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/test.jsonl" \
  --calibration "$SENSITIVITY_EVAL_DIR/calibration.json" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --output-dir "$SENSITIVITY_EVAL_DIR/test"

"$VLM_PYTHON" scripts/classify_pages.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/all.jsonl" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --model "$COLPALI_MODEL_PATH" \
  --calibration "$SENSITIVITY_EVAL_DIR/calibration.json" \
  --batch-size 16 \
  --format both \
  --output-dir "$SENSITIVITY_PRED_DIR"

"$VLM_PYTHON" scripts/build_sensitivity_catalog.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/sensitivity/all.jsonl" \
  --predictions "$SENSITIVITY_PRED_DIR/predictions.jsonl" \
  --errors "$SENSITIVITY_PRED_DIR/errors.jsonl" \
  --checkpoint "$SENSITIVITY_FINETUNE_DIR/best.pt" \
  --calibration "$SENSITIVITY_EVAL_DIR/calibration.json" \
  --output "$SENSITIVITY_CATALOG"
```

分类预测只是风险筛查结果。catalog 是外发门控的唯一输入；损坏图、缺预测和敏感原图均默认阻断。

## 7. Retrieval 全局阶段

```bash
"$VLM_PYTHON" scripts/train_retrieval_global.py \
  --config configs/retrieval_global_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --output-dir "$RETRIEVAL_GLOBAL_DIR"
```

不要把 `--data-root` 改为 `desensitized`。

## 8. Hard negatives

```bash
"$VLM_PYTHON" scripts/mine_global_hard_negatives.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/train.jsonl" \
  --checkpoint "$RETRIEVAL_GLOBAL_DIR" \
  --base-model "$COLPALI_MODEL_PATH" \
  --output "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --page-batch-size 64 \
  --query-batch-size 128 \
  --candidate-top-k 64 \
  --negatives-per-query 4
```

## 9. Retrieval 多向量阶段

先验证本机 LIK 与 chunked 在 mean/sum 两种定义下的分数和反向梯度一致：

```bash
"$VLM_PYTHON" scripts/verify_maxsim_backends.py
```

必须输出 `"status": "passed"`；失败时不要开始正式训练。

先用真实 forward/backward 测试 4/6/8/10 个页面组的显存：

```bash
"$VLM_PYTHON" scripts/profile_training_memory.py \
  --task late-symmetric \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --work-dir "$PROJECT_ROOT/outputs/memory_profiles" \
  --max-reserved-gb 28 \
  --num-workers 8 \
  --candidates 4 6 8 10
```

正式配置默认使用 8 个页面组。若 profile 中 8 超过 28GB，改用 `--pages-per-batch 6`；若 10 稳定，也不要在没有端到端吞吐对比时只因显存尚有余量就盲目增大。

```bash
"$VLM_PYTHON" scripts/train_retrieval_late.py \
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
  2>&1 | tee logs/retrieval_late_symmetric_fast.log
```

终端会显示每个 epoch 的 batch 进度、实时 loss、平均 loss、学习率、已处理 query 和 GPU reserved 显存。详细记录位于输出目录的 `training.log` 和 `metrics.jsonl`。本配置要求 LIK；如果 LIK 不可用，先修复环境，不要在正式训练中静默切换后端。

首次运行建议先用 4,096 条 query 验证：

```bash
"$VLM_PYTHON" scripts/train_retrieval_late.py \
  --config configs/retrieval_late_symmetric_fast_5090.yaml \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --hard-negatives "$DOCVQA_DATA_ROOT/manifests/retrieval/hard_negatives.jsonl" \
  --init-adapter "$PROJECT_ROOT/outputs/retrieval_late/best" \
  --output-dir "$PROJECT_ROOT/outputs/retrieval_late_symmetric_smoke" \
  --pages-per-batch 4 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --max-train-samples 4096 \
  --max-val-samples 512
```

## 10. 建 test 索引、评估和生产索引

正式检索指标先只在 test 页面语料上计算：

```bash
"$VLM_PYTHON" scripts/build_multivector_index.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --output-dir "$RETRIEVAL_TEST_INDEX_DIR" \
  --batch-size 4 \
  --pages-per-shard 128

"$VLM_PYTHON" scripts/evaluate_retrieval_multivector.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --index-dir "$RETRIEVAL_TEST_INDEX_DIR" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --coarse-top-k 128 \
  --maxsim-backend lik \
  --maxsim-normalization mean \
  --output "$RETRIEVAL_LATE_DIR/test_metrics.json"
```

通过后再为实际问答建立全量生产索引：

```bash
"$VLM_PYTHON" scripts/build_multivector_index.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/all.jsonl" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --output-dir "$RETRIEVAL_INDEX_DIR" \
  --batch-size 4 \
  --pages-per-shard 128
```

索引会保存 adapter、基础模型元数据和 manifest 指纹。问答时不匹配会直接报错，不能混用索引与权重。

## 11. 安全问答与 Doubao-Seed-2.1-pro

不要把 Key 写进项目文件。用户曾在会话中贴出的 Key 已视为泄露，应在控制台吊销并创建新的最小权限 Key：

```bash
read -s -p "ARK_API_KEY: " ARK_API_KEY; echo
export ARK_API_KEY
export DOUBAO_MODEL='doubao-seed-2-1-pro'
# 如果控制台要求使用接入点：export DOUBAO_MODEL='ep-xxxxxxxx'
```

没有 `redaction_manifest.jsonl` 时，敏感页会被阻断，非敏感页仍可进入回答。真实调用必须显式确认：

```bash
"$VLM_PYTHON" scripts/answer_query.py \
  --query "What was the total amount?" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --index-dir "$RETRIEVAL_INDEX_DIR" \
  --sensitivity-catalog "$SENSITIVITY_CATALOG" \
  --output "$SECURE_RAG_DIR/answers.jsonl" \
  --audit "$SECURE_RAG_DIR/audit.jsonl" \
  --allow-real-api
```

如果已人工或通过独立工具生成脱敏副本，把副本放到 data-root 内，准备 `page_id/redacted_path` JSONL，并执行：

```bash
"$VLM_PYTHON" scripts/build_redaction_manifest.py \
  --data-root "$DOCVQA_DATA_ROOT" \
  --catalog "$SENSITIVITY_CATALOG" \
  --mappings redaction_mappings.jsonl \
  --output "$SENSITIVITY_PRED_DIR/redaction_manifest.jsonl" \
  --approve
```

随后在 `answer_query.py` 增加：

```bash
--redaction-manifest "$SENSITIVITY_PRED_DIR/redaction_manifest.jsonl"
```

先用 1–10 条查询做实际 API 验收，确认费用、延迟、模型服务开通状态和返回格式：

```bash
"$VLM_PYTHON" scripts/evaluate_secure_rag.py \
  --manifest "$DOCVQA_DATA_ROOT/manifests/retrieval/test.jsonl" \
  --data-root "$DOCVQA_DATA_ROOT" \
  --model "$COLPALI_MODEL_PATH" \
  --adapter "$RETRIEVAL_LATE_DIR/best" \
  --index-dir "$RETRIEVAL_INDEX_DIR" \
  --sensitivity-catalog "$SENSITIVITY_CATALOG" \
  --output-dir "$SECURE_RAG_DIR/eval" \
  --max-queries 10 \
  --allow-real-api
```

`sensitive_original_exposure` 和 `missing_catalog_exposure` 必须为 0。

## 12. 中断恢复

Sensitivity 同阶段恢复：

```bash
"$VLM_PYTHON" scripts/train_sensitivity.py ... \
  --resume "$SENSITIVITY_FINETUNE_DIR/last.pt"
```

Retrieval late 恢复：

```bash
"$VLM_PYTHON" scripts/train_retrieval_late.py ... \
  --resume "$RETRIEVAL_LATE_DIR/last"
```

新的 `last/training_state.pt` 会恢复 optimizer、scheduler、epoch、global step 和 memory queue。历史旧模型没有该文件，必须使用 `--init-adapter`，不能用 `--resume`。

阶段切换不能使用 `--resume`：

- Sensitivity 冻结阶段 → 解冻阶段使用 `--init-checkpoint`；
- Retrieval global → 首次 late 使用 `--init-global`；已有旧 late 模型进入快速对称微调使用 `--init-adapter`。

## 13. 最终检查清单

- [ ] 数据根包含完整页面、OCR、QA 和 `desensitized`
- [ ] sensitivity / retrieval manifest 统计正确
- [ ] 三个 split 文档零交叉
- [ ] 依赖、CUDA、BF16 和 checkpoint 分片完整
- [ ] 四类任务都完成 5090 显存 profile
- [ ] 正式 batch 的 reserved 峰值不超过 28GB
- [ ] Sensitivity 阶段 2 从阶段 1 初始化
- [ ] 分类 checkpoint 选择策略满足目标 Recall
- [ ] Global hard negatives 排除了同文档页面
- [ ] Late 模型继承 global LoRA 且没有嵌套适配器
- [ ] 索引分 shard 构建，GPU 不常驻完整语料
- [ ] test 指标与 smoke/tiny 指标分开记录
- [ ] 损坏图片和异常记录已审计
- [ ] val 校准阈值固定后再计算 test 指标
- [ ] 全量 sensitivity catalog 覆盖 12,767 页
- [ ] adapter / base model / index 指纹一致
- [ ] 外部生成 API 只接收非敏感原图或已批准脱敏副本
- [ ] `sensitive_original_exposure == 0`
- [ ] API Key 只由环境变量注入，未进入代码、配置和日志
