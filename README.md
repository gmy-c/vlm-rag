# VLM-RAG：基于页面图像的视觉检索增强生成

<div align="center">

**面向复杂企业文档的页面级跨模态检索与多页视觉问答原型**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![ColPali](https://img.shields.io/badge/Retriever-ColPali-green.svg)](https://huggingface.co/vidore/colpali-v1.3-merged)

</div>

## 项目简介

传统文档 RAG 通常依赖“OCR → 文本切分 → 文本向量检索”。当输入包含表格、图表、签章、多栏排版或手写内容时，OCR 错误和版面信息丢失会继续影响召回与生成。

本项目将文档页作为基本检索单元：文本问题和页面图像分别经过文本塔与视觉塔，被投影到同一向量空间；系统检索 Top-K 页面后，调用视觉语言模型逐页提取候选答案，再结合检索分数与模型置信度完成融合。主检索链路不依赖 OCR，OCR 数据仅可作为数据筛选或对照实验的辅助信息。

本仓库目前是研究原型，不是已经完成生产验收的系统。代码中同时存在新版 ColPali 主链路和早期轻量 Demo 的遗留入口；使用前请先阅读“当前实现状态与已知限制”。

## 系统架构

### 1. 离线页面编码

```text
文档页面图像 + 页面元数据
        │
        ▼
冻结的 SigLIP ViT
        │
        ▼
选定隐藏层 [0, 8, 16, 23]
        │
        ▼
可学习 Softmax 权重融合
        │
        ▼
Patch 均值池化 → 视觉投影头 → L2 归一化
        │
        ▼
页面向量矩阵 [N, 768]
```

当前 `DualTowerRetriever` 将页面向量保存在内存中的 PyTorch Tensor 中。仓库尚未把新版 ColPali 检索器接入可持久化的 FAISS、Milvus 或 Elasticsearch 索引。

### 2. 在线问题检索

```text
用户问题
   │
   ▼
Gemma 文本塔 + LoRA
   │
   ▼
Masked Mean Pooling → 文本投影头 → L2 归一化
   │
   ▼
问题向量 [768]
   │
   ├── 与页面向量做内积（等价于余弦相似度）
   ▼
Top-K 页面
```

训练阶段使用 InfoNCE：同一 batch 中第 `i` 个问题与第 `i` 个页面构成正样本，其余页面作为 batch 内负样本。视觉主干保持冻结，LoRA、隐藏层融合权重和两个投影头参与训练。

### 3. 多页视觉问答

```text
用户问题 + Top-K 页面
          │
          ▼
逐页调用 Doubao Vision API
          │
          ▼
{relevant, answer, evidence_quote, confidence}
          │
          ▼
combined_score = retrieval_score × vlm_confidence
          │
          ▼
归一化后的同答案合并分数
          │
          ▼
最终答案 + 证据页面 ID + 融合置信度
```

当前最终 `Answer` 保存答案、证据页面 ID 和融合置信度；逐页返回的 `evidence_quote` 尚未传递到最终结果。

## 核心实现

- **页面级跨模态检索**：`ColPaliDualEncoder` 分别编码问题与页面图像，并输出 L2 归一化的 768 维全局向量。
- **隐藏层加权池化**：从视觉塔选取 `[0, 8, 16, 23]` 层，使用可学习 Softmax 权重融合，再对图像 patch 做均值池化。
- **参数高效训练**：视觉塔冻结；文本塔通过 LoRA 微调，同时训练隐藏层权重和投影头。
- **InfoNCE 对比学习**：利用 batch 内负样本构造问题—页面对比目标。
- **逐页视觉推理**：Top-K 页面分别调用视觉模型，避免直接拼接多页导致分辨率下降和注意力稀释。
- **分数加权融合**：按“检索相似度 × VLM 置信度”累计相同答案的分数。
- **图像拼接基线**：`ImageStitchingGenerator` 将 Top-K 页面纵向拼接，用于与逐页推理方案比较。
- **文档级数据划分**：`split_by_document()` 按 `doc_id` 划分训练、验证和测试集，降低同文档跨集合泄漏的风险。

## 当前实现状态与已知限制

| 模块或入口 | 当前状态 |
| --- | --- |
| `src/vlm_rag/encoders.py` | 已实现 ColPali 双塔、隐藏层融合、LoRA 接入和 InfoNCE |
| `src/vlm_rag/retriever.py` | 已实现基于内存 Tensor 的批量页面编码与 Top-K 检索 |
| `src/vlm_rag/generator.py` | 已实现逐页 API 调用、重试、结构化解析和答案融合 |
| `src/vlm_rag/baselines.py` | 已实现图像拼接生成基线；未实现 OCR-RAG、SigLIP 与原生 ColPali MaxSim 基线 |
| `src/vlm_rag/data.py` | 已实现模拟数据构建、DocVQA 加载和文档级划分 |
| `scripts/test_integration_generator.py` | 当前可在无 GPU、无 API Key 的环境中运行 Mock 与安全检查 |
| `scripts/test_integration.py` | 当前会因测试期望 `colpali-v1.2`、配置使用 `v1.3-merged` 而失败 |
| `scripts/run_demo.py`、`build_index.py`、`train_retriever.py`、`evaluate.py`、统一 CLI | 仍引用已移除的 `HashingVLMEncoder`、`EncoderConfig`、`train_retriever` 或 `evaluate_method`，暂不可作为有效入口 |
| `scripts/train_colpali.py` | 是新版训练入口，但仍硬编码 `colpali-v1.2`，未完整使用 YAML 中的模型配置 |
| `scripts/evaluate_generator.py` | 是新版生成评估入口，需要 GPU、DocVQA 和有效 API Key；真实 API 链路尚未在本仓库结果中完成复现 |

另外需要注意：

1. `configs/config.yaml` 中的 `embedding_dim: 384` 属于旧轻量链路配置；新版 `ColPaliDualEncoderConfig.proj_dim` 默认是 `768`，训练入口目前没有读取该字段。
2. `configs/config.yaml` 使用 `vidore/colpali-v1.3-merged`，但 `ProjectConfig` 默认值和 `train_colpali_retriever()` 仍使用 `vidore/colpali-v1.2`。
3. 配置中的 Doubao `base_url` 已包含 `/api/v3`，而 `generator.py` 又追加 `/api/v3/chat/completions`。真实调用前应统一 URL 拼接规则，避免出现重复路径。
4. `index_store.py` 是旧轻量索引实现，依赖已经移除的哈希编码器；它不是当前 ColPali 检索器的持久化索引。
5. 训练日志写入、短数据集验证频率和最佳模型保存逻辑仍需端到端验证，不应将现有 checkpoint 元数据视为正式实验结果。

## 环境要求

- Python 3.10+
- 完整模型训练与评估建议使用 Linux、CUDA 和 NVIDIA GPU
- ColPali 训练建议至少准备 24 GB 显存；实际占用取决于 batch size、图像尺寸和依赖版本
- 逐页生成评估需要可访问 Doubao Vision API
- 数据使用方需自行确认文档授权、脱敏要求和外部 API 传输合规性

`requirements.txt` 包含完整模型栈，其中 `bitsandbytes`、`flash-attn` 和 CUDA 版本需要与操作系统、显卡驱动和 PyTorch 匹配。Windows 环境通常不适合作为完整训练环境。

## 安装

```bash
git clone https://github.com/gmy-c/vlm-rag.git
cd vlm-rag
python -m pip install -r requirements.txt
```

如果只检查生成器的 Mock 融合与安全逻辑，可先安装轻量依赖：

```bash
python -m pip install Pillow requests
python scripts/test_integration_generator.py
```

该测试不会发起真实 Doubao API 请求。

## 数据准备

本项目的数据结构适配 DocVQA SP-DocVQA Task 1。下载数据后可整理为：

```text
data/
├── docvqa_extracted/
│   ├── train_v1.0_withQT.json
│   ├── val_v1.0_withQT.json
│   └── test_v1.0.json
├── docvqa_images/
└── ocr/                         # 可选，不进入主检索链路
```

加载示例：

```python
from pathlib import Path

from vlm_rag.data import load_docvqa_dataset, split_by_document

pages, queries = load_docvqa_dataset(
    Path("data/docvqa_extracted/train_v1.0_withQT.json"),
    Path("data/docvqa_images"),
    split_name="train",
)

splits = split_by_document(
    pages,
    queries,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
)

train_pages, train_queries = splits["train"]
val_pages, val_queries = splits["val"]
test_pages, test_queries = splits["test"]
```

`load_docvqa_dataset()` 当前只取每条标注的第一个标准答案，并通过 `question_types` 做粗粒度文档类型推断；这两点在正式实验中应按评估协议进一步完善。

## 训练与评估

### ColPali 检索器训练

新版训练入口为：

```bash
python scripts/train_colpali.py \
  --config configs/config.yaml \
  --data-root data \
  --train-qa docvqa_extracted/train_v1.0_withQT.json \
  --images-dir docvqa_images \
  --epochs 5 \
  --batch_size 8 \
  --device cuda
```

这里显式传入相对于 `--data-root` 的路径，是为了避免脚本默认参数再次拼接 `data/`。在正式训练前，还应先统一 `v1.2` 与 `v1.3-merged` 的模型配置，并修正“已知限制”中列出的训练问题。

### 生成器评估

Linux/macOS：

```bash
export DOUBAO_API_KEY="your-key"
python scripts/evaluate_generator.py \
  --data-root data \
  --sample 10 \
  --top-k 3 \
  --device cuda
```

Windows PowerShell：

```powershell
$env:DOUBAO_API_KEY = "your-key"
python scripts/evaluate_generator.py `
  --data-root data `
  --sample 10 `
  --top-k 3 `
  --device cuda
```

在修正 API URL 拼接前，不应将此命令视为可直接复现的完整评估入口。生成评估会把页面图像发送到所配置的外部 API；请勿直接使用未经授权或未脱敏的企业文档。

## 配置说明

主要配置位于 `configs/config.yaml`：

```yaml
top_k: 3
temperature: 0.07
epochs: 5

batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 5.0e-5

colpali_model: "vidore/colpali-v1.3-merged"
lora_rank: 32
lora_alpha: 32
selected_layers_str: "0,8,16,23"

generator:
  backend: "doubao"
  doubao_model: "doubao-seed-1-6-vision-250815"
  timeout: 30
```

API Key 不写入配置文件，由 `ProjectConfig.get_api_key()` 从 `DOUBAO_API_KEY` 环境变量读取。当前代码提供超时和最多三次请求尝试，但没有完整的审计、访问控制、密钥轮换或生产级日志脱敏机制。

## 项目结构

```text
.
├── README.md
├── PROJECT_GUIDE.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── docs/
│   └── technical_report.md
├── scripts/
│   ├── train_colpali.py
│   ├── evaluate_generator.py
│   ├── test_integration.py
│   ├── test_integration_generator.py
│   └── ...                         # 早期轻量 Demo 入口，见状态表
└── src/vlm_rag/
    ├── config.py
    ├── data.py
    ├── encoders.py
    ├── retriever.py
    ├── training.py
    ├── generator.py
    ├── baselines.py
    ├── metrics.py
    ├── index_store.py
    ├── pipeline.py
    └── workflows.py
```

## 指标口径

- `MRR@10`：正确证据页首次出现排名的倒数均值。
- `Recall@K`：Top-K 中至少命中一个正例页面的问题比例。
- `EM`：预测答案与标准答案归一化后完全一致的比例。
- `Accuracy`：当前实现与 EM 使用相同计算逻辑，因此两者数值相同。

项目文档中出现的 `MRR@10 = 77.91`、`Top-3 Accuracy = 56.12` 和 `121 ms/page` 等数字属于原始项目目标或方案参考值，不是由当前仓库可复现脚本生成的实测结果。正式报告应同时给出数据划分、样本量、硬件、模型版本、随机种子和原始输出文件。

## 安全与隐私

- API Key 仅从环境变量读取，不应提交到代码、YAML、日志或实验输出。
- HTTP 请求设置了超时和重试；请求失败时生成器返回安全兜底结果。
- 页面图像会以 Base64 形式发送给配置的 Doubao 接口，敏感数据必须在发送前完成授权与脱敏。
- 当前 `base_url` 可配置，但代码未实现域名白名单；“只会发送到某一固定域名”并不是现有代码能够强制保证的安全属性。
- 仓库中的安全测试属于静态检查和 Mock 测试，不能替代渗透测试、合规评审或生产审计。

## 参考资料

- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
- [DocVQA: A Dataset for VQA on Document Images](https://www.docvqa.org/)
- [PaliGemma](https://ai.google.dev/gemma)
- [ColPali v1.3](https://huggingface.co/vidore/colpali-v1.3-merged)
- [火山引擎方舟 API 文档](https://www.volcengine.com/docs/82379)
