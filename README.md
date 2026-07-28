# VLM-RAG 页面图像检索增强生成演示项目

这是一个可直接运行的企业内部多模态大模型算法原型工程，主题是“基于页面图像的视觉检索增强生成（VLM-RAG）”。

本项目为了便于本地快速跑通，不依赖在线大模型或 GPU。代码用标准库实现了一个轻量版流程：

- 自动生成六类企业文档页面图像样例，覆盖合同、报表、PPT、单据、手册、制度
- 使用文本 Query 与页面图像统一编码
- 基于 InfoNCE 风格目标实现双塔检索器训练/权重搜索
- Top-K 页面检索、分数加权、多页答案融合
- 输出 MRR@10、Recall@K、EM、Accuracy 和基线对比表

真实生产环境中，可将 `src/vlm_rag/encoders.py` 替换为 MiniCPM-V、SigLIP、ColPali 或 GPT-4o 相关编码/生成接口。

## 快速运行

环境要求：

- Python 3.10+
- 无需联网
- 无需 GPU
- 无需额外安装第三方依赖

```bash
python3 scripts/run_demo.py
```

运行后会生成：

- `data/sample_pages/*.svg`：模拟企业图文页面
- `outputs/retrieval_results.json`：检索与问答结果
- 控制台指标：MRR@10、Recall@3、EM、Accuracy

```bash
python3 scripts/run_demo.py --data-dir data --output-dir outputs
```

## 完整实验流程

```bash
python3 scripts/build_dataset.py
python3 scripts/build_index.py
python3 scripts/train_retriever.py
python3 scripts/evaluate.py
python3 scripts/run_demo.py
```

各脚本作用：

- `scripts/build_dataset.py`：生成 18 页模拟企业图文页面、24 条问答样本和 train/dev/test 划分。
- `scripts/build_index.py`：构建并落盘页面向量索引，输出 `indexes/page_vectors.json` 和 `indexes/index_metadata.json`。
- `scripts/train_retriever.py`：用 InfoNCE 目标对隐藏层池化权重做轻量训练/搜索，输出 `models/retriever_config.json` 和 `models/training_log.csv`。
- `scripts/evaluate.py`：评估 VLM-RAG、OCR-RAG、SigLIP、ColPali 四种方案，输出 `outputs/metrics_report.csv`。
- `scripts/run_demo.py`：跑端到端检索增强生成，输出 `outputs/retrieval_results.json`。

也可以使用统一 CLI：

```bash
python3 scripts/vlm_rag_cli.py all
```

## 当前模拟实验结果

`scripts/evaluate.py` 会生成如下对比维度：

| method    | 含义                                   |
| --------- | -------------------------------------- |
| `vlm_rag` | 页面图像 VLM-RAG 主方案                |
| `ocr_rag` | OCR 文本链路模拟基线                   |
| `siglip`  | 全局图文向量检索模拟基线               |
| `colpali` | 版式感知 late-interaction 检索模拟基线 |

本项目中的基线是离线可运行模拟，用来展示实验框架和指标计算。真实项目中可以在 `src/vlm_rag/baselines.py` 替换为实际模型调用。

## 项目结构

```text
.
├── README.md
├── configs
│   └── config.yaml
├── docs
│   └── technical_report.md
├── scripts
│   ├── build_index.py
│   ├── build_dataset.py
│   ├── evaluate.py
│   ├── run_demo.py
│   ├── train_retriever.py
│   └── vlm_rag_cli.py
└── src
    └── vlm_rag
        ├── baselines.py
        ├── cli.py
        ├── config.py
        ├── dataset_split.py
        ├── __init__.py
        ├── data.py
        ├── encoders.py
        ├── generator.py
        ├── index_store.py
        ├── logging_utils.py
        ├── metrics.py
        ├── pipeline.py
        ├── retriever.py
        ├── training.py
        └── workflows.py
```

## 文档

- 技术方案文档：`docs/technical_report.md`
