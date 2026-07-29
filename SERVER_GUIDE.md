# 服务器部署与运行指南

## 一、你需要上传到服务器的文件

```
服务器项目根目录 (例如 /root/autodl-tmp/aiproject)
│
├── 从 GitHub clone 的代码（不含 data/ 和 checkpoint/）
├── checkpoint/          ← 你从百度网盘下载的 ColPali 权重 (~5.5GB)
└── data/
    ├── docvqa_extracted/    ← Q&A JSON 文件
    │   ├── train_v1.0_withQT.json
    │   ├── val_v1.0_withQT.json
    │   └── test_v1.0.json
    └── docvqa_images/       ← 页面 PNG 图片 (~12,767 张)
```

---

## 二、第一步：拉取代码

```bash
# SSH 登录服务器后
cd /root/autodl-tmp   # 或你的工作目录

# 从 GitHub clone
git clone https://github.com/gmy-c/vlm-rag.git aiproject
cd aiproject
```

---

## 三、第二步：放置数据和权重

```bash
# === 权重 ===
# 用 bypy 下载到项目目录
bypy down /checkpoint/ ./
# 确认: ls checkpoint/ 应看到 model-00001-of-00002.safetensors 等文件

# === DocVQA 数据 ===
# 放到 data/ 下
mkdir -p data/docvqa_extracted data/docvqa_images

# 上传 Q&A JSON（用 scp 或 bypy）
# scp train_v1.0_withQT.json server:/root/autodl-tmp/aiproject/data/docvqa_extracted/
# scp val_v1.0_withQT.json   server:/root/autodl-tmp/aiproject/data/docvqa_extracted/
# scp test_v1.0.json         server:/root/autodl-tmp/aiproject/data/docvqa_extracted/

# 上传图片
# scp *.png server:/root/autodl-tmp/aiproject/data/docvqa_images/
```

---

## 四、第三步：环境配置

```bash
# 确保在项目根目录
cd /root/autodl-tmp/aiproject

# 运行一键配置脚本
bash setup.sh
```

这个脚本会自动：
1. 从 `.env.example` 创建 `.env`
2. 安装 `requirements.txt` 所有依赖
3. 检查数据和权重是否就位

---

## 五、第四步：验证环境（不上 GPU 也可以跑）

```bash
# AST 结构测试（不需要 GPU，不需要 API Key，秒级完成）
python scripts/test_integration.py

# 看到 "All tests passed! OK" 即表示代码没问题
```

---

## 六、第五步：训练检索器（需要 GPU）

```bash
# 确认 checkpoint 存在
ls checkpoint/model-00001-of-00002.safetensors && echo "权重就绪"

# 开始训练
python scripts/train_colpali.py
```

**预期情况：**
- 首次运行会加载本地 checkpoint 权重（不走网络）
- ColPali 3B 模型加载到 GPU（约 11GB 显存）
- 训练参数：LoRA r=32，只训约 20M 参数，视觉塔冻结
- batch_size=8 + gradient_accumulation=4，有效 batch=32
- 4090 24GB 可跑，5090 更从容

**输出的内容：**
```
models/colpali_retriever/
├── lora/              ← LoRA 权重
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── head_weights.pt    ← 投影头 + 层权重
├── retriever_config.json
└── training_log.csv   ← 每 50 步的 loss/MRR/Recall
```

---

## 七、第六步：预测评估（需要 GPU + 豆包 API Key）

```bash
# 设置 API Key
export DOUBAO_API_KEY="你的豆包key"

# 小样本测试（10 条，验证链路通）
python scripts/evaluate_generator.py --sample 10

# 确认无误后，全量评估
python scripts/evaluate_generator.py
```

**输出的内容：**
```
outputs/generator_metrics.csv   ← per_page vs stitched 对比
```

---

## 八、常见问题速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `colpali-engine` 装不上 | flash-attn 编译失败 | `pip install flash-attn --no-build-isolation` 后再装 colpali-engine |
| `CUDA out of memory` | 显存不够 | 改 `config.yaml` 里 `batch_size: 2`、`gradient_accumulation_steps: 16` |
| checkpoint 找不到 | 路径不对 | 确认 `checkpoint/` 在项目根目录，里面有 `.safetensors` 文件 |
| DocVQA 数据报错 | 路径/格式不对 | 确认 `data/docvqa_images/` 下有 .png 文件，`data/docvqa_extracted/` 下有 .json |
| HuggingFace 要下载模型 | checkpoint 没被识别 | 检查 `ls checkpoint/config.json` 是否存在 |
| 豆包 API 报错 | Key 未设置 | `echo $DOUBAO_API_KEY` 确认有值 |
| 豆包 API 返回空 | 网络不通或余额不足 | 先 `curl` 测试 API 连通性 |

---

## 九、快捷命令汇总

```bash
# 从头到尾一键（训练完成后手动设置 API Key 再跑预测）
git clone https://github.com/gmy-c/vlm-rag.git && cd aiproject
bash setup.sh
python scripts/test_integration.py
python scripts/train_colpali.py

# 预测
export DOUBAO_API_KEY="你的key"
python scripts/evaluate_generator.py --sample 10
```
