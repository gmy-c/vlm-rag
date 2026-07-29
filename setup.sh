#!/usr/bin/env bash
# ============================================================
# VLM-RAG 服务器一键环境配置脚本
# 使用方法: bash setup.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  VLM-RAG 环境配置"
echo "  项目路径: $SCRIPT_DIR"
echo "============================================================"

# ── 1. 环境变量检查 ──
echo ""
echo "[1/4] 检查环境变量..."

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "  已从 .env.example 创建 .env，请编辑填入 DOUBAO_API_KEY"
        echo "  vim .env"
    fi
else
    echo "  .env 已存在"
fi

# shellcheck disable=SC1090
source .env 2>/dev/null || true

if [ -z "${DOUBAO_API_KEY:-}" ] || [ "$DOUBAO_API_KEY" = "your-doubao-api-key-here" ]; then
    echo "  ⚠ 警告: DOUBAO_API_KEY 未设置！评估脚本将无法调用 API。"
    echo "  编辑 .env 文件并填入真实 Key，然后重新运行本脚本。"
else
    KEY_LEN=${#DOUBAO_API_KEY}
    echo "  ✓ DOUBAO_API_KEY 已设置 (${KEY_LEN} 字符)"
fi

# ── 2. Python 环境 ──
echo ""
echo "[2/4] 检查 Python 环境..."

PYTHON=""
for candidate in python3 python python3.10 python3.11 python3.12 python3.13; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ✗ 错误: 未找到 Python！请安装 Python 3.10+"
    exit 1
fi

PY_VER=$("$PYTHON" --version 2>&1)
echo "  ✓ $PY_VER ($PYTHON)"

# ── 3. CUDA 检查 ──
echo ""
echo "[3/5] 检查 CUDA 环境..."

if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | cut -d' ' -f1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    export GPU_NAME
    echo "  GPU:           $GPU_NAME"
    echo "  Driver:        $DRIVER_VER"
    echo "  CUDA:          $CUDA_VER"

    # RTX 5090 检查
    if echo "$GPU_NAME" | grep -qi "5090"; then
        echo ""
        echo "  ⚠ 检测到 RTX 5090 (Blackwell 架构)！"
        echo "  RTX 5090 需要 CUDA >= 12.8 + PyTorch 2.7+"
        if [ "${CUDA_VER:-0}" \< "12.8" ] 2>/dev/null || [ "$CUDA_VER" = "12.8" ]; then
            :
        else
            echo "  当前 CUDA $CUDA_VER 可能不兼容，需要 >= 12.8"
        fi
    fi
else
    echo "  ⚠ nvidia-smi 不可用，跳过 GPU 检测"
fi

# ── 4. 安装依赖 ──
echo ""
echo "[4/5] 安装 Python 依赖..."

if [ "${1:-}" = "--no-install" ]; then
    echo "  跳过安装 (--no-install)"
else
    echo "  升级 pip ..."
    "$PYTHON" -m pip install --upgrade pip -q

    # RTX 5090: 从 cu128 索引安装 PyTorch 2.7+
    if echo "${GPU_NAME:-}" | grep -qi "5090"; then
        echo "  RTX 5090 检测到，使用 CUDA 12.8 PyTorch nightly ..."
        "$PYTHON" -m pip install --pre torch torchvision \
            --index-url https://download.pytorch.org/whl/nightly/cu128
        echo ""
        echo "  ▶ 说明: 使用了 PyTorch nightly build (CUDA 12.8)"
        echo "    如果 PyTorch 2.7 稳定版已发布，可改用:"
        echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    else
        echo "  安装 PyTorch (CUDA 12.x) ..."
        "$PYTHON" -m pip install torch torchvision
    fi

    echo "  安装其余依赖 ..."
    "$PYTHON" -m pip install -r requirements.txt

    # flash-attn: 尝试安装，失败则跳过（会自动降级到 SDPA）
    echo ""
    echo "  尝试安装 flash-attn (RTX 5090 需要从源码编译) ..."
    if "$PYTHON" -m pip install flash-attn --no-build-isolation 2>/dev/null; then
        echo "  ✓ flash-attn 安装成功"
    else
        echo "  ℹ flash-attn 安装失败，将使用 PyTorch SDPA 作为替代"
        echo "    这对训练不影响，只是注意力计算会稍慢 (~5-10%)"
    fi

    echo "  ✓ 依赖安装完成"
fi

# ── 5. 准备数据 ──
echo ""
echo "[5/5] 检查数据..."

# DocVQA 数据检查
if [ -d "data/docvqa_extracted" ] && [ -d "data/docvqa_images" ]; then
    IMG_COUNT=$(find data/docvqa_images -name "*.png" 2>/dev/null | wc -l)
    echo "  ✓ DocVQA 数据已就绪 ($IMG_COUNT 张图片)"
else
    echo "  ⚠ DocVQA 数据未找到。请将数据放入:"
    echo "    data/docvqa_extracted/  → Q&A JSON 文件"
    echo "    data/docvqa_images/     → 页面 PNG 图片"
    echo ""
    echo "  下载方法:"
    echo "    1. 访问 https://www.docvqa.org/datasets"
    echo "    2. 下载 Task 1: Single Page Document Visual Question Answering"
    echo "    3. 解压 Q&A JSON 到 data/docvqa_extracted/"
    echo "    4. 解压图片到 data/docvqa_images/"
fi

# ColPali 权重检查
if [ -d "models/colpali_retriever" ]; then
    echo "  ✓ ColPali 训练权重已存在"
else
    echo "  ℹ ColPali 权重将在首次运行 train_colpali.py 时自动从 HuggingFace 下载"
    echo "    (~11GB, 下载时间取决于网速)"
    echo "    如需预下载: python -c \"from colpali_engine.models import ColPali; ColPali.from_pretrained('vidore/colpali-v1.3-merged')\""
fi

# ── 验证 ──
echo ""
echo "============================================================"
echo "  配置完成！"
echo "============================================================"
echo ""
echo "  验证安装:"
echo "    $PYTHON scripts/test_integration.py"
echo "    $PYTHON scripts/test_integration_generator.py"
echo ""
echo "  训练 ColPali 检索器:"
echo "    $PYTHON scripts/train_colpali.py"
echo ""
echo "  评估生成模块 (需 API Key):"
echo "    $PYTHON scripts/evaluate_generator.py --sample 10"
echo ""
echo "  构建合成数据集（快速测试）:"
echo "    $PYTHON scripts/build_dataset.py"
echo ""
echo "============================================================"
